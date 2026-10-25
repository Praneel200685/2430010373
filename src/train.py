"""Training engine and CLI.

Examples
--------
python -m src.train --models proposed
python -m src.train --models all --epochs 40
python -m src.train --models proposed --epochs 3 --patience 2      # quick check
"""
import argparse
import json
import time

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from .config import (ALL_BASELINES_AND_ABLATIONS, CONFIGS, DEFAULT_HP, MODELS_DIR, RESULTS_DIR, ensure_dirs)
from .data import class_alpha, download_or_verify_dataset, make_loader, augment_batch, prepare_splits
from .losses import FocalLoss
from .models import build_model, count_params
from .utils import get_device, set_seed


@torch.no_grad()
def predict(model, loader, device, criterion=None):
    model.eval()
    logits_all, y_all, loss_sum, n = [], [], 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        if criterion is not None:
            loss_sum += criterion(out, yb).item() * len(yb)
        n += len(yb)
        logits_all.append(out.cpu())
        y_all.append(yb.cpu())
    return torch.cat(logits_all), torch.cat(y_all), loss_sum / max(n, 1)


def run_training(cfg_name, data, device, overrides=None, save=True, verbose=True):
    cfg = CONFIGS[cfg_name]
    hp = {**DEFAULT_HP, **(overrides or {})}
    set_seed(hp["seed"])

    tr_loader = make_loader(data["X_train"], data["y_train"], hp["batch_size"], shuffle=True)
    va_loader = make_loader(data["X_val"], data["y_val"], hp["batch_size"] * 2, shuffle=False)

    model = build_model(cfg["model"], hidden=hp["hidden"], dropout=hp["dropout"], **cfg["kw"]).to(device)
    n_params = count_params(model)

    criterion = (FocalLoss(class_alpha(data["y_train"]), hp["focal_gamma"])
                 if cfg["loss"] == "focal" else nn.CrossEntropyLoss())
    val_criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"])

    best_f1, best_state, bad, hist = -1.0, None, 0, []
    t0 = time.time()
    if verbose:
        print(f"\n{'=' * 70}\n[*] Training {cfg_name.upper()} | params: {n_params:,}\n{'=' * 70}")

    for ep in range(1, hp["epochs"] + 1):
        model.train()
        tl, correct, n = 0.0, 0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            if cfg["augment"]:
                xb = augment_batch(xb)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            opt.step()
            tl += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(yb)
        sched.step()

        lg, yv, vl = predict(model, va_loader, device, val_criterion)
        pv = lg.argmax(1).numpy()
        vf1 = float(f1_score(yv.numpy(), pv, average="macro", zero_division=0))
        vac = float((pv == yv.numpy()).mean())
        hist.append(dict(epoch=ep, train_loss=tl / n, train_acc=correct / n,
                         val_loss=vl, val_acc=vac, val_macro_f1=vf1, lr=opt.param_groups[0]["lr"]))

        if verbose and (ep % 2 == 0 or ep == 1 or ep == hp["epochs"]):
            print(f"[{cfg_name}] Ep {ep:02d}/{hp['epochs']:02d} | train loss {tl / n:.4f} acc {correct / n:.4f} | "
                  f"val loss {vl:.4f} acc {vac:.4f} macro-F1 {vf1:.4f}")

        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                if verbose:
                    print(f"[*] Early stopping at epoch {ep} (best val macro-F1 {best_f1:.4f})")
                break

    result = dict(config=cfg_name, best_val_macro_f1=best_f1, epochs_run=len(hist),
                  trainable_params=n_params, train_time_s=round(time.time() - t0, 1))

    if save:
        ensure_dirs()
        torch.save(dict(state_dict=best_state, cfg=cfg, hp=hp, **result), MODELS_DIR / f"{cfg_name}.pt")
        pd.DataFrame(hist).to_csv(RESULTS_DIR / f"{cfg_name}_history.csv", index=False)
        with open(RESULTS_DIR / f"{cfg_name}_train_info.json", "w") as f:
            json.dump({**result, "hyperparameters": hp}, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser(description="Train ECG arrhythmia models")
    ap.add_argument("--models", nargs="+", default=["proposed"],
                    help=f"config names or 'all'. Choices: {list(CONFIGS)}")
    ap.add_argument("--epochs", type=int, default=DEFAULT_HP["epochs"])
    ap.add_argument("--patience", type=int, default=DEFAULT_HP["patience"])
    ap.add_argument("--seed", type=int, default=DEFAULT_HP["seed"])
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="smoke-test only: generate fake data if download fails")
    args = ap.parse_args()

    targets = ALL_BASELINES_AND_ABLATIONS if "all" in args.models else args.models
    for t in targets:
        if t not in CONFIGS:
            raise SystemExit(f"Unknown model '{t}'. Choices: {list(CONFIGS)}")

    ensure_dirs()
    device = get_device()
    download_or_verify_dataset(allow_synthetic=args.allow_synthetic)
    data = prepare_splits(seed=args.seed)
    overrides = dict(epochs=args.epochs, patience=args.patience, seed=args.seed)
    for t in targets:
        run_training(t, data, device, overrides=overrides)
    print("\n[✓] Training complete:", targets)


if __name__ == "__main__":
    main()
