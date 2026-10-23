"""Training script.  Usage:
    python src/train.py --config proposed
    python src/train.py --config all            # trains every config sequentially
Configs: cnn, lstm, bilstm, resnet1d, cnn_bilstm (ablation baseline), abl_A, abl_AB, proposed
"""
import os, time, argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from preprocessing import get_splits, make_loader, class_alpha, augment_batch
from model import build_model
from utils import set_seed, count_params, plot_curves, RESULTS, MODELS

# Ablation components:  A = multi-scale SE-CNN,  B = temporal attention,  C = focal loss + augmentation
CONFIGS = {
    "cnn":        dict(model="cnn",      kw={}, loss="ce", augment=False),
    "lstm":       dict(model="lstm",     kw={}, loss="ce", augment=False),
    "bilstm":     dict(model="bilstm",   kw={}, loss="ce", augment=False),
    "resnet1d":   dict(model="resnet1d", kw={}, loss="ce", augment=False),
    "cnn_bilstm": dict(model="hybrid",   kw=dict(multiscale=False, se=False, attn=False), loss="ce",    augment=False),
    "abl_A":      dict(model="hybrid",   kw=dict(multiscale=True,  se=True,  attn=False), loss="ce",    augment=False),
    "abl_AB":     dict(model="hybrid",   kw=dict(multiscale=True,  se=True,  attn=True),  loss="ce",    augment=False),
    "proposed":   dict(model="hybrid",   kw=dict(multiscale=True,  se=True,  attn=True),  loss="focal", augment=True),
}

HP = dict(lr=1e-3, batch_size=256, epochs=40, weight_decay=1e-4, dropout=0.3, hidden=64,
          patience=8, grad_clip=1.0, focal_gamma=2.0, seed=42)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, y):
        logpt = F.log_softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
        loss = -((1 - logpt.exp()) ** self.gamma) * logpt
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[y]
        return loss.mean()


@torch.no_grad()
def predict(model, loader, device, criterion=None):
    model.eval(); logits_all, y_all, loss_sum, n = [], [], 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        if criterion is not None:
            loss_sum += criterion(out, yb).item() * len(yb)
        n += len(yb); logits_all.append(out.cpu()); y_all.append(yb.cpu())
    return torch.cat(logits_all), torch.cat(y_all), loss_sum / max(n, 1)


def run_training(cfg_name, overrides=None, save=True, verbose=True):
    cfg = CONFIGS[cfg_name]
    hp = {**HP, **(overrides or {})}
    set_seed(hp["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (Xtr, ytr), (Xva, yva), _ = get_splits()
    tr_loader = make_loader(Xtr, ytr, hp["batch_size"], True)
    va_loader = make_loader(Xva, yva, hp["batch_size"] * 2, False)

    model = build_model(cfg["model"], hidden=hp["hidden"], dropout=hp["dropout"], **cfg["kw"]).to(device)
    n_params = count_params(model)
    criterion = FocalLoss(class_alpha(ytr), hp["focal_gamma"]) if cfg["loss"] == "focal" else nn.CrossEntropyLoss()
    val_criterion = nn.CrossEntropyLoss()        # same validation loss for every config -> comparable curves
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"])

    best_f1, best_state, bad, hist = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(1, hp["epochs"] + 1):
        model.train(); tl, correct, n = 0.0, 0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            if cfg["augment"]:
                xb = augment_batch(xb)
            opt.zero_grad(set_to_none=True)
            out = model(xb); loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            opt.step()
            tl += loss.item() * len(yb); correct += (out.argmax(1) == yb).sum().item(); n += len(yb)
        sched.step()
        lg, yv, vl = predict(model, va_loader, device, val_criterion)
        pv = lg.argmax(1).numpy()
        vf1 = f1_score(yv.numpy(), pv, average="macro", zero_division=0)
        hist.append(dict(epoch=ep, train_loss=tl / n, train_acc=correct / n, val_loss=vl,
                         val_acc=float((pv == yv.numpy()).mean()), val_macro_f1=vf1, lr=opt.param_groups[0]["lr"]))
        if verbose:
            h = hist[-1]; print(f"[{cfg_name}] ep {ep:02d} loss {h['train_loss']:.4f} acc {h['train_acc']:.4f} | "
                                f"val loss {h['val_loss']:.4f} acc {h['val_acc']:.4f} F1 {vf1:.4f}")
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                if verbose: print(f"Early stopping at epoch {ep}")
                break
    train_time = time.time() - t0
    result = dict(config=cfg_name, best_val_macro_f1=best_f1, epochs_run=len(hist),
                  trainable_params=n_params, train_time_s=round(train_time, 1))
    if save:
        os.makedirs(RESULTS, exist_ok=True); os.makedirs(MODELS, exist_ok=True)
        torch.save(dict(state_dict=best_state, cfg=cfg, hp=hp, **result), os.path.join(MODELS, f"{cfg_name}.pt"))
        hdf = pd.DataFrame(hist); hdf.to_csv(os.path.join(RESULTS, f"{cfg_name}_history.csv"), index=False)
        plot_curves(hdf, cfg_name)
        with open(os.path.join(RESULTS, f"{cfg_name}_train_info.json"), "w") as f:
            json.dump({**result, "hyperparameters": hp}, f, indent=2)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="proposed", choices=list(CONFIGS) + ["all"])
    for k, v in HP.items():
        ap.add_argument(f"--{k}", type=type(v), default=None)
    a = ap.parse_args()
    ov = {k: getattr(a, k) for k in HP if getattr(a, k) is not None}
    for c in (CONFIGS if a.config == "all" else [a.config]):
        print(run_training(c, ov))
