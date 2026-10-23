"""Evaluate trained model(s) on the held-out official test set.  Usage:
    python src/test.py --config proposed
    python src/test.py --config all
    python src/test.py --config proposed --robustness     # noise-robustness analysis (Section 5F)
"""
import os, json, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from preprocessing import get_splits, make_loader
from model import build_model
from train import CONFIGS, predict
from utils import compute_metrics, plot_confusion, plot_roc, RESULTS, MODELS, count_params, CLASS_NAMES


def load_model(cfg_name, device):
    ck = torch.load(os.path.join(MODELS, f"{cfg_name}.pt"), map_location=device, weights_only=False)
    hp, cfg = ck["hp"], ck["cfg"]
    m = build_model(cfg["model"], hidden=hp["hidden"], dropout=hp["dropout"], **cfg["kw"]).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


def evaluate(cfg_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(cfg_name, device)
    _, _, (Xt, yt) = get_splits()
    loader = make_loader(Xt, yt, 512, False)
    t0 = time.time(); logits, y, _ = predict(model, loader, device); dt = time.time() - t0
    prob = F.softmax(logits, 1).numpy(); pred = prob.argmax(1); y = y.numpy()
    summary, per_class, cm = compute_metrics(y, pred, prob)
    summary.update(config=cfg_name, parameters=count_params(model), train_time_s=ck["train_time_s"],
                   epochs_run=ck["epochs_run"], inference_ms_per_sample=round(1000 * dt / len(y), 4))
    per_class.to_csv(os.path.join(RESULTS, f"{cfg_name}_per_class_metrics.csv"), index=False)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(os.path.join(RESULTS, f"{cfg_name}_confusion_matrix.csv"))
    plot_confusion(cm, cfg_name); plot_roc(y, prob, cfg_name)
    with open(os.path.join(RESULTS, f"{cfg_name}_metrics.json"), "w") as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in summary.items()}, f, indent=2)
    print(f"\n=== {cfg_name} (test) ===")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()})
    print(per_class.round(4).to_string(index=False))
    return summary


def robustness(cfg_name, sigmas=(0.0, 0.05, 0.1, 0.2, 0.3)):
    """Accuracy / macro-F1 when Gaussian noise (std in z-score units) is added to test beats."""
    from sklearn.metrics import f1_score
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(cfg_name, device)
    _, _, (Xt, yt) = get_splits()
    rows = []
    g = np.random.RandomState(0)
    for s in sigmas:
        Xn = (Xt + g.normal(0, s, Xt.shape)).astype(np.float32)
        lg, y, _ = predict(model, make_loader(Xn, yt, 512, False), device)
        p = lg.argmax(1).numpy(); y = y.numpy()
        rows.append(dict(config=cfg_name, noise_std=s, accuracy=(p == y).mean(),
                         macro_f1=f1_score(y, p, average="macro", zero_division=0)))
    df = pd.DataFrame(rows); print(df.round(4).to_string(index=False))
    path = os.path.join(RESULTS, "robustness.csv")
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="proposed", choices=list(CONFIGS) + ["all"])
    ap.add_argument("--robustness", action="store_true")
    a = ap.parse_args()
    for c in (CONFIGS if a.config == "all" else [a.config]):
        robustness(c) if a.robustness else evaluate(c)
