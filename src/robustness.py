"""Gaussian-noise robustness benchmark (clean vs. noisy test beats).

python -m src.robustness --models cnn_bilstm proposed
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import MODEL_LABELS, RESULTS_DIR, ensure_dirs
from .data import make_loader, prepare_splits
from .evaluate import load_model
from .train import predict
from .utils import get_device

SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["cnn_bilstm", "proposed"])
    ap.add_argument("--sigmas", nargs="+", type=float, default=SIGMAS)
    args = ap.parse_args()

    ensure_dirs()
    device = get_device()
    data = prepare_splits()
    X, y = data["X_test"], data["y_test"]

    rows = []
    for cfg in args.models:
        model, _ = load_model(cfg, device)
        rng = np.random.RandomState(42)
        for s in args.sigmas:
            Xn = (X + rng.normal(0, s, X.shape)).astype(np.float32)
            lg, yt, _ = predict(model, make_loader(Xn, y, 512, False), device)
            pred, yt = lg.argmax(1).numpy(), yt.numpy()
            rows.append({"Model": MODEL_LABELS.get(cfg, cfg), "Noise Std (σ)": s,
                         "Accuracy": round(float((pred == yt).mean()), 4),
                         "Macro F1-Score": round(float(f1_score(yt, pred, average="macro", zero_division=0)), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "robustness.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
