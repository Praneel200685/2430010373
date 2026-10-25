"""One-factor-at-a-time (OFAT) hyperparameter sweep on the proposed model.

python -m src.tune            # ~13 short runs of 15 epochs each
Writes results/hyperparameter_tuning.csv (values come from real runs only).
"""
import argparse
import pandas as pd

from .config import RESULTS_DIR, ensure_dirs
from .data import download_or_verify_dataset, prepare_splits
from .train import run_training
from .utils import get_device

GRID = {
    "lr": [3e-4, 1e-3, 3e-3],
    "dropout": [0.2, 0.3, 0.5],
    "hidden": [32, 64, 128],
    "weight_decay": [1e-5, 1e-4, 1e-3],
}


def main():
    ap = argparse.ArgumentParser(description="OFAT hyperparameter sweep")
    ap.add_argument("--epochs", type=int, default=15, help="Epochs per run")
    ap.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = ap.parse_args()

    ensure_dirs()
    device = get_device()
    download_or_verify_dataset()
    data = prepare_splits()
    base = dict(epochs=args.epochs, patience=args.patience)

    rows = []
    res = run_training("proposed", data, device, base, save=False, verbose=False)
    rows.append({"Parameter": "default", "Tested Value": "lr=1e-3, drop=0.3, h=64, wd=1e-4", **res})
    for name, values in GRID.items():
        for v in values:
            res = run_training("proposed", data, device, {**base, name: v}, save=False, verbose=False)
            rows.append({"Parameter": name, "Tested Value": str(v), **res})
            print(f"  {name} = {v:<8} -> best val macro-F1 {res['best_val_macro_f1']:.4f}")

    df = pd.DataFrame(rows)[["Parameter", "Tested Value", "best_val_macro_f1", "epochs_run",
                             "trainable_params", "train_time_s"]]
    df.columns = ["Parameter", "Tested Value", "Val Macro-F1", "Epochs", "Parameters", "Time (s)"]
    df.to_csv(RESULTS_DIR / "hyperparameter_tuning.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
