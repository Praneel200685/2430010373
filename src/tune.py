"""Hyperparameter search for the proposed model (validation macro-F1 only; test set never touched).
    python src/tune.py            -> results/hyperparameter_tuning.csv
Grid is run with a shortened schedule (15 epochs) for cost; final model retrains with the best values.
"""
import os, itertools
import pandas as pd
from train import run_training
from utils import RESULTS

GRID = {"lr": [3e-4, 1e-3, 3e-3], "dropout": [0.2, 0.3, 0.5], "hidden": [32, 64, 128], "weight_decay": [1e-5, 1e-4, 1e-3]}

if __name__ == "__main__":
    rows = []
    # one-factor-at-a-time around the default point (keeps cost to ~10 runs instead of 81)
    base = dict(epochs=15, patience=5)
    rows.append({"param": "default", "value": "-", **run_training("proposed", base, save=False, verbose=False)})
    for k, vals in GRID.items():
        for v in vals:
            r = run_training("proposed", {**base, k: v}, save=False, verbose=False)
            rows.append({"param": k, "value": v, **r}); print(k, v, round(r["best_val_macro_f1"], 4))
    df = pd.DataFrame(rows)[["param", "value", "best_val_macro_f1", "epochs_run", "trainable_params", "train_time_s"]]
    df.to_csv(os.path.join(RESULTS, "hyperparameter_tuning.csv"), index=False)
    print(df.to_string(index=False))
