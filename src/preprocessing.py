"""Data loading, preprocessing, augmentation and dataset analysis.

Dataset: MIT-BIH Arrhythmia heartbeats (Kaggle 'ECG Heartbeat Categorization Dataset',
derived from PhysioNet MIT-BIH Arrhythmia Database).
Files expected in data/: mitbih_train.csv, mitbih_test.csv
Each row = 187 samples of one heartbeat (already cropped and scaled to [0,1]) + label (last column).
Classes (AAMI): 0=N normal, 1=S supraventricular ectopic, 2=V ventricular ectopic, 3=F fusion, 4=Q unknown.
"""
import os, argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from utils import ROOT, RESULTS, FIGURES, CLASS_NAMES

DATA_DIR = os.path.join(ROOT, "data")
VAL_SIZE = 0.10   # fraction of the official training file held out for validation (stratified)
SEED = 42


def load_raw(data_dir=DATA_DIR):
    tr = pd.read_csv(os.path.join(data_dir, "mitbih_train.csv"), header=None).values.astype(np.float32)
    te = pd.read_csv(os.path.join(data_dir, "mitbih_test.csv"), header=None).values.astype(np.float32)
    return tr[:, :-1], tr[:, -1].astype(np.int64), te[:, :-1], te[:, -1].astype(np.int64)


def standardize(x):
    """Per-beat z-score normalisation (removes amplitude/offset differences between beats)."""
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)


def get_splits(data_dir=DATA_DIR, val_size=VAL_SIZE, seed=SEED):
    X, y, Xte, yte = load_raw(data_dir)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=val_size, stratify=y, random_state=seed)
    return (standardize(Xtr), ytr), (standardize(Xva), yva), (standardize(Xte), yte)


def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X).unsqueeze(1), torch.from_numpy(y))  # (N, 1, 187)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available())


def class_alpha(y, num_classes=5):
    """Inverse-sqrt-frequency class weights, normalised to mean 1 (used by focal loss)."""
    cnt = np.bincount(y, minlength=num_classes).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(cnt, 1)); w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def augment_batch(x, p=0.5, noise_std=0.05, scale=(0.9, 1.1), max_shift=8):
    """Batched on-device ECG augmentation (training only): Gaussian noise, amplitude scaling, time shift."""
    B, C, L = x.shape
    dev = x.device
    mask = (torch.rand(B, 1, 1, device=dev) < p).float()
    xa = x + torch.randn_like(x) * noise_std
    xa = xa * torch.empty(B, 1, 1, device=dev).uniform_(*scale)
    shift = torch.randint(-max_shift, max_shift + 1, (B, 1), device=dev)
    idx = (torch.arange(L, device=dev)[None, :] + shift).clamp(0, L - 1)
    xa = torch.gather(xa, 2, idx.unsqueeze(1))
    return mask * xa + (1 - mask) * x


# --------------------------------------------------------------- dataset analysis tables (Section 5A)
def analyze():
    X, y, Xte, yte = load_raw()
    (Xtr, ytr), (Xva, yva), (Xt, yt) = get_splits()
    full = np.concatenate([y, yte]); total = len(full)

    d = pd.DataFrame({"class": CLASS_NAMES,
                      "description": ["Normal beat", "Supraventricular ectopic", "Ventricular ectopic",
                                      "Fusion beat", "Unknown / paced"]})
    d["total_samples"] = np.bincount(full, minlength=5)
    d["percentage"] = (100 * d["total_samples"] / total).round(2)
    d.to_csv(os.path.join(RESULTS, "dataset_distribution.csv"), index=False)

    rows = []
    for name, lab in [("Train", ytr), ("Validation", yva), ("Test", yt)]:
        r = {"split": name, "samples": len(lab), "percentage_of_total": round(100 * len(lab) / total, 2)}
        for i, c in enumerate(CLASS_NAMES):
            r[c] = int((lab == i).sum())
        rows.append(r)
    split_df = pd.DataFrame(rows)
    split_df.to_csv(os.path.join(RESULTS, "split_distribution.csv"), index=False)

    cnt = np.bincount(full, minlength=5)
    pd.DataFrame({"class": CLASS_NAMES, "count": cnt,
                  "imbalance_ratio_vs_majority": (cnt.max() / np.maximum(cnt, 1)).round(2)}
                 ).to_csv(os.path.join(RESULTS, "class_imbalance.csv"), index=False)

    pd.DataFrame([
        ["Source", "Official mitbih_train.csv + mitbih_test.csv (Kaggle ECG Heartbeat Categorization)"],
        ["Beat length", "187 samples per beat (pre-segmented in the source dataset)"],
        ["Normalisation", "Per-beat z-score: (x - mean) / (std + 1e-6)"],
        ["Validation split", f"{int(VAL_SIZE * 100)}% of the train file, stratified, seed={SEED}"],
        ["Test set", "Official mitbih_test.csv; never used for training or model selection"],
        ["Augmentation (proposed model only, train only)",
         "p=0.5: Gaussian noise (std 0.05), amplitude scale U(0.9,1.1), time shift +-8 samples"],
    ], columns=["step", "parameters"]).to_csv(os.path.join(RESULTS, "preprocessing_steps.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(CLASS_NAMES, cnt, color="steelblue"); ax[0].set_yscale("log")
    ax[0].set_title("Class distribution (log scale)"); ax[0].set_ylabel("Samples")
    for i, c in enumerate(cnt):
        ax[0].text(i, c, str(c), ha="center", va="bottom")
    for i, c in enumerate(CLASS_NAMES):
        if (yt == i).any():
            ax[1].plot(Xt[yt == i][0], label=c)
    ax[1].set_title("Example standardised beat per class"); ax[1].set_xlabel("Sample"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "dataset_overview.png"), dpi=200); plt.close()
    print(d.to_string(index=False)); print(split_df.to_string(index=False))


if __name__ == "__main__":
    argparse.ArgumentParser(description="Generate dataset analysis tables/figures").parse_args()
    analyze()
