"""Dataset download, per-beat standardization, stratified split, loaders, augmentation."""
import os
import shutil
import urllib.request

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .config import TRAIN_CSV, TEST_CSV, ensure_dirs


# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------
def download_or_verify_dataset(allow_synthetic: bool = False):
    """Make sure mitbih_train.csv / mitbih_test.csv exist in data/.

    Order: already present -> KaggleHub -> HuggingFace mirror -> (opt-in) synthetic data.
    Synthetic data is NEVER used unless allow_synthetic=True, so real results are never
    silently produced from fake beats.
    """
    ensure_dirs()
    if TRAIN_CSV.exists() and TEST_CSV.exists():
        print("[✓] Dataset files already present.")
        return

    print("[*] Downloading MIT-BIH Arrhythmia dataset...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("shayanfazeli/heartbeat")
        for name, dst in (("mitbih_train.csv", TRAIN_CSV), ("mitbih_test.csv", TEST_CSV)):
            src = os.path.join(path, name)
            if os.path.exists(src):
                shutil.copy(src, dst)
        if TRAIN_CSV.exists() and TEST_CSV.exists():
            print("[✓] Downloaded via KaggleHub.")
            return
    except Exception as e:  # noqa: BLE001
        print(f"[*] KaggleHub unavailable: {e}")

    try:
        base = "https://huggingface.co/datasets/mstz/mit-bih-arrhythmia/resolve/main/"
        urllib.request.urlretrieve(base + "mitbih_train.csv", TRAIN_CSV)
        urllib.request.urlretrieve(base + "mitbih_test.csv", TEST_CSV)
        print("[✓] Downloaded from HuggingFace mirror.")
        return
    except Exception as e:  # noqa: BLE001
        print(f"[*] Mirror download failed: {e}")
        for p in (TRAIN_CSV, TEST_CSV):
            if p.exists():
                p.unlink()

    if allow_synthetic:
        print("[!] Generating SYNTHETIC data (smoke test only; results are NOT valid).")
        _generate_synthetic()
        return

    raise FileNotFoundError(
        "Could not obtain the dataset. Download 'mitbih_train.csv' and 'mitbih_test.csv' from "
        "https://www.kaggle.com/datasets/shayanfazeli/heartbeat and place them in data/. "
        "For a pipeline smoke test only, re-run with --allow-synthetic."
    )


def _generate_synthetic():
    rng = np.random.RandomState(42)
    t = np.linspace(0, 1.5, 187)
    props = [0.828, 0.025, 0.066, 0.007, 0.074]

    def gen(n):
        counts = [int(p * n) for p in props]
        counts[0] += n - sum(counts)
        X, y = [], []
        for cls, cnt in enumerate(counts):
            for _ in range(cnt):
                p_wave = 0.15 * np.exp(-((t - 0.3) ** 2) / (2 * 0.03 ** 2))
                qrs = (-0.15 * np.exp(-((t - 0.65) ** 2) / (2 * 0.01 ** 2))
                       + 1.2 * np.exp(-((t - 0.7) ** 2) / (2 * 0.015 ** 2))
                       - 0.25 * np.exp(-((t - 0.75) ** 2) / (2 * 0.01 ** 2)))
                t_wave = 0.35 * np.exp(-((t - 1.05) ** 2) / (2 * 0.06 ** 2))
                if cls == 1:
                    p_wave = 0.35 * np.sin(2 * np.pi * 5 * t) * np.exp(-((t - 0.35) ** 2) / (2 * 0.05 ** 2))
                elif cls == 2:
                    qrs = -1.1 * np.exp(-((t - 0.7) ** 2) / (2 * 0.04 ** 2))
                elif cls == 3:
                    qrs = 0.6 * np.exp(-((t - 0.7) ** 2) / (2 * 0.03 ** 2))
                elif cls == 4:
                    qrs = qrs + 0.8 * np.exp(-((t - 0.62) ** 2) / (2 * 0.005 ** 2))
                X.append(np.clip(p_wave + qrs + t_wave + rng.normal(0, 0.03, 187), 0, 1))
                y.append(cls)
        return np.column_stack([np.array(X), np.array(y)])

    pd.DataFrame(gen(20000)).to_csv(TRAIN_CSV, header=False, index=False)
    pd.DataFrame(gen(5000)).to_csv(TEST_CSV, header=False, index=False)


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def load_raw_data():
    tr = pd.read_csv(TRAIN_CSV, header=None).values.astype(np.float32)
    te = pd.read_csv(TEST_CSV, header=None).values.astype(np.float32)
    return tr[:, :-1], tr[:, -1].astype(np.int64), te[:, :-1], te[:, -1].astype(np.int64)


def standardize(x: np.ndarray) -> np.ndarray:
    """Per-beat z-score: removes baseline offset and amplitude differences (Eq. 1)."""
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)


def prepare_splits(seed: int = 42, val_size: float = 0.10) -> dict:
    """Official test file is kept untouched; 10% of official train becomes a stratified validation set."""
    X_tr_raw, y_tr_raw, X_te_raw, y_te = load_raw_data()
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr_raw, y_tr_raw, test_size=val_size, stratify=y_tr_raw, random_state=seed
    )
    data = dict(
        X_train=standardize(X_tr), y_train=y_tr,
        X_val=standardize(X_va), y_val=y_va,
        X_test=standardize(X_te_raw), y_test=y_te,
        y_train_raw=y_tr_raw,
    )
    print(f"[*] Train {len(y_tr):,} | Val {len(y_va):,} | Test {len(y_te):,} beats")
    return data


# ----------------------------------------------------------------------------
# Loaders, class weights, augmentation
# ----------------------------------------------------------------------------
def make_loader(X, y, batch_size=256, shuffle=False) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).unsqueeze(1).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available())


def class_alpha(y, num_classes: int = 5) -> torch.Tensor:
    """Inverse-sqrt-frequency class weights normalized to mean 1.0 (Eq. 7)."""
    cnt = np.bincount(y, minlength=num_classes).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(cnt, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def augment_batch(x, p=0.5, noise_std=0.05, scale=(0.9, 1.1), max_shift=8):
    """Batched on-device augmentation: Gaussian noise, gain scaling, temporal jitter."""
    B, _, L = x.shape
    dev = x.device
    mask = (torch.rand(B, 1, 1, device=dev) < p).float()
    xa = x + torch.randn_like(x) * noise_std
    xa = xa * torch.empty(B, 1, 1, device=dev).uniform_(*scale)
    shift = torch.randint(-max_shift, max_shift + 1, (B, 1), device=dev)
    idx = (torch.arange(L, device=dev)[None, :] + shift).clamp(0, L - 1)
    xa = torch.gather(xa, 2, idx.unsqueeze(1))
    return mask * xa + (1 - mask) * x
