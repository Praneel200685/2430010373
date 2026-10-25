"""Reproducibility and device helpers."""
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print(f"[*] Execution device: {device}")
        print(f"[*] GPU: {torch.cuda.get_device_name(0)} ({props.total_memory / 1024**3:.2f} GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[*] Execution device: {device} (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print(f"[*] Execution device: {device}")
        print("[!] Running on CPU; GPU is strongly recommended for training.")
    return device
