"""Publication figures: EDA overview, learning curves, confusion matrices, ROC, attention, noise.

python -m src.visualize --all
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from .config import (CLASS_DESCRIPTIONS, CLASS_NAMES, FIGURES_DIR, RESULTS_DIR, ensure_dirs)
from .data import prepare_splits
from .evaluate import load_model
from .utils import get_device

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def plot_dataset_overview(data):
    y_all = np.concatenate([data["y_train_raw"], data["y_test"]])
    counts = np.bincount(y_all, minlength=5)

    pd.DataFrame({"Class": CLASS_NAMES, "Clinical Description": CLASS_DESCRIPTIONS, "Total Samples": counts,
                  "Percentage (%)": (100 * counts / counts.sum()).round(2),
                  "Imbalance Ratio (vs Majority)": (counts.max() / np.maximum(counts, 1)).round(2)}
                 ).to_csv(RESULTS_DIR / "dataset_distribution.csv", index=False)
    rows = []
    for name, lab in [("Train", data["y_train"]), ("Validation", data["y_val"]), ("Test", data["y_test"])]:
        rows.append({"Split": name, "Samples": len(lab), **{c: int((lab == i).sum()) for i, c in enumerate(CLASS_NAMES)}})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "split_distribution.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    bars = ax[0].bar(CLASS_NAMES, counts, color=COLORS, edgecolor="#333333")
    ax[0].set_yscale("log")
    ax[0].set_title("Class Sample Distribution (Log Scale)", fontweight="bold")
    ax[0].set_ylabel("Number of Heartbeats (log)")
    ax[0].set_xlabel("AAMI EC57 Class")
    for b in bars:
        ax[0].text(b.get_x() + b.get_width() / 2, b.get_height() * 1.15, f"{int(b.get_height()):,}",
                   ha="center", fontsize=9)
    for i, c in enumerate(CLASS_NAMES):
        idx = np.where(data["y_test"] == i)[0]
        if len(idx):
            ax[1].plot(data["X_test"][idx[0]], label=f"Class {c}", color=COLORS[i], linewidth=1.6)
    ax[1].set_title("Standardized ECG Beat Morphology (187 samples)", fontweight="bold")
    ax[1].set_xlabel("Time Step (Sample Index)")
    ax[1].set_ylabel("Standardized Amplitude (Z-score)")
    ax[1].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dataset_overview.png", dpi=200)
    plt.close()


def plot_training_curve(cfg="proposed"):
    p = RESULTS_DIR / f"{cfg}_history.csv"
    if not p.exists():
        return
    h = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(h["epoch"], h["train_loss"], label="Train Loss", linewidth=2)
    ax[0].plot(h["epoch"], h["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
    ax[0].set_title("Loss Progression", fontweight="bold")
    ax[1].plot(h["epoch"], h["train_acc"], label="Train Acc", linewidth=2)
    ax[1].plot(h["epoch"], h["val_acc"], label="Val Acc", linewidth=2, linestyle="--")
    ax[1].plot(h["epoch"], h["val_macro_f1"], label="Val Macro-F1", linewidth=2, linestyle=":")
    ax[1].set_title("Accuracy & Macro-F1", fontweight="bold")
    for a in ax:
        a.set_xlabel("Epoch")
        a.legend()
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{cfg}_training_curve.png", dpi=200)
    plt.close()


def plot_confusion(cfg="proposed"):
    p = RESULTS_DIR / f"{cfg}_confusion_matrix.csv"
    if not p.exists():
        return
    cm = pd.read_csv(p, index_col=0).values
    cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax[0])
    sns.heatmap(cmn, annot=True, fmt=".3f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax[1])
    ax[0].set_title("Confusion Matrix (Raw Counts)", fontweight="bold")
    ax[1].set_title("Confusion Matrix (Row-Normalized)", fontweight="bold")
    for a in ax:
        a.set_xlabel("Predicted Label")
        a.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{cfg}_confusion_matrix.png", dpi=200)
    plt.close()


def plot_confusion_grid(models=("cnn", "lstm", "bilstm", "resnet1d", "cnn_bilstm", "proposed")):
    titles = dict(cnn="1D-CNN", lstm="LSTM", bilstm="BiLSTM", resnet1d="ResNet-1D",
                  cnn_bilstm="CNN-BiLSTM", proposed="Proposed System")
    avail = {m: pd.read_csv(RESULTS_DIR / f"{m}_confusion_matrix.csv", index_col=0).values
             for m in models if (RESULTS_DIR / f"{m}_confusion_matrix.csv").exists()}
    if not avail:
        return
    rows = (len(avail) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (m, cm) in zip(axes, avail.items()):
        cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(f"{titles.get(m, m)} (Normalized)", fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    for ax in axes[len(avail):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices_all_models.png", dpi=200)
    plt.close()


def plot_roc(data, cfg="proposed"):
    p = RESULTS_DIR / f"{cfg}_test_probs.npy"
    if not p.exists():
        return
    prob = np.load(p)
    onehot = np.eye(5)[data["y_test"]]
    plt.figure(figsize=(7, 6))
    for i, c in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(onehot[:, i], prob[:, i])
        plt.plot(fpr, tpr, label=f"Class {c} (AUC = {roc_auc_score(onehot[:, i], prob[:, i]):.4f})", linewidth=1.8)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Multi-Class One-vs-Rest ROC Curves", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{cfg}_roc.png", dpi=200)
    plt.close()


def plot_attention(data, device, cfg="proposed"):
    from .config import MODELS_DIR
    if not (MODELS_DIR / f"{cfg}.pt").exists():
        return
    model, _ = load_model(cfg, device)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, cls in zip(axes, [0, 1, 2]):
        idx = np.where(data["y_test"] == cls)[0]
        if not len(idx):
            continue
        beat = data["X_test"][idx[0]]
        with torch.no_grad():
            model(torch.from_numpy(beat[None, None, :]).float().to(device))
        a = model.last_attn[0].cpu().numpy()
        a_interp = np.interp(np.linspace(0, 1, 187), np.linspace(0, 1, len(a)), a)
        ax.plot(beat, color="#333333", linewidth=1.6, label="ECG Beat")
        ax.fill_between(range(187), beat, alpha=0.3, color="#ff7f0e", label="Attention Weight")
        ax.plot(a_interp * (beat.max() - beat.min()) + beat.min(), "--", color="#d62728", label="Attention Profile")
        ax.set_title(f"Class {CLASS_NAMES[cls]}: Attention Focus", fontweight="bold")
        ax.set_xlabel("Time Step (Lead II)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "attention_visualization.png", dpi=200)
    plt.close()


def plot_noise():
    p = RESULTS_DIR / "robustness.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a, metric in zip(ax, ["Accuracy", "Macro F1-Score"]):
        sns.lineplot(data=df, x="Noise Std (σ)", y=metric, hue="Model", marker="o", linewidth=2.5, ax=a)
        a.set_title(f"{metric} vs. Noise Severity (σ)", fontweight="bold")
        a.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "noise_robustness_comparison.png", dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", default=True)
    ap.parse_args()
    ensure_dirs()
    device = get_device()
    data = prepare_splits()
    plot_dataset_overview(data)
    plot_training_curve()
    plot_confusion()
    plot_confusion_grid()
    plot_roc(data)
    plot_attention(data, device)
    plot_noise()
    print("[✓] Figures saved to figures/ and results/")


if __name__ == "__main__":
    main()
