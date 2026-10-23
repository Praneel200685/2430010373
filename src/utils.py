"""Shared helpers: seeding, parameter count, metrics, plotting."""
import os, random
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score, roc_curve)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
MODELS = os.path.join(ROOT, "models")
CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_metrics(y_true, y_pred, y_prob):
    """Returns (summary dict, per-class DataFrame, confusion matrix)."""
    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    spec = []
    for i in labels:
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)
    onehot = np.eye(len(CLASS_NAMES))[y_true]
    aucs = [roc_auc_score(onehot[:, i], y_prob[:, i]) if onehot[:, i].sum() > 0 else np.nan for i in labels]
    per_class = pd.DataFrame({"class": CLASS_NAMES, "precision": p, "recall(sensitivity)": r,
                              "specificity": spec, "f1": f, "auc": aucs, "support": sup})
    summary = dict(accuracy=accuracy_score(y_true, y_pred),
                   macro_precision=p.mean(), macro_recall=r.mean(), macro_f1=f.mean(),
                   weighted_precision=pw, weighted_recall=rw, weighted_f1=fw,
                   macro_specificity=float(np.mean(spec)), macro_auc=float(np.nanmean(aucs)))
    return summary, per_class, cm


def plot_curves(hist, name):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["epoch"], hist["train_loss"], label="Train"); ax[0].plot(hist["epoch"], hist["val_loss"], label="Validation")
    ax[0].set_title(f"{name}: Loss"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(hist["epoch"], hist["train_acc"], label="Train"); ax[1].plot(hist["epoch"], hist["val_acc"], label="Validation")
    ax[1].set_title(f"{name}: Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, f"{name}_training_curve.png"), dpi=200); plt.close()


def plot_confusion(cm, name):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax[0])
    ax[0].set_title(f"{name}: Confusion matrix (counts)"); ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual")
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    sns.heatmap(cmn, annot=True, fmt=".3f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax[1])
    ax[1].set_title("Normalised (row = actual)"); ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Actual")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, f"{name}_confusion_matrix.png"), dpi=200); plt.close()


def plot_roc(y_true, y_prob, name):
    onehot = np.eye(len(CLASS_NAMES))[y_true]
    plt.figure(figsize=(6, 5))
    for i, c in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(onehot[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{c} (AUC={roc_auc_score(onehot[:, i], y_prob[:, i]):.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title(f"{name}: One-vs-rest ROC")
    plt.legend(loc="lower right"); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, f"{name}_roc.png"), dpi=200); plt.close()
