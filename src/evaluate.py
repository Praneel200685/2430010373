"""Test-set evaluation, comparison and ablation tables.

Examples
--------
python -m src.evaluate --models all
python -m src.evaluate --models proposed abl_AB
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from .config import (ALL_BASELINES_AND_ABLATIONS, CLASS_NAMES, MODEL_LABELS, MODELS_DIR, RESULTS_DIR, ensure_dirs)
from .data import make_loader, prepare_splits
from .models import build_model, count_params
from .train import predict
from .utils import get_device


def compute_metrics(y_true, y_pred, y_prob):
    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    spec = []
    for i in labels:
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)

    onehot = np.eye(len(CLASS_NAMES))[y_true]
    aucs = [roc_auc_score(onehot[:, i], y_prob[:, i]) if onehot[:, i].sum() > 0 else np.nan for i in labels]

    per_class = pd.DataFrame({"Class": CLASS_NAMES, "Precision": p, "Recall (Sensitivity)": r,
                              "Specificity": spec, "F1-Score": f, "ROC-AUC": aucs, "Support": sup})
    summary = dict(accuracy=accuracy_score(y_true, y_pred), macro_precision=float(p.mean()),
                   macro_recall=float(r.mean()), macro_f1=float(f.mean()),
                   macro_specificity=float(np.mean(spec)), macro_auc=float(np.nanmean(aucs)),
                   weighted_precision=float(pw), weighted_recall=float(rw), weighted_f1=float(fw))
    return summary, per_class, cm


def load_model(cfg_name, device):
    ck = torch.load(MODELS_DIR / f"{cfg_name}.pt", map_location=device, weights_only=False)  # trusted local file
    model = build_model(ck["cfg"]["model"], hidden=ck["hp"]["hidden"], dropout=ck["hp"]["dropout"],
                        **ck["cfg"]["kw"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def evaluate_model(cfg_name, data, device):
    if not (MODELS_DIR / f"{cfg_name}.pt").exists():
        print(f"[!] No checkpoint for '{cfg_name}', skipping.")
        return None
    model, ck = load_model(cfg_name, device)
    loader = make_loader(data["X_test"], data["y_test"], batch_size=512, shuffle=False)
    t0 = time.time()
    logits, y, _ = predict(model, loader, device)
    dt = time.time() - t0

    prob = F.softmax(logits, dim=1).numpy()
    pred = prob.argmax(axis=1)
    y = y.numpy()
    summary, per_class, cm = compute_metrics(y, pred, prob)
    summary.update(config=cfg_name, parameters=count_params(model), train_time_s=ck["train_time_s"],
                   epochs_run=ck["epochs_run"], inference_ms_per_sample=round(1000 * dt / len(y), 4))

    per_class.to_csv(RESULTS_DIR / f"{cfg_name}_per_class_metrics.csv", index=False)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(RESULTS_DIR / f"{cfg_name}_confusion_matrix.csv")
    np.save(RESULTS_DIR / f"{cfg_name}_test_probs.npy", prob)
    with open(RESULTS_DIR / f"{cfg_name}_metrics.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in summary.items()}, f, indent=2)
    print(f"[{cfg_name}] acc {summary['accuracy']:.4f} | macro-F1 {summary['macro_f1']:.4f} | "
          f"macro-recall {summary['macro_recall']:.4f} | AUC {summary['macro_auc']:.4f}")
    return summary, per_class, cm, prob


def _load_metrics(cfg):
    p = RESULTS_DIR / f"{cfg}_metrics.json"
    return json.load(open(p)) if p.exists() else None


def build_tables():
    """Baseline comparison (CO4), ablation (CO5) and merged results CSVs from saved metrics."""
    rows = []
    for cfg, label in MODEL_LABELS.items():
        m = _load_metrics(cfg)
        if m:
            rows.append({"Model Architecture": label, "Accuracy": round(m["accuracy"], 4),
                         "Macro Precision": round(m["macro_precision"], 4), "Macro Recall": round(m["macro_recall"], 4),
                         "Macro F1-Score": round(m["macro_f1"], 4), "ROC-AUC": round(m["macro_auc"], 4),
                         "Trainable Params": m["parameters"], "Train Time (s)": m["train_time_s"],
                         "Inference (ms/sample)": m["inference_ms_per_sample"]})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "comparison.csv", index=False)

    spec = [("Baseline (CNN-BiLSTM)", "cnn_bilstm", "-", "-", "-"),
            ("Exp. 1 (+ MS-SE-CNN)", "abl_A", "yes", "-", "-"),
            ("Exp. 2 (+ Temporal Attn)", "abl_AB", "yes", "yes", "-"),
            ("Full system", "proposed", "yes", "yes", "yes")]
    abl = []
    for name, cfg, a, b, c in spec:
        m = _load_metrics(cfg)
        if m:
            abl.append({"Configuration": name, "A: Multi-Scale SE": a, "B: Temporal Attention": b,
                        "C: Focal Loss + Aug": c, "Macro F1-Score": round(m["macro_f1"], 4),
                        "Accuracy": round(m["accuracy"], 4), "ROC-AUC": round(m["macro_auc"], 4),
                        "Parameters": m["parameters"]})
    pd.DataFrame(abl).to_csv(RESULTS_DIR / "ablation.csv", index=False)

    merged = [m for m in (_load_metrics(c) for c in ALL_BASELINES_AND_ABLATIONS) if m]
    pd.DataFrame(merged).to_csv(RESULTS_DIR / "results.csv", index=False)
    print("[✓] Wrote results/comparison.csv, ablation.csv, results.csv")


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained models on the official test set")
    ap.add_argument("--models", nargs="+", default=["all"])
    args = ap.parse_args()
    targets = ALL_BASELINES_AND_ABLATIONS if "all" in args.models else args.models

    ensure_dirs()
    device = get_device()
    data = prepare_splits()
    for t in targets:
        evaluate_model(t, data, device)
    build_tables()


if __name__ == "__main__":
    main()
