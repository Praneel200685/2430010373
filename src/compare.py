"""Builds results/comparison.csv (CO4) and results/ablation.csv (CO5) from per-model test metrics.
Run after:  python src/train.py --config all  &&  python src/test.py --config all
"""
import os, json
import pandas as pd
from utils import RESULTS

def load(cfg):
    p = os.path.join(RESULTS, f"{cfg}_metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None

NAMES = {"cnn": "1D-CNN", "lstm": "LSTM (2-layer)", "bilstm": "BiLSTM (2-layer)", "resnet1d": "ResNet-1D (Kachuee-style)",
         "cnn_bilstm": "CNN-BiLSTM (plain)", "proposed": "Proposed (MS-SE-CNN + BiLSTM + Attn)"}

def comparison():
    rows = []
    for cfg, name in NAMES.items():
        m = load(cfg)
        if m: rows.append({"Model": name, "Accuracy": m["accuracy"], "Precision": m["macro_precision"], "Recall": m["macro_recall"],
                           "F1": m["macro_f1"], "AUC": m["macro_auc"], "Parameters": m["parameters"],
                           "Train time (s)": m["train_time_s"], "Inference (ms/sample)": m["inference_ms_per_sample"]})
    df = pd.DataFrame(rows).round(4); df.to_csv(os.path.join(RESULTS, "comparison.csv"), index=False)
    print(df.to_string(index=False))

def ablation():
    spec = [("Baseline (CNN-BiLSTM)", "cnn_bilstm", "✗", "✗", "✗"), ("Experiment 1", "abl_A", "✓", "✗", "✗"),
            ("Experiment 2", "abl_AB", "✓", "✓", "✗"), ("Proposed", "proposed", "✓", "✓", "✓")]
    rows = []
    for name, cfg, a, b, c in spec:
        m = load(cfg)
        if m: rows.append({"Experiment": name, "A: Multi-scale SE-CNN": a, "B: Temporal attention": b, "C: Focal loss + augmentation": c,
                           "F1 (macro)": round(m["macro_f1"], 4), "Accuracy": round(m["accuracy"], 4), "AUC": round(m["macro_auc"], 4),
                           "Parameters": m["parameters"]})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS, "ablation.csv"), index=False)
    print(df.to_string(index=False))

def merged_results():
    rows = [load(c) for c in ["cnn", "lstm", "bilstm", "resnet1d", "cnn_bilstm", "abl_A", "abl_AB", "proposed"]]
    pd.DataFrame([r for r in rows if r]).round(4).to_csv(os.path.join(RESULTS, "results.csv"), index=False)

if __name__ == "__main__":
    comparison(); print(); ablation(); merged_results()
