"""Central configuration: paths, class names, model configs, default hyperparameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

TRAIN_CSV = DATA_DIR / "mitbih_train.csv"
TEST_CSV = DATA_DIR / "mitbih_test.csv"

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
CLASS_DESCRIPTIONS = [
    "Normal / Bundle branch block (N)",
    "Supraventricular ectopic (S)",
    "Ventricular ectopic (V)",
    "Fusion beat (F)",
    "Unknown / Paced beat (Q)",
]

# name -> model family, constructor kwargs, loss, augmentation flag
CONFIGS = {
    "cnn":        dict(model="cnn",      kw={}, loss="ce", augment=False),
    "lstm":       dict(model="lstm",     kw={}, loss="ce", augment=False),
    "bilstm":     dict(model="bilstm",   kw={}, loss="ce", augment=False),
    "resnet1d":   dict(model="resnet1d", kw={}, loss="ce", augment=False),
    "cnn_bilstm": dict(model="hybrid", kw=dict(multiscale=False, se=False, attn=False), loss="ce", augment=False),
    "abl_A":      dict(model="hybrid", kw=dict(multiscale=True, se=True, attn=False), loss="ce", augment=False),
    "abl_AB":     dict(model="hybrid", kw=dict(multiscale=True, se=True, attn=True), loss="ce", augment=False),
    "proposed":   dict(model="hybrid", kw=dict(multiscale=True, se=True, attn=True), loss="focal", augment=True),
    "ms_transformer": dict(model="ms_transformer", kw={}, loss="focal", augment=True),
}

ALL_BASELINES_AND_ABLATIONS = ["cnn", "lstm", "bilstm", "resnet1d", "cnn_bilstm", "abl_A", "abl_AB", "proposed"]

DEFAULT_HP = dict(
    lr=1e-3, batch_size=256, epochs=40, weight_decay=1e-4,
    dropout=0.3, hidden=64, patience=8, grad_clip=1.0, focal_gamma=2.0, seed=42,
)

MODEL_LABELS = {
    "cnn": "1D-CNN (Baseline)",
    "lstm": "LSTM (2-Layer Baseline)",
    "bilstm": "BiLSTM (2-Layer Baseline)",
    "resnet1d": "ResNet-1D (Kachuee et al.)",
    "cnn_bilstm": "CNN-BiLSTM (Plain Hybrid)",
    "abl_A": "Exp. 1: + MS-SE-CNN",
    "abl_AB": "Exp. 2: + MS-SE-CNN + Attn",
    "proposed": "Proposed (MS-SE-CNN-BiLSTM-Attn)",
    "ms_transformer": "Proposed (MS-SE-CNN-Transformer)",
}


def ensure_dirs():
    for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
