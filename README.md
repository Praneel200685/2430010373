# Multi-Scale SE-CNN + BiLSTM with Temporal Attention for ECG Arrhythmia Classification

PyTorch implementation of a multi-scale 1D-CNN front end (k ∈ {3, 5, 7}) with Squeeze-and-Excitation channel
attention, followed by a BiLSTM with additive temporal attention pooling, for five-class (AAMI EC57) heartbeat
classification on MIT-BIH. A Pre-LN Transformer variant, class-weighted focal loss, and physiological
augmentation are also included.

## Student Details
- **Student Name:** Swastiek
- **Registration Number:** `2430010378`
- **Section:** CSE (Data Science) - Semester 5
- **Course:** Deep Learning

## Repository layout

```
.
├── src/
│   ├── config.py        # paths, class names, model configs, default hyperparameters
│   ├── data.py          # download, per-beat z-score, stratified split, loaders, augmentation
│   ├── models.py        # baselines, HybridNet (ablation-configurable), Transformer variant
│   ├── losses.py        # class-weighted focal loss
│   ├── train.py         # training engine + CLI
│   ├── evaluate.py      # test metrics, comparison/ablation tables
│   ├── tune.py          # OFAT hyperparameter sweep
│   ├── robustness.py    # Gaussian-noise robustness benchmark
│   ├── visualize.py     # figures (EDA, curves, confusion, ROC, attention, noise)
│   └── utils.py         # seeding, device
├── scripts/run_all.sh   # full pipeline
├── notebooks/           # original Colab notebook
├── data/  models/  results/  figures/   # generated at run time
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd ecg-arrhythmia-ms-se-cnn
pip install -r requirements.txt
```

**Dataset:** [ECG Heartbeat Categorization Dataset (Kaggle)](https://www.kaggle.com/datasets/shayanfazeli/heartbeat),
derived from the PhysioNet MIT-BIH Arrhythmia Database. The code tries KaggleHub automatically; otherwise download
`mitbih_train.csv` and `mitbih_test.csv` manually into `data/`.

## Usage

```bash
# Train the proposed model
python -m src.train --models proposed

# Train every baseline, ablation and the proposed model
python -m src.train --models all --epochs 40

# Evaluate on the official test set and build comparison/ablation tables
python -m src.evaluate --models all

# Noise robustness, figures, hyperparameter sweep
python -m src.robustness --models cnn_bilstm proposed
python -m src.visualize --all
python -m src.tune

# Or everything at once
bash scripts/run_all.sh

# Check parameter counts
python -m src.models
```

Available model names: `cnn`, `lstm`, `bilstm`, `resnet1d`, `cnn_bilstm`, `abl_A`, `abl_AB`, `proposed`,
`ms_transformer`. Add `--allow-synthetic` only for a pipeline smoke test; it generates fake beats and the
resulting metrics are not valid.

## Experimental protocol

| Item | Setting |
|---|---|
| Split | Official train/test files; stratified 10% validation from train (seed 42) |
| Normalization | Per-beat z-score |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4, cosine annealing (40 epochs) |
| Early stopping | Patience 8 on validation macro-F1 |
| Loss (proposed) | Inverse-sqrt class-weighted focal loss, γ = 2 |
| Augmentation (proposed) | p = 0.5: Gaussian noise (σ = 0.05), gain U(0.9, 1.1), jitter ±8 samples |

## Reported results (single run, seed 42, Tesla T4)

| Model | Loss | Accuracy | Macro-F1 | Macro-AUC | Params |
|---|---|---|---|---|---|
| 1D-CNN | CE | 0.9808 | 0.8955 | 0.9880 | 52,453 |
| LSTM | CE | 0.9722 | 0.8666 | 0.9793 | 50,757 |
| BiLSTM | CE | 0.9832 | 0.9093 | 0.9923 | 134,277 |
| ResNet-1D | CE | 0.9858 | 0.9181 | 0.9946 | 52,933 |
| CNN–BiLSTM | CE | 0.9853 | 0.9128 | 0.9909 | 115,093 |
| + MS-SE-CNN | CE | 0.9847 | 0.9170 | 0.9920 | 118,135 |
| + MS-SE-CNN + Attn | CE | 0.9859 | 0.9236 | 0.9934 | 134,776 |
| + Focal + Aug (proposed) | Focal | 0.9683 | 0.8647 | 0.9869 | 134,776 |

Under Gaussian noise σ = 0.20 the proposed model reaches macro-F1 0.7396 vs. 0.6738 for the plain CNN–BiLSTM
(+6.58 points); at σ ≤ 0.10 the plain baseline is higher. Regenerate all numbers with `scripts/run_all.sh`.

## Limitations

- The official split is **not patient-disjoint**; results may overestimate inter-patient generalization.
- All results come from a **single seed** without confidence intervals.
- The noise baseline was trained without augmentation, so the robustness gain cannot yet be attributed to focal
  loss alone; a CE + augmentation control is needed.
- Gaussian noise is a simple proxy for ambulatory artifacts (see the MIT-BIH Noise Stress Test Database).

## References

MIT-BIH: Moody & Mark, IEEE EMB Mag., 2001 · Kachuee et al., ICHI 2018 · Hu et al., CVPR 2018 (SE) ·
Bahdanau et al., ICLR 2015 (additive attention) · Lin et al., ICCV 2017 (focal loss).
