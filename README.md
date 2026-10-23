# 1. Project Title
**Multi-Scale SE-CNN + BiLSTM with Temporal Attention for ECG Arrhythmia Classification on the MIT-BIH Dataset**

# 2. Student Details
| Field | Value |
|---|---|
| Name | `Praneel Kapoor` |
| Registration Number | `2430010373` |
| Section | `D` |

# 3. Dataset
| Item | Details |
|---|---|
| Name | ECG Heartbeat Categorization Dataset (MIT-BIH Arrhythmia part) |
| Source | Kaggle (shayanfazeli/heartbeat), derived from PhysioNet MIT-BIH Arrhythmia Database |
| Samples | 109,446 heartbeats (87,554 official train file + 21,892 official test file), 187 time-steps each |
| Classes | 5 (AAMI): N, S, V, F, Q |
| Split | Train / Validation / Test = 78,798 / 8,756 / 21,892 (about 72.0% / 8.0% / 20.0%). Validation is a stratified 10% of the official train file (seed 42); the official test file is untouched until final evaluation. Exact numbers: `results/split_distribution.csv` |
| Class imbalance | Heavily imbalanced (N is about 82.8%, F about 0.7%); see `results/class_imbalance.csv`, `figures/dataset_overview.png` |

**Preprocessing:** per-beat z-score normalisation. For the proposed model only (training set only): Gaussian noise (std 0.05), amplitude scaling U(0.9, 1.1), time shift of up to 8 samples, each applied with p = 0.5. Details: `results/preprocessing_steps.csv`.

# 4. Model
- **Architecture name:** MS-SE-CNN-BiLSTM-Attn (proposed); baselines: 1D-CNN, LSTM, BiLSTM, ResNet-1D, plain CNN-BiLSTM.
- **Description:** two multi-scale convolution blocks (parallel kernels 3/5/7) with Squeeze-and-Excitation channel attention -> BiLSTM (hidden 64) -> additive temporal attention pooling -> MLP classifier (5 classes). Source: `src/model.py`.
- **Pretrained model:** none, all models are trained from scratch.
- **Proposed modifications:** A) multi-scale SE-CNN front end, B) temporal attention pooling, C) class-weighted focal loss + ECG-specific augmentation.

# 5. Hyperparameters
| Hyperparameter | Value |
|---|---|
| Learning rate | 1e-3 (`<update after tuning>`) |
| Batch size | 256 |
| Number of epochs | up to 40, early stopping (patience 8 on validation macro-F1) |
| Optimizer | AdamW |
| Loss function | Cross-entropy (baselines); class-weighted focal loss, gamma = 2 (proposed) |
| Scheduler | CosineAnnealingLR (T_max = epochs) |
| Dropout | 0.3 |
| Weight decay | 1e-4 |
| Other | BiLSTM hidden = 64, grad-clip 1.0, seed 42, SE reduction 8 |

# 6. Hyperparameter Tuning
- **Parameters considered:** learning rate, dropout, LSTM hidden size, weight decay (`src/tune.py`, one-factor-at-a-time, 15-epoch runs, selection by *validation* macro-F1 only).
- **Values tested:** lr {3e-4, 1e-3, 3e-3}; dropout {0.2, 0.3, 0.5}; hidden {32, 64, 128}; weight decay {1e-5, 1e-4, 1e-3}.
- **Final selected values / reason:** `<fill from results/hyperparameter_tuning.csv>`: pick the value with the best validation macro-F1, with ties broken by fewer parameters.

# 7. Results
Run order is in the section below. All numbers below must be copied from the generated files.

| Output | File |
|---|---|
| Dataset tables | `results/dataset_distribution.csv`, `split_distribution.csv`, `class_imbalance.csv`, `preprocessing_steps.csv` |
| Confusion matrix | `results/proposed_confusion_matrix.png` / `.csv` |
| Accuracy, macro/weighted P/R/F1, specificity, ROC-AUC | `results/proposed_metrics.json`, `results/proposed_per_class_metrics.csv`, `results/proposed_roc.png` |
| Training vs validation loss/accuracy | `results/proposed_training_curve.png`, `results/proposed_history.csv` |
| Compute cost, trainable params | `results/results.csv` (parameters, train_time_s, inference_ms_per_sample) |
| CO4 comparison | `results/comparison.csv` (reproduced on same split), `results/literature_comparison.csv` (published papers) |
| CO5 ablation | `results/ablation.csv` |
| Robustness | `results/robustness.csv` |

**Headline test results (fill after running):** Accuracy `__`, macro-F1 `__`, macro-AUC `__`, parameters `__`.

### How to reproduce
```bash
pip install -r requirements.txt
# place mitbih_train.csv and mitbih_test.csv in data/
python src/preprocessing.py        # dataset tables + figure
python src/tune.py                 # optional hyperparameter search
python src/train.py --config all   # trains 8 configs (baselines, ablations, proposed)
python src/test.py  --config all   # test metrics, confusion matrices, ROC
python src/test.py  --config proposed --robustness
python src/test.py  --config cnn_bilstm --robustness
python src/compare.py              # comparison.csv, ablation.csv, results.csv
```

# 8. Innovation
**What is the innovation compared with baseline/state-of-the-art methods?**
1. **Multi-scale SE-CNN front end (A):** parallel kernels 3/5/7 capture narrow QRS complexes and wider P/T-wave morphology at the same time; SE attention re-weights channels. Plain CNN baselines use a single kernel size.
2. **Temporal attention over BiLSTM states (B):** replaces mean/last-state pooling so that the classifier focuses on the diagnostically relevant segment of the beat (attention weights can be visualised via `HybridNet.last_attn`).
3. **Imbalance-aware training (C):** class-weighted focal loss plus ECG-specific augmentation to improve recall on the rare S, F and Q classes.

The contribution of each component is isolated in the ablation study (`results/ablation.csv`) and noise robustness is reported in `results/robustness.csv`. `<Summarise the measured gains here once the experiments are done.>`

# 9. Folder Structure
```text
RegistrationNumber/
├── README.md
├── requirements.txt
├── data/
│   └── dataset_information.txt      # download instructions (CSV files go here)
├── src/
│   ├── preprocessing.py             # loading, splits, normalisation, augmentation, dataset tables
│   ├── model.py                     # all architectures (baselines + proposed)
│   ├── train.py                     # training (configs: cnn, lstm, bilstm, resnet1d, cnn_bilstm, abl_A, abl_AB, proposed)
│   ├── test.py                      # test metrics, confusion matrix, ROC, robustness
│   ├── tune.py                      # hyperparameter search
│   ├── compare.py                   # comparison.csv, ablation.csv, results.csv
│   └── utils.py                     # metrics, plots, seeding
├── notebooks/
│   └── experiments.ipynb            # runs the pipeline and displays results
├── results/                         # generated: csv/json tables, confusion matrices, curves, ROC
├── models/
│   └── model_description.txt        # trained weights (.pt) are saved here
└── figures/
    └── dataset_overview.png
```

### References
`<List the papers used in results/literature_comparison.csv, plus: Moody & Mark 2001 (MIT-BIH); Hu et al. 2018 (Squeeze-and-Excitation); Lin et al. 2017 (Focal Loss).>`
