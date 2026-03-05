# 🛢️ ROP Prediction — Algerian Oil Fields

> **Rate of Penetration (ROP) prediction for drilling operations using cluster-aware deep learning.**

---

## Overview

This project predicts the Rate of Penetration (ROP, measured in ft/hr) during drilling operations in Algerian oil and gas fields. ROP quantifies how fast the drill bit advances through rock and is a critical efficiency metric — optimizing it reduces drilling time and cost significantly.

The model ingests real-time drilling sensor readings and outputs a predicted ROP, enabling drillers and engineers to adjust parameters proactively rather than reactively.

> ⚠️ **Data Privacy**: The well data used in this project is proprietary and collected from active fields in Algeria. It is not included in this repository and cannot be shared publicly.

---

## The Problem

A single neural network trained across all wells performed poorly on unseen wells (Leave-One-Well-Out R² ranging from **−35% to +55%**). The root cause is **domain shift**: wells in different geological zones obey different physics. The relationship between drilling parameters (WOB, RPM, Torque) and ROP changes depending on rock type, formation hardness, and pore pressure — none of which are directly measured.

---

## Solution: Cluster-Aware Modeling

Wells are automatically grouped by **geological and operational similarity** using K-Means clustering on well fingerprints (statistical summaries of each well's sensor distributions). A separate neural network is trained per cluster. At inference time, a new well is assigned to its nearest cluster and predicted by that cluster's dedicated model.

```
All Wells
    │
    ▼
Well Fingerprinting        ← median, std, p25, p75 of each sensor per well
    │
    ▼
K-Means Clustering         ← groups wells by similarity (k=4, selected by silhouette score)
    │
    ├── Cluster 0 → Model 0   (Wells 7, 77)
    ├── Cluster 1 → Model 1   (Wells 9, 14, 22)
    ├── Cluster 2 → Model 2   (Wells 10, 11)
    └── Cluster 3 → Model 3   (Wells 1, 13)
```

---

## Dataset

| Property | Details |
|---|---|
| Source | Algerian oil and gas fields (confidential) |
| Wells | 9 wells (IDs: 1, 7, 9, 10, 11, 13, 14, 22, 77) |
| Total rows (cleaned) | ~411,000 time-series records |
| Sampling interval | 60-second averages |
| Target variable | `ROP_inst_cal` — instantaneous calibrated ROP (ft/hr) |

### Input Features

| Feature | Description | Unit |
|---|---|---|
| `WOB Avg` | Weight on Bit average | klbs |
| `WOH Avg` | Weight on Hook average | klbs |
| `RPM Avg` | Rotary speed average | rpm |
| `Torque Avg` | Drill string torque average | ft·lbs |
| `Pressure: SPP Avg` | Standpipe pressure average | psi |
| `Flow In Pumps Avg` | Mud flow rate average | gpm |
| `Depth` | True measured depth | ft |

### Data Cleaning Steps

- Remove rows with `na_values = [0, -9999.99]`
- Exclude early formations (For = 1–4) — surface casing, unreliable ROP
- Filter `Flow In Pumps Avg < 50,000` (sensor noise / pump-off periods)
- Clip ROP to the 1st–99th percentile per well
- Drop all rows with any remaining NaN

---

## Well Clustering Results

Silhouette scores guided automatic selection of **k = 4** clusters:

| k | Silhouette Score |
|---|---|
| 2 | 0.205 |
| 3 | 0.290 |
| **4** | **0.351** ← selected |

### Cluster Assignments

| Cluster | Wells | Rows | Mean ROP | Std ROP |
|---|---|---|---|---|
| 0 | 7, 77 | 66,608 | 23.48 ft/hr | 20.34 |
| 1 | 9, 14, 22 | 131,476 | 20.75 ft/hr | 20.20 |
| 2 | 10, 11 | 107,436 | 15.82 ft/hr | 14.35 |
| 3 | 1, 13 | 105,874 | 15.82 ft/hr | 14.02 |

---

## Model Architecture

Each cluster uses an identical feed-forward neural network:

```
Input (7 features)
    │
    ├─ Dense(128) → BatchNorm → ReLU → Dropout(0.25)
    ├─ Dense(128) → BatchNorm → ReLU → Dropout(0.25)
    ├─ Dense(128) → BatchNorm → ReLU → Dropout(0.25)
    ├─ Dense(128) → BatchNorm → ReLU → Dropout(0.25)
    │
    └─ Dense(1, activation='softplus')   ← ensures ROP > 0
```

| Hyperparameter | Value | Reason |
|---|---|---|
| Optimizer | Adam (lr=0.001, clipnorm=1.0) | Adaptive LR + gradient clipping for heterogeneous data |
| Loss | Huber | Less sensitive to outliers than MSE |
| Batch size | 512 | Stable gradient estimates on large dataset |
| Early stopping | patience=60, restore_best_weights=True | Prevents overfitting, returns best checkpoint |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=20) | Helps escape local minima |
| Normalization | StandardScaler (fit on train split only) | Prevents data leakage |

---

## Results

### Cluster-Aware Leave-One-Well-Out Evaluation

Each well is held out completely, assigned to its cluster, and predicted by that cluster's model. This is the honest generalization metric.

| Well | Cluster | R² | MAE (ft/hr) | RMSE (ft/hr) |
|---|---|---|---|---|
| 1 | 3 | **88.2%** | 3.20 | 5.07 |
| 7 | 0 | **91.1%** | 3.79 | 6.04 |
| 9 | 1 | **90.7%** | 4.41 | 7.21 |
| 10 | 2 | **84.7%** | 3.02 | 4.46 |
| 11 | 2 | **90.2%** | 3.49 | 5.41 |
| 13 | 3 | **88.0%** | 2.95 | 4.58 |
| 14 | 1 | **87.0%** | 4.18 | 6.98 |
| 22 | 1 | **91.5%** | 3.00 | 5.07 |
| 77 | 0 | **91.4%** | 3.75 | 5.99 |
| **Mean** | | **89.2%** | **3.53** | **5.65** |
| **Median** | | **90.2%** | | |

All 9 wells achieve 🟢 R² > 84% — a major improvement over the naive single-model baseline (−35% to +55%).

### Training per Cluster

| Cluster | Epochs (early stopped) | Validation R² |
|---|---|---|
| 0 (Wells 7, 77) | 584 | 90.8% |
| 1 (Wells 9, 14, 22) | 661 | 90.2% |
| 2 (Wells 10, 11) | 464 | 88.0% |
| 3 (Wells 1, 13) | 535 | 87.9% |

---

## Notebook Versions

| Version | Key Change | Best LOWO R² |
|---|---|---|
| v1 | Baseline single model, one well training | ~43–69% (some wells) |
| v2 | Multi-well training, Huber loss, Depth as feature | ~55% mean |
| **v3** | **Cluster-aware modeling (current)** | **89.2% mean** |

---

## Project Structure

```
rop-prediction/
│
├── ROP_prediction_v3.ipynb     # Main notebook (run on Google Colab)
│
├── README.md                   # This file
│
└── rop_model_v3/               # Saved model artifacts (not in repo — saved to Drive)
    ├── model_cluster_0.h5
    ├── model_cluster_1.h5
    ├── model_cluster_2.h5
    ├── model_cluster_3.h5
    ├── scaler_cluster_0.pkl    # StandardScaler per cluster
    ├── scaler_cluster_1.pkl
    ├── scaler_cluster_2.pkl
    ├── scaler_cluster_3.pkl
    ├── kmeans.pkl              # K-Means clustering model
    ├── fp_scaler.pkl           # Fingerprint scaler
    ├── fingerprints.pkl        # Per-well fingerprint reference
    ├── features.pkl            # Feature column names
    └── well_to_cluster.pkl     # Well ID → cluster mapping
```

---

## How to Run

### Requirements

```
tensorflow >= 2.10
scikit-learn >= 1.0
pandas
numpy
matplotlib
joblib
```

### Steps

1. Open `ROP_prediction_v3.ipynb` in [Google Colab](https://colab.research.google.com)
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Mount Google Drive when prompted
4. Set your data path in **Cell 2 (Configuration)**:
   ```python
   DATA_PATH = 'drive/My Drive/your_folder/well_{well_id}_60.csv'
   ALL_WELLS = [...]   # your well IDs
   ```
5. Run all cells in order

### Predicting a New Well

```python
from tensorflow import keras
import joblib, pandas as pd

# Load artifacts
model_dir    = 'drive/My Drive/rop_model_v3'
km           = joblib.load(f'{model_dir}/kmeans.pkl')
fp_scaler    = joblib.load(f'{model_dir}/fp_scaler.pkl')
fingerprints = joblib.load(f'{model_dir}/fingerprints.pkl')
feat_cols    = joblib.load(f'{model_dir}/features.pkl')
models       = {i: keras.models.load_model(f'{model_dir}/model_cluster_{i}.h5') for i in range(4)}
scalers      = {i: joblib.load(f'{model_dir}/scaler_cluster_{i}.pkl') for i in range(4)}

# Predict (uses the predict_new_well function defined in Cell 13 of the notebook)
predictions, cluster = predict_new_well(
    new_well_df, feat_cols, 'ROP_inst_cal',
    km, fp_scaler, fingerprints, models, scalers
)
```

---

## Key Lessons Learned

**1. Never train on one well and test on another** without first verifying they share similar distributions. Geological heterogeneity makes this a near-impossible task for a single model.

**2. Data leakage from normalization** — always fit the scaler on the training split only, then apply it to validation and test sets. Fitting on the test set inflates scores artificially.

**3. Depth matters** — ROP at 3,000 ft behaves differently than at 9,000 ft even with identical drilling parameters, because rock properties change with depth.

**4. Huber loss outperforms MSE** when the target has a wide range (1–106 ft/hr) with real-world noise. MSE over-penalizes the tail and destabilizes training on heterogeneous data.

**5. Cluster first, model second** — grouping wells by operational fingerprint (sensor statistics) before training is the single biggest improvement, taking mean LOWO R² from ~20% to **89.2%**.

---

## Authors

Developed as an applied ML research project for drilling optimization in the Algerian oil and gas sector.
