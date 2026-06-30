# A Hybrid Cost-Sensitive Ensemble Approach for Indonesian Local Script Handwritten Recognition

**Course:** Research Methodology × Machine Learning — BINUS University, Semester 4

---

## Introduction

Proyek ini membangun sistem **Handwritten Character Recognition (HCR)** untuk 9 aksara lokal Indonesia (Nusantara) menggunakan pendekatan **Hybrid CNN-Ensemble** dengan cost-sensitive learning. Pipeline menggabungkan frozen CNN sebagai feature extractor (MobileNetV2 & EfficientNet-B0) dengan tiga ensemble classifier berbasis gradient boosting (XGBoost, LightGBM, CatBoost), dilengkapi Optuna hyperparameter tuning.

Gap penelitian yang diisi:
- Belum ada studi yang mencakup seluruh 9 aksara Nusantara dalam satu eksperimen terstandar.
- Class imbalance antar aksara belum pernah ditangani dengan cost-sensitive weighting pada konteks multi-script HCR.
- Belum ada karya yang mengombinasikan frozen CNN feature extraction dengan gradient-boosting ensemble.

**Dataset:** [Indonesian Local Script Characters Handwriting Dataset (Ihsan, 2024) — Mendeley Data](https://data.mendeley.com/)  
55.713 gambar · 228 kelas · 9 aksara

---

## Team Members

| Name | Student ID |
|------|-----|
| Dian Rakhmawati Lestari | 2802539085 |
| Fadhlan Nur Rachman | 2802491690 |
| Bintang Nur Fadhlillah | 2802536083 |

---

## Isi Repositori & Penjelasan Pipeline

### File & Folder Utama

| Path | Keterangan |
|------|------------|
| `resultfromkaggle.ipynb` | **Notebook utama** — pipeline lengkap dari loading data hingga deployment |
| `pipeline_hcr_nusantara.ipynb` | Notebook eksplorasi lokal (versi awal) |
| `clean_dataset.py` | Script sanitasi folder/filename dataset ke format ASCII |
| `datasetscript/` | Dataset mentah (9 subfolder per aksara) |
| `features/` | Cache numpy array hasil feature extraction CNN |
| `models/` | Artefak model terlatih, label encoder, visualisasi EDA |
| `paper/` | Draft paper IEEE dan proposal final |

---

### Pipeline `resultfromkaggle.ipynb`

```
Raw Images (55,713)
      │
      ▼
┌─────────────────────────────────────────────────┐
│  Sek. 1 — Data Loading & EDA                    │
│  Baca 9 script folder → DataFrame (filepath,    │
│  label, script). Plot distribusi kelas & sampel │
│  gambar per aksara.                             │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Sek. 2 — Preprocessing & Stratified Split      │
│  LabelEncoder (228 kelas) → Stratified 80/20    │
│  Preprocessing per gambar:                      │
│    Grayscale → Gaussian Blur Binarize           │
│    → Replicate 3-ch RGB → ImageNet Normalize    │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Sek. 3 — Feature Extraction (Frozen CNN)       │
│  MobileNetV2  (ImageNet weights, frozen)        │
│  EfficientNet-B0 (ImageNet weights, frozen)     │
│  Output: vektor 1280-d per gambar               │
│  Hasil di-cache ke features/*.npy               │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Sek. 4 — PCA Dimensionality Reduction          │
│  1280-d → 256-d (MobileNetV2: 95.9% var,       │
│                   EfficientNet-B0: 94.1% var)   │
│  Demo visual satu gambar Jawi step-by-step      │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Sek. 5 — Class Weights (Cost-Sensitive)        │
│  sklearn compute_class_weight("balanced")       │
│  Atasi imbalance 14.5x (Sunda vs Jawi)         │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Sek. 6 — Hyperparameter Tuning (Optuna TPE)   │
│  20 trials per model · 30% subsample data       │
│  Tuning: XGBoost, LightGBM, CatBoost           │
│  per backbone (total 6 model configuration)     │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Sek. 7 — Final Model Training + Learning Curves│
│  XGBoost (GPU hist), LightGBM, CatBoost (GPU)  │
│  500 estimators (XGB/CAT), 100 (LGBM)          │
│  Plot learning curves per model                 │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Sek. 8 — Evaluasi Komprehensif                 │
│  Individual model metrics + Ensemble voting:    │
│    · SoftVote per-backbone                      │
│    · SoftVote (6 models)                        │
│    · WeightedVote (6 models) ← BEST             │
│  Confusion matrix 228×228, ROC curves,          │
│  per-script & per-class analysis                │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Sek. 9 — Inference Demo (single image)         │
│  Sek. 11 — Deployment (Streamlit app.py)        │
└─────────────────────────────────────────────────┘
```

---

## Hasil Evaluasi

### Full Leaderboard (Individual + Ensemble)

| Rank | Backbone | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|------|----------|-------|----------|-----------------|--------------|----------|
| **1** | **ALL** | **WeightedVote (6 models)** | **0.9003** | **0.9192** | **0.9170** | **0.9160** |
| 2 | ALL | SoftVote (6 models) | 0.8992 | 0.9181 | 0.9161 | 0.9149 |
| 3 | MobileNetV2 | CatBoost | 0.8506 | 0.8778 | 0.8805 | 0.8770 |
| 4 | MobileNetV2 | SoftVote (XGB+LGBM+CAT) | 0.8524 | 0.8763 | 0.8745 | 0.8735 |
| 5 | EfficientNet-B0 | SoftVote (XGB+LGBM+CAT) | 0.8508 | 0.8769 | 0.8716 | 0.8712 |
| 6 | EfficientNet-B0 | CatBoost | 0.8385 | 0.8681 | 0.8698 | 0.8668 |
| 7 | EfficientNet-B0 | XGBoost | 0.8464 | 0.8650 | 0.8561 | 0.8574 |
| 8 | MobileNetV2 | XGBoost | 0.8457 | 0.8610 | 0.8578 | 0.8571 |
| 9 | MobileNetV2 | LightGBM | 0.7846 | 0.7996 | 0.8010 | 0.7976 |
| 10 | EfficientNet-B0 | LightGBM | 0.7840 | 0.8001 | 0.7984 | 0.7957 |

> Ensemble **WeightedVote (6 models)** unggul +3.90% Macro F1 dibanding model individual terbaik (CatBoost + MobileNetV2).

### Per-Script Macro F1 (Best Model — WeightedVote)

| Script | Classes | Train | Test | Macro P | Macro R | Macro F1 |
|--------|---------|-------|------|---------|---------|----------|
| Lontara | 23 | 1,839 | 460 | 0.9877 | 0.9913 | **0.9892** |
| Pallawa | 33 | 3,168 | 792 | 0.9844 | 0.9886 | **0.9862** |
| Kawi | 31 | 4,995 | 1,251 | 0.9841 | 0.9862 | **0.9849** |
| Lampung | 20 | 3,997 | 999 | 0.9762 | 0.9760 | **0.9759** |
| Batak | 19 | 1,520 | 380 | 0.9370 | 0.9289 | **0.9307** |
| Sunda | 32 | 16,179 | 4,045 | 0.9121 | 0.9105 | **0.9107** |
| Jawi | 32 | 1,280 | 320 | 0.8546 | 0.8469 | **0.8417** |
| Bali | 18 | 3,595 | 898 | 0.8228 | 0.8250 | **0.8225** |
| Jawa | 20 | 7,997 | 1,998 | 0.7595 | 0.7414 | **0.7464** |
| **OVERALL** | **228** | **44,570** | **11,143** | **0.9192** | **0.9170** | **0.9160** |

### ROC AUC

| Model | Macro AUC | Micro AUC |
|-------|-----------|-----------|
| WeightedVote (6 models) | **0.9992** | **0.9996** |
| SoftVote (6 models) | 0.9992 | 0.9996 |
| MobileNetV2 + CatBoost | 0.9987 | 0.9993 |
| EfficientNet-B0 + XGBoost | 0.9987 | 0.9992 |

### Statistik Per-Kelas

- **F1 ≥ 0.90:** 156 kelas (68.4%)
- **F1 ≥ 0.80:** 201 kelas (88.2%)
- **F1 terbaik:** 1.0000 (`Batak_ta`)
- **F1 terendah:** 0.4516 (`Jawa_nga`)

---

## Struktur Folder

```
Handscript-Local-Project/
│
├── resultfromkaggle.ipynb       # Notebook pipeline utama (Kaggle/GPU)
├── pipeline_hcr_nusantara.ipynb # Notebook eksplorasi awal
├── clean_dataset.py             # Sanitasi nama folder & file ke ASCII
│
├── datasetscript/               # Dataset mentah
│   ├── Bali/                    #  18 kelas, 4,493 gambar (JPEG)
│   ├── Batak/                   #  19 kelas, 1,900 gambar (PNG 100×100)
│   ├── Jawa/                    #  20 kelas, 9,995 gambar
│   ├── Jawi/                    #  32 kelas, 1,600 gambar
│   ├── Kawi/                    #  31 kelas, 6,246 gambar
│   ├── Lampung/                 #  20 kelas, 4,996 gambar
│   ├── Lontara/                 #  23 kelas, 2,299 gambar (RGBA)
│   ├── Pallawa/                 #  33 kelas, 3,960 gambar
│   └── Sunda/                   #  32 kelas, 20,224 gambar
│
├── features/                    # Cache feature numpy arrays
│   ├── features_train_mobilenet_v2.npy
│   ├── features_test_mobilenet_v2.npy
│   ├── features_train_mobilenet_v2_pca256.npy
│   ├── features_test_mobilenet_v2_pca256.npy
│   ├── features_train_efficientnet_b0.npy
│   ├── features_test_efficientnet_b0.npy
│   ├── features_train_efficientnet_b0_pca256.npy
│   ├── features_test_efficientnet_b0_pca256.npy
│   ├── labels_train.npy
│   ├── labels_test.npy
│   ├── pca_mobilenet_v2.pkl
│   └── pca_efficientnet_b0.pkl
│
├── models/                      # Artefak model & visualisasi
│   ├── class_names.npy          # 228 label kelas terurut
│   ├── label_encoder.pkl        # sklearn LabelEncoder
│   ├── class_weights.png        # Visualisasi distribusi class weights
│   ├── eda_class_distribution.png
│   ├── eda_sample_images.png
│   └── eda_script_overview.png
│
├── paper/                       # Dokumen akademis
│   ├── IEEE RM Project.pdf
│   └── Proposal Final Project Machine Learning.pdf
│
├── hcr_output.zip               # Artefak output terkompresi dari Kaggle
├── CLAUDE.md                    # Dokumentasi teknis proyek
└── README.md
```

---

## Closing

Proyek ini menunjukkan bahwa pendekatan **Hybrid CNN-Ensemble dengan cost-sensitive learning** mampu mencapai **Macro F1 = 0.916** pada 228 kelas aksara lokal Indonesia — melampaui model individual terbaik sebesar +3.90 poin. Kombinasi frozen CNN (tanpa fine-tuning), PCA, class weighting, dan ensemble voting terbukti efektif menangani heterogenitas visual antar aksara sekaligus ketidakseimbangan distribusi kelas yang signifikan (imbalance ratio 14.5×).

Hasil ini diharapkan menjadi baseline terbuka untuk penelitian HCR aksara Nusantara selanjutnya.

---

*Submitted as part of Research Methodology × Machine Learning Final Project — BINUS University, 2025/2026.*
