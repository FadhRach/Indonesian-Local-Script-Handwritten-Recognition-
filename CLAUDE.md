# Handscript HCR Nusantara — Project Core

## Project Overview
**Title:** A Hybrid Cost Sensitive Ensemble Approach for Indonesian Local Script Handwritten Recognition  
**Course:** Research Methodology × Machine Learning (BINUS University)  
**Goal:** Multi-class HCR across 9 Indonesian local scripts using CNN feature extraction + ensemble ML

## Research Gap Addressed
- No prior study covers all 9 Nusantara scripts in a single standardized experiment
- Class imbalance has not been addressed via cost-sensitive learning in multi-script HCR
- No prior work combines frozen CNN feature extraction with gradient-boosting ensemble classifiers

## Pipeline Architecture (Hybrid CNN-Ensemble)
```
Raw Images → Preprocessing → Feature Extraction (frozen CNN) → Class Weight Balancing
                                                                        ↓
Dataset Splitting ← Data Augmentation ← EDA             XGBoost / LightGBM / CatBoost
       ↓                                                        ↓
  Train/Test Sets ————————————————————————————→ Hyperparameter Tuning → Final Model
                                                        ↓
                                              Evaluation (Macro F1, Precision, Recall, Confusion Matrix)
                                                        ↓
                                              Streamlit/FastAPI Deployment
```

## Dataset
**Source:** Mendeley Data — Indonesian Local Script Characters Handwriting Dataset (Ihsan, 2024)  
**Local path:** `./datasetscript/`  
**Total:** 55,713 images | 228 classes | 9 scripts

| Script | Folder | Classes | Images | Notes |
|--------|--------|---------|--------|-------|
| Bali   | `Bali/` | 18 | 4,493 | JPEG |
| Batak  | `Batak/` | 19 | 1,900 | PNG 100×100 |
| Jawa   | `Jawa/` | 20 | 9,995 | PNG (filenames sanitised) |
| Jawi   | `Jawi/` | 32 | 1,600 | PNG |
| Kawi   | `Kawi/` | 31 | 6,246 | PNG (diacritic folders → ASCII) |
| Lampung| `Lampung/` | 20 | 4,996 | PNG (filenames sanitised) |
| Lontara| `Lontara/` | 23 | 2,299 | PNG 100×100 RGBA |
| Pallawa| `Pallawa/` | 33 | 3,960 | PNG (diacritic folders → ASCII) |
| Sunda  | `Sunda/` | 32 | 20,224 | PNG (filenames sanitised, é→etaling) |

**Dataset sanitised by `clean_dataset.py`** — all folder and file names are ASCII-safe:
- `name (N).ext` → `name_N.ext` (35,215 files: Jawa, Lampung, Sunda)
- `é (N).png` / `é_N.png` → `etaling_N.png` (623 files: Sunda)
- Unicode diacritic folders → ASCII: `ḍa`→`dda`, `ṭa`→`tta`, `ṣa`→`ssa`, `śa`→`sha`, `ṅa`→`nga`, `ṇa`→`nna`, `ña`→`nya`, `ḍha`→`ddha`, `ṭha`→`ttha`, `é`→`etaling`

**Label format:** `{Script}_{classname}` → 228 unique labels total  
**Class imbalance:** Sunda (36.3%) vs Jawi (2.87%) — requires cost-sensitive weighting

## Key Technical Decisions
- **Image standardization:** 224×224 grayscale → binarize → replicate to 3-ch RGB → normalize ImageNet stats
- **Feature extractor:** MobileNetV2 (primary) or EfficientNet-B0 (comparison), frozen, no fine-tuning
- **Feature dim:** 1280-d (MobileNetV2) or 1280-d (EfficientNet-B0) per image
- **Class weights:** `sklearn.utils.class_weight.compute_class_weight('balanced', ...)`
- **Split:** Stratified 80/20 (train/test), random_state=42
- **Ensemble models:** XGBoost (`mlogloss`), LightGBM (`multi_logloss`), CatBoost (`MultiClass`)
- **Primary metric:** Macro F1-Score (equal weight per class, handles imbalance)
- **Deployment:** Streamlit app

## Hyperparameter Search Space (Table II from paper)
| Param | XGBoost | LightGBM | CatBoost |
|-------|---------|----------|----------|
| n_estimators | 300–500 | 300–500 | 300–500 |
| max_depth | 6–10 | leaf-wise | 6–10 |
| learning_rate | 0.05–0.2 | 0.05–0.2 | 0.05–0.2 |
| subsample | 0.7–1.0 | 0.7–1.0 | — |
| colsample_bytree | 0.7–1.0 | 0.7–1.0 | — |
| num_leaves | — | 31–127 | — |
| l2_leaf_reg | — | — | 1–10 |
| reg_alpha/lambda | 0–1 | 0–1 | — |
| random_state | 42 | 42 | 42 |

## Environment
- **Platform:** macOS M1 (Apple Silicon) — use `torch.device("mps")` for GPU acceleration
- **Conda env:** `ai_core`
- **Notebook:** `pipeline_hcr_nusantara.ipynb`

## Pre-upload Steps (Kaggle)
1. Run `python3 clean_dataset.py` once — already done, idempotent if re-run.
2. Zip `datasetscript/` and upload as a Kaggle Dataset.
3. In the notebook, set Accelerator → **GPU T4 x2**, Internet → **On**.

## Output Artifacts
- `features_train.npy`, `features_test.npy`, `labels_train.npy`, `labels_test.npy` — cached features
- `class_names.npy` — ordered class label array
- `models/xgb_model.pkl`, `models/lgbm_model.pkl`, `models/cat_model.pkl` — trained models
- `models/label_encoder.pkl` — label encoder
- `models/feature_extractor_name.txt` — which CNN backbone was used
- `app.py` — Streamlit inference app
