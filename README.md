# G4·Hemin Biosensor — 3‑Class Pipeline (Weak / Mild / Strong)

_Last updated: 2025-08-21_

This repository processes G‑quadruplex (G4) DNA biosensor data measured in **NaCl** and **KCl**, builds
**sequence‑derived features** (loops, flanks, G‑tracts), and trains **three machine‑learning models**
(HistGradientBoosting, RandomForest, Multinomial Logistic) to predict **weak / mild / strong** signal classes.
It also produces **plate heatmaps** and **grouped bar plots** aligned to your plate layout.

## Folder layout (recommended)

```
project_root/
├─ data_raw/                         # original inputs (never edited)
│  ├─ 20250814_test.xlsx
│  └─ G4_with_flanking_sequence_list.xlsx
├─ data_processed/                   # tidy tables & matrices derived from raw
│  ├─ averaged_by_well_NaCl_KCl.csv
│  ├─ grouped_by_row_summary.csv
│  ├─ heatmap_matrix_NaCl.csv
│  └─ heatmap_matrix_KCl.csv
├─ figures/                          # exported plots (png, pdf)
│  ├─ heatmap_NaCl_avg.png
│  └─ heatmap_KCl_avg.png
├─ src/                              # code
│  ├─ build_dataset_3class.py
│  └─ train_models_3class.py
├─ models/                           # saved pipelines & CV outputs
│  └─ ML_3class_outputs/
└─ README.md
```

> Put the files generated here into the corresponding folders; update paths as needed.

## Data sources
- **Plate & measurements**: `data_raw/20250814_test.xlsx` (plate map on "Data processing").
- **Sequences**: `data_raw/G4_with_flanking_sequence_list.xlsx` (first two cols: sample_id, sequence).

## Replicate merge rules
- **NaCl**: average replicate 1 & 2, but ignore entire **row D** from NaCl₁ and use row D from NaCl₂ instead.
- **KCl**: average replicate 1 & 2 normally.

Outputs are stored in `data_processed/averaged_by_well_NaCl_KCl.csv` with:
`row, col, well, sample_id, sequence, NaCl_avg, KCl_avg, is_control`.

## Grouped lists of sample names
Two artifacts for the plate groups (rows A–H):
- CSV: `data_processed/grouped_samples_by_row.csv` (row, well, col, sample_id, sequence)
- JSON: `data_processed/grouped_samples.json`:
  ```json
  {
    "A": [{"well":"A1","sample_id":"..."}],
    "B": [...],
    "...": "...",
    "H": [{"well":"H11","sample_id":"Ctrl0002"}, {"well":"H12","sample_id":"Ctrl0001"}]
  }
  ```

## Build the 3‑class dataset
```bash
python src/build_dataset_3class.py   --model_matrix data_processed/averaged_by_well_NaCl_KCl.csv   --seq_list data_raw/G4_with_flanking_sequence_list.xlsx   --out_csv data_processed/NaCl_3class_features.csv
```

## Train models and evaluate
```bash
python src/train_models_3class.py   --in_csv data_processed/NaCl_3class_features.csv   --out_dir models/ML_3class_outputs
```

Outputs include: CV summaries, per‑fold metrics, confusion matrices, best pipeline (`.joblib`),
and feature importances/coefficients.

## Post‑ML plots
Merge predictions with `averaged_by_well_NaCl_KCl.csv` by `sample_id`, aggregate by `row` (A–H), and draw bar charts of mean class probabilities.

## Reproducibility
GroupKFold by `sample_id`; one‑hot for categoricals; numerics scaled for logistic; intensity-like columns excluded from predictors.

