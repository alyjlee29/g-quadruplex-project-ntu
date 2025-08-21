
# Preprocessing Guide for G4-Flanking Sequence Project

This document explains how to preprocess the **NaCl biosensor test data** and associated DNA sequence metadata for machine learning.  
The goal is to classify DNA samples into **weak** or **strong** G4 signals.

---

## 1. Understanding the Data

- **Experiment**:  
  DNA sequences containing **G-quadruplexes (G4s)** with flanking regions were tested in an **electrical biosensor** using NaCl buffer.  
  Each DNA sample was loaded into wells, and **signal intensity** values were measured.

- **Input Files**:  
  1. `20250814 test.xlsx` → raw biosensor measurement data (NaCl wells only).  
  2. `G4 with flanking sequence list.xlsx` → DNA sequences and sample names.

- **Important Notes**:  
  - Ignore KCl data (we focus only on NaCl).  
  - For NaCl 1, the highlighted abnormal row is removed. Instead, we use NaCl 2 (row D) as replacement.  
  - NaCl 1 and 2 data are **merged by averaging**.  

---

## 2. Data Cleaning

Steps:

1. **Remove invalid wells** (yellow-highlighted in NaCl 1).  
2. **Match wells to DNA sample names** (from the `sample` table).  
3. **Map sample names to DNA sequences** (from `G4 with flanking sequence list.xlsx`).  
4. **Average replicates**: If the same sample appears in NaCl 1 and 2, compute the mean intensity.  

Output: A clean dataset with columns:  

```
Sample | Sequence | Intensity (mean) | Label
```

---

## 3. Labeling (Weak vs Strong)

- Define thresholds:  
  - **Weak**: intensity below threshold  
  - **Strong**: intensity above threshold  

Thresholds can be chosen by:  
- **Percentile cutoffs** (e.g., bottom 33% = weak, top 33% = strong).  
- **Domain knowledge** (e.g., based on known G4 signal levels).  

---

## 4. Feature Engineering

Features are **numerical values derived from DNA sequences**.  
Machine learning models require numbers, not raw sequences.

Examples:  
- **Sequence length**  
- **GC content** (% of G and C bases)  
- **Number of G-runs** (consecutive G bases, important for G4)  
- **Loop lengths** (distance between G-runs in G4)  
- **Nucleotide composition** (A%, T%, G%, C%)  
- **K-mer frequencies** (counts of subsequences of length k, e.g., 2-mers, 3-mers)  

These become columns in the training dataset.

---

## 5. Final Dataset Structure

The merged preprocessed dataset (`NaCl_model_matrix_STRICT_features_QC.csv`) should look like:

```
Sample | Sequence | GC_Content | Num_G_runs | Loop1_len | Loop2_len | Loop3_len | Intensity | Label
```

- **Features (X)**: GC_Content, Num_G_runs, Loop lengths, etc.  
- **Target (y)**: Label (Weak or Strong).  

---

## 6. Machine Learning Pipeline

1. **Data Split**:  
   - Training set (80%)  
   - Test set (20%)  

2. **Normalization**:  
   - Scale numeric features (e.g., 0–1 range).  

3. **Models to Train**:  
   - **Logistic Regression** → baseline linear model.  
   - **Random Forest** → captures nonlinear sequence effects.  
   - **XGBoost** → powerful gradient boosting for tabular data.  

4. **Evaluation**:  
   - Accuracy, Precision, Recall, F1-score.  
   - Confusion matrix (Weak vs Strong predictions).  

---

## 7. Biological Interpretation

- **What a Feature Means in Biology**:  
  - GC Content → higher GC stabilizes G4s.  
  - G-runs → more consecutive Gs = stronger G4.  
  - Loop lengths → affect folding stability.  

- **Why Predict Weak/Strong?**  
  If we can **predict G4 strength from sequence**, future DNA designs can be screened computationally before running wet-lab experiments.

---

## 8. Deliverables

1. **Clean dataset** (`NaCl_model_matrix_STRICT_features_QC.csv`)  
2. **Feature extraction code** (sequence → numerical features)  
3. **Preprocessing pipeline** (scaling, splitting)  
4. **Models trained & compared**  

---

✅ You are now ready to train machine learning models to predict **weak vs strong G4 activity** from DNA sequences!
