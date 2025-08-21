
"""
train_model.py
--------------
Train 3-class classifiers on the encoded, section-padded model matrix.
Uses GroupKFold by sample_id to avoid leakage across conditions replicates.

Input:
- data_processed/model_matrix_3class.csv (from feature_build.py)

Outputs (to models/3class/):
- cv_summary.csv
- cv_per_fold.csv
- confusion_matrix_<Model>.csv
- best_model_<Model>.joblib
- predictions.csv (per sample_id x condition: class probabilities + predicted class)
"""
from pathlib import Path
import pandas as pd, numpy as np, joblib
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

LABEL_COL = "label_3class"

def macro_scores(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1_macro": f1_score(y_true, y_pred, average="macro"),
        "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

def main():
    project_root = Path(".")
    proc = project_root / "data_processed"
    mm_csv = proc / "model_matrix_3class.csv"
    outdir = project_root / "models" / "3class"
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(mm_csv)

    # Build feature sets
    drop_cols = {"sample_id","intensity",LABEL_COL}
    # condition is categorical; all other columns are numeric ints from encoding
    cat_cols = ["condition"]
    num_cols = [c for c in df.columns if c not in drop_cols.union(cat_cols)]

    X = df[cat_cols + num_cols].copy()
    y = df[LABEL_COL].astype(str).values
    groups = df["sample_id"].astype(str).values

    # Pipelines
    prep_tree = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
                                  remainder="passthrough")
    prep_logit = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("scale", StandardScaler(with_mean=False), num_cols)
    ], remainder="drop")

    models = {
        "HistGB": Pipeline([("prep", prep_tree),
                            ("clf", HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31,
                                                                   learning_rate=0.07, min_samples_leaf=20,
                                                                   random_state=42))]),
        "RandomForest": Pipeline([("prep", prep_tree),
                                  ("clf", RandomForestClassifier(n_estimators=600, min_samples_leaf=8,
                                                                 class_weight="balanced_subsample",
                                                                 n_jobs=-1, random_state=42))]),
        "Logistic": Pipeline([("prep", prep_logit),
                              ("clf", LogisticRegression(max_iter=5000, C=1.0,
                                                         multi_class="multinomial", class_weight="balanced",
                                                         n_jobs=-1, solver="lbfgs"))]),
    }

    gkf = GroupKFold(n_splits=5)
    labels = np.unique(y)

    all_rows, per_fold_rows = [], []
    for name, pipe in models.items():
        f1s, accs, precs, recs = [], [], [], []
        cm_sum = np.zeros((len(labels), len(labels)), dtype=int)

        for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y[tr], y[va]
            pipe.fit(X_tr, y_tr)
            yhat = pipe.predict(X_va)

            sc = macro_scores(y_va, yhat)
            f1s.append(sc["F1_macro"]); accs.append(sc["Accuracy"])
            precs.append(sc["Precision_macro"]); recs.append(sc["Recall_macro"])
            cm_sum += confusion_matrix(y_va, yhat, labels=labels)

            per_fold_rows.append({"model":name,"fold":fold, **sc, "n_val":int(len(y_va))})

        all_rows.append({
            "model": name, "n_folds": 5,
            "Accuracy_median": float(np.median(accs)),
            "F1_macro_median": float(np.median(f1s)),
            "Precision_macro_median": float(np.median(precs)),
            "Recall_macro_median": float(np.median(recs)),
        })
        pd.DataFrame(cm_sum, index=[f"true_{c}" for c in labels],
                              columns=[f"pred_{c}" for c in labels]).to_csv(outdir / f"confusion_matrix_{name}.csv", index=True)

    cv_summary = pd.DataFrame(all_rows).sort_values("F1_macro_median", ascending=False)
    cv_summary.to_csv(outdir / "cv_summary.csv", index=False)
    pd.DataFrame(per_fold_rows).to_csv(outdir / "cv_per_fold.csv", index=False)

    # Fit the best on all data and save predictions
    best_name = cv_summary.iloc[0]["model"]
    best_pipe = models[best_name]
    best_pipe.fit(X, y)
    joblib.dump(best_pipe, outdir / f"best_model_{best_name}.joblib")

    # Predict per-row probabilities
    yhat = best_pipe.predict(X)
    preds = pd.DataFrame(best_pipe.predict_proba(X), columns=[f"p_{c}" for c in best_pipe.classes_])
    preds.insert(0, "predicted_class", yhat)
    preds.insert(0, "condition", df["condition"].values)
    preds.insert(0, "sample_id", df["sample_id"].values)
    preds.to_csv(outdir / "predictions.csv", index=False)

    print("Saved CV results and best model to:", outdir)

if __name__ == "__main__":
    main()
