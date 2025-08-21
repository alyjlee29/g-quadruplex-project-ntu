import argparse, sys, pandas as pd, numpy as np, joblib
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

LABEL_COL = "label_3class"
DROP_BASE = {"sample_id","sequence", LABEL_COL}
INTENSITY_TOKENS = ("intensity","rfu")

def split_features(df: pd.DataFrame):
    # drop any direct intensity-like columns to avoid leakage
    drop_cols = list(DROP_BASE.union([c for c in df.columns if any(tok in c for tok in INTENSITY_TOKENS)]))
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in object_cols if c not in DROP_BASE]
    num_cols = [c for c in df.columns if c not in drop_cols + cat_cols]
    X = df[cat_cols + num_cols].copy()
    y = df[LABEL_COL].astype(str).values
    groups = df["sample_id"].astype(str).values
    return X, y, groups, cat_cols, num_cols

def build_pipelines(cat_cols, num_cols):
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
    return models

def macro_scores(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1_macro": f1_score(y_true, y_pred, average="macro"),
        "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv", required=True, help="Path to NaCl_3class_features.csv")
    p.add_argument("--out_dir", required=True, help="Directory to write models & reports")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.in_csv)

    # Optional: drop QC fails
    # df = df[df["qc_fail"] == 0].copy()

    X, y, groups, cat_cols, num_cols = split_features(df)
    models = build_pipelines(cat_cols, num_cols)

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

            scores = macro_scores(y_va, yhat)
            f1s.append(scores["F1_macro"]); accs.append(scores["Accuracy"])
            precs.append(scores["Precision_macro"]); recs.append(scores["Recall_macro"])
            cm_sum += confusion_matrix(y_va, yhat, labels=labels)

            per_fold_rows.append({
                "model": name, "fold": fold,
                "Accuracy": scores["Accuracy"],
                "F1_macro": scores["F1_macro"],
                "Precision_macro": scores["Precision_macro"],
                "Recall_macro": scores["Recall_macro"],
                "n_val": int(len(y_va))
            })

        all_rows.append({
            "model": name, "n_folds": 5,
            "Accuracy_median": float(np.median(accs)),
            "F1_macro_median": float(np.median(f1s)),
            "Precision_macro_median": float(np.median(precs)),
            "Recall_macro_median": float(np.median(recs)),
        })
        pd.DataFrame(cm_sum, index=[f"true_{c}" for c in labels],
                              columns=[f"pred_{c}" for c in labels]
                     ).to_csv(out_dir / f"confusion_matrix_{name}.csv", index=True)

    cv_summary = pd.DataFrame(all_rows).sort_values("F1_macro_median", ascending=False)
    cv_summary.to_csv(out_dir / "cv_summary_3class.csv", index=False)
    pd.DataFrame(per_fold_rows).to_csv(out_dir / "cv_per_fold_3class.csv", index=False)

    best_name = cv_summary.iloc[0]["model"]
    best_pipe = models[best_name]
    best_pipe.fit(X, y)
    joblib.dump(best_pipe, out_dir / f"best_model_3class_{best_name}.joblib")

    # Interpretability
    if best_name in ["RandomForest","HistGB"]:
        pre = best_pipe.named_steps["prep"]
        cat_encoder = pre.named_transformers_["cat"]
        cat_features = list(cat_encoder.get_feature_names_out(cat_cols))
        feat_names = cat_features + num_cols
        clf = best_pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            imp = pd.DataFrame({"feature": feat_names, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
            imp.to_csv(out_dir / "feature_importances_3class.csv", index=False)
    else:
        pre = best_pipe.named_steps["prep"]
        cat_encoder = pre.named_transformers_["cat"]
        pre.fit(X)
        cat_features = list(cat_encoder.get_feature_names_out(cat_cols))
        num_features = num_cols
        feat_names = cat_features + num_features
        clf = best_pipe.named_steps["clf"]
        coefs = pd.DataFrame({"feature": feat_names, "coefficient": clf.coef_})  # n_classes x n_features
        coefs.to_csv(out_dir / "logistic_coefficients_3class.csv", index=False)

    with open(out_dir / "SUMMARY_3class.txt","w") as f:
        f.write(cv_summary.to_string(index=False))
        f.write(f"\n\nBest model: {best_name}\n")

    print("Saved 3-class CV results and best model to:", out_dir)

if __name__ == "__main__":
    main()
