import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# -------------------- Helpers --------------------

def label_from_intensity(x: float) -> str:
    """Weak ≤ 80, Mild (80,135), Strong ≥ 135"""
    if x <= 80:
        return "weak"
    if x >= 135:
        return "strong"
    return "mild"


def parse_loop_logic(name: str) -> dict:
    """
    Parse sample_name into loop descriptors.
      - '111T' / '222A' / '333C' ...        -> same_nucleotide
      - '121T' / '212A' ...                 -> different_length_combo
      - 'L1A' / 'L2T' ...                   -> point_mutation
    """
    token = re.sub(r'[^A-Za-z0-9]', '', str(name)).upper()

    m_simple = re.match(r'^(?P<core>\d{3})(?P<nt>[ATCG])', token)
    if m_simple:
        core = m_simple.group('core')
        nt = m_simple.group('nt')
        cat = "same_nucleotide" if len(set(core)) == 1 else "different_length_combo"
        return {"loop_core": core, "loop_nt": nt, "category": cat}

    m_point = re.match(r'^L(?P<which>\d)(?P<nt>[ATCG])', token)
    if m_point:
        core = f"L{m_point.group('which')}"
        return {"loop_core": core, "loop_nt": m_point.group('nt'), "category": "point_mutation"}

    if len(token) >= 2 and token[-1] in "ATCG":
        return {"loop_core": token[:-1], "loop_nt": token[-1], "category": "other"}

    return {"loop_core": token, "loop_nt": "", "category": "other"}


def build_features(seq: str) -> dict:
    s = str(seq).upper().replace(" ", "")
    length = len(s) if s else 0
    gc = (s.count('G') + s.count('C')) / length if length else 0.0
    a_count = s.count('A')
    t_count = s.count('T')
    c_count = s.count('C')
    g_count = s.count('G')
    g_runs = len(re.findall(r'G{3,}', s))                       # count of G-tracts (>=3)
    max_g_run = max([len(m.group()) for m in re.finditer(r'G+', s)], default=0)
    return {
        "seq_len": length,
        "gc_content": gc,
        "a_count": a_count,
        "t_count": t_count,
        "c_count": c_count,
        "g_count": g_count,
        "g_tracts_ge3": g_runs,
        "max_g_run": max_g_run,
    }


def train_and_eval(df, target_col: str, condition_name: str, out_dir: Path) -> pd.DataFrame:
    """
    Train three models to predict labels from sequence-derived features (no intensity leakage).
    Save:
      - accuracies_{condition}.csv
      - accuracy_bar_{condition}.png
      - classification_report_{condition}_{Model}.csv
      - confusion_matrix_{condition}_{Model}.png
    """
    feat_cols = ["seq_len","gc_content","a_count","t_count","c_count","g_count","g_tracts_ge3","max_g_run"]
    X = df[feat_cols].values
    y = df[target_col].values

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
        # LogisticRegression default (multinomial from sklearn>=1.5) – leave multi_class default to avoid warnings
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000))
        ])
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    records = []
    for name, model in models.items():
        # Cross-validated accuracy (scalar per fold)
        scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
        records.append({
            "condition": condition_name,
            "model": name,
            "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores)),
            "fold_accuracies": [float(s) for s in scores]
        })

        # Cross-validated predictions for diagnostics (precision/recall/F1 + confusion matrix)
        y_pred = cross_val_predict(model, X, y, cv=skf)

        # Classification report (per-class precision/recall/F1 + support)
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(out_dir / f"classification_report_{condition_name}_{name}.csv")

        # Confusion matrix plot
        labels_order = ["weak", "mild", "strong"]
        cm = confusion_matrix(y, y_pred, labels=labels_order)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Weak","Mild","Strong"])
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix — {condition_name} ({name})")
        plt.tight_layout()
        plt.savefig(out_dir / f"confusion_matrix_{condition_name}_{name}.png", dpi=200)
        plt.close()

    # Save accuracy table & plot
    acc_df = pd.DataFrame.from_records(records)
    acc_df.to_csv(out_dir / f"accuracies_{condition_name}.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(acc_df["model"], acc_df["mean_accuracy"] * 100.0,
            yerr=acc_df["std_accuracy"] * 100.0, capsize=4)
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title(f"Model Accuracy — {condition_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"accuracy_bar_{condition_name}.png", dpi=200)
    plt.close()

    return acc_df


def intensity_bar(df, intensity_col: str, condition_name: str, out_dir: Path):
    """Plot intensity by loop logic (sample_name). y-axis = intensity (a.u.)."""
    df2 = df.copy()
    df2["label"] = df2["sample_name"]
    df2 = df2.sort_values(by=["category", "loop_core", "loop_nt", "label"])

    plt.figure(figsize=(12, 5))
    plt.bar(df2["label"], df2[intensity_col])
    plt.xticks(rotation=90)
    plt.ylabel("Intensity (a.u.)")
    plt.title(f"Intensity by Loop Logic — {condition_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"intensity_by_loop_{condition_name}.png", dpi=220)
    plt.close()


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="G4 ML 3-class pipeline (bundled)")
    ap.add_argument("--in_csv", required=False,
                    help="Input CSV with columns: sample_name, sequence, nacl_value, kcl_value")
    ap.add_argument("--out_dir", default="g4_outputs", help="Directory to write results")
    args = ap.parse_args()

    # Default CSV path if none provided
    if args.in_csv is None:
        default_csv = Path(__file__).parent / "ntu - sequence_mapped.csv"
        if default_csv.exists():
            args.in_csv = str(default_csv)
            print(f"[INFO] Using default CSV: {args.in_csv}")
        else:
            raise FileNotFoundError("No --in_csv given and default 'ntu - sequence_mapped.csv' not found.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.in_csv)

    # Parse loop logic + build sequence features
    logic = df["sample_name"].apply(parse_loop_logic).apply(pd.Series)
    feats = df["sequence"].apply(build_features).apply(pd.Series)
    df_proc = pd.concat([df, logic, feats], axis=1)

    # Labels per condition (separate NaCl/KCl training)
    df_proc["label_NaCl"] = df_proc["nacl_value"].apply(label_from_intensity)
    df_proc["label_KCl"]  = df_proc["kcl_value"].apply(label_from_intensity)

    # Save processed features table for reproducibility
    df_proc.to_csv(out_dir / "processed_dataset.csv", index=False)

    # Train & evaluate (NaCl, KCl)
    acc_nacl = train_and_eval(df_proc, "label_NaCl", "NaCl", out_dir)
    acc_kcl  = train_and_eval(df_proc, "label_KCl",  "KCl",  out_dir)

    # Intensity plots (y-axis = intensity)
    intensity_bar(df_proc, "nacl_value", "NaCl", out_dir)
    intensity_bar(df_proc, "kcl_value", "KCl", out_dir)

    # Print quick summaries to console
    print("\n== ACCURACY (NaCl) ==")
    print(acc_nacl[["model","mean_accuracy","std_accuracy"]])
    print("\n== ACCURACY (KCl) ==")
    print(acc_kcl[["model","mean_accuracy","std_accuracy"]])


if __name__ == "__main__":
    main()
