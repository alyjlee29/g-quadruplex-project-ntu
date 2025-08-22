"""
visualize_results.py — robust against duplicate wells
- Pre-ML: draw NaCl/KCl heatmaps from averaged_by_well_NaCl_KCl.csv (deduped by mean)
- Post-ML: draw grouped bar charts of mean predicted probabilities per plate row (A–H)

Inputs:
- data_processed/averaged_by_well_NaCl_KCl.csv
- models/3class/predictions.csv  (optional, for bars)

Outputs:
- figures/heatmap_NaCl_avg.png
- figures/heatmap_KCl_avg.png
- figures/bar_means_pred_probs_<COND>.png
- data_processed/averaged_by_well_dedup.csv  (helper)
"""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

ROWS = list("ABCDEFGH")
COLS = list(range(1, 13))

def clean_plate(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize and coerce types
    df = df.copy()
    df["row"] = df["row"].astype(str).str.strip().str.upper()
    df["col"] = pd.to_numeric(df["col"], errors="coerce").astype("Int64")
    df = df[df["row"].isin(ROWS)]
    df = df[df["col"].isin(COLS)]
    # Keep only needed columns
    keep = ["row","col","well","sample_id","NaCl_avg","KCl_avg"]
    exist = [c for c in keep if c in df.columns]
    df = df[exist]
    return df

def dedupe_by_mean(df: pd.DataFrame) -> pd.DataFrame:
    # Count dupes
    dupe_mask = df.duplicated(subset=["row","col"], keep=False)
    n_dupes = int(dupe_mask.sum())
    if n_dupes:
        print(f"[visualize] Found {n_dupes} duplicate well rows; aggregating mean by (row,col).")
    agg = (df.groupby(["row","col"], as_index=False)
             .agg(NaCl_avg=("NaCl_avg","mean"),
                  KCl_avg =("KCl_avg","mean")))
    return agg

def heatmap_matrix(agg: pd.DataFrame, value_col: str) -> pd.DataFrame:
    mat = (agg.pivot_table(index="row", columns="col", values=value_col, aggfunc="mean")
              .reindex(index=ROWS)
              .reindex(columns=COLS))
    return mat

def plot_heatmap(mat: pd.DataFrame, title: str, outpath: Path):
    plt.figure(figsize=(10, 6))
    im = plt.imshow(mat.values, aspect="auto")
    plt.xticks(ticks=np.arange(len(COLS)), labels=[str(c) for c in COLS])
    plt.yticks(ticks=np.arange(len(ROWS)), labels=ROWS)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    root = Path(".")
    proc = root / "data_processed"
    figs = root / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    plate_path = proc / "averaged_by_well_NaCl_KCl.csv"
    if not plate_path.exists():
        raise FileNotFoundError(f"{plate_path} not found. Run scripts/preprocess.py first.")

    plate = pd.read_csv(plate_path)
    plate = clean_plate(plate)
    agg = dedupe_by_mean(plate)

    # Save a helper deduped file for inspection
    dedup_path = proc / "averaged_by_well_dedup.csv"
    agg.to_csv(dedup_path, index=False)
    print(f"[visualize] Wrote deduped wells to {dedup_path}")

    # Heatmaps
    for col, title, fname in [
        ("NaCl_avg","NaCl averaged intensity (by well)","heatmap_NaCl_avg.png"),
        ("KCl_avg","KCl averaged intensity (by well)","heatmap_KCl_avg.png"),
    ]:
        if col not in agg.columns:
            print(f"[visualize] WARNING: {col} not in data; skipping heatmap.")
            continue
        mat = heatmap_matrix(agg, col)
        plot_heatmap(mat, title, figs / fname)
        # also save matrix CSV for convenience
        mat.to_csv(proc / f"heatmap_matrix_{'NaCl' if col=='NaCl_avg' else 'KCl'}.csv")

    # Post-ML bar charts (if predictions exist)
    pred_path = root / "models" / "3class" / "predictions.csv"
    if pred_path.exists():
        preds = pd.read_csv(pred_path)
        # Merge by sample_id; keep matching condition between plate & predictions
        # We need the per-well "condition" mapping: derive it from which average exists
        long = []
        for cond, valcol in [("NaCl","NaCl_avg"), ("KCl","KCl_avg")]:
            tmp = plate[["row","col","sample_id",valcol]].copy()
            tmp = tmp.rename(columns={valcol:"intensity"})
            tmp["condition"] = cond
            long.append(tmp)
        plate_long = pd.concat(long, ignore_index=True)

        df = plate_long.merge(preds, on=["sample_id","condition"], how="inner")
        prob_cols = [c for c in df.columns if c.startswith("p_")]
        if not prob_cols:
            print("[visualize] No probability columns found in predictions.csv; skipping bars.")
            return

        # Average probabilities by plate row & condition
        agg_probs = df.groupby(["row","condition"], as_index=False)[prob_cols].mean()
        for cond in agg_probs["condition"].unique():
            sub = agg_probs[agg_probs["condition"] == cond].sort_values("row")
            x = np.arange(len(sub["row"]))
            plt.figure(figsize=(10,5))
            width = 0.25
            # Ensure the expected class order; fall back to sorting if needed
            classes = ["p_weak","p_mild","p_strong"]
            have = [c for c in classes if c in prob_cols]
            if not have:
                have = sorted(prob_cols)
            # Plot three adjacent bars
            for i, c in enumerate(have):
                plt.bar(x + (i - (len(have)-1)/2)*width, sub[c], width=width, label=c.replace("p_",""))
            plt.xticks(x, sub["row"])
            plt.ylabel("Mean predicted probability")
            plt.title(f"Predicted class probabilities by group (row) — {cond}")
            plt.legend(title="Class")
            plt.tight_layout()
            out = figs / f"bar_means_pred_probs_{cond}.png"
            plt.savefig(out, dpi=200)
            plt.close()
            print(f"[visualize] Wrote {out}")
    else:
        print("[visualize] No predictions.csv found — generated heatmaps only.")

if __name__ == "__main__":
    main()
