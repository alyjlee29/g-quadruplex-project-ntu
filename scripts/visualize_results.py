
"""
visualize_results.py
--------------------
- Pre‑ML: draw NaCl/KCl heatmaps from averaged_by_well_NaCl_KCl.csv
- Post‑ML: draw grouped bar charts of mean predicted probabilities per plate row (A–H)

Inputs:
- data_processed/averaged_by_well_NaCl_KCl.csv
- models/3class/predictions.csv

Outputs:
- figures/heatmap_NaCl_avg.png
- figures/heatmap_KCl_avg.png
- figures/bar_means_pred_probs.png
"""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

def pivot_heat(df, value_col):
    mat = (df.pivot(index="row", columns="col", values=value_col)
             .reindex(index=list("ABCDEFGH"))
             .reindex(columns=list(range(1,13))))
    return mat

def main():
    project_root = Path(".")
    proc = project_root / "data_processed"
    figs = project_root / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    plate = pd.read_csv(proc / "averaged_by_well_NaCl_KCl.csv")

    # Heatmaps
    for col, title, fname in [("NaCl_avg","NaCl averaged intensity (by well)","heatmap_NaCl_avg.png"),
                              ("KCl_avg","KCl averaged intensity (by well)","heatmap_KCl_avg.png")]:
        mat = pivot_heat(plate, col)
        plt.figure(figsize=(10,6))
        im = plt.imshow(mat.values, aspect="auto")
        plt.xticks(ticks=np.arange(12), labels=[str(i) for i in range(1,13)])
        plt.yticks(ticks=np.arange(8), labels=list("ABCDEFGH"))
        plt.title(title)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(figs / fname, dpi=200)
        plt.close()

    # Post‑ML bar charts (if predictions exist)
    pred_path = project_root / "models" / "3class" / "predictions.csv"
    if pred_path.exists():
        preds = pd.read_csv(pred_path)
        df = plate.merge(preds, on="sample_id", how="inner")
        # keep condition consistency (NaCl rows with NaCl predictions, KCl with KCl)
        df = df[df["condition_x"] == df["condition_y"]].copy()
        df = df.rename(columns={"condition_x":"condition"})
        prob_cols = [c for c in df.columns if c.startswith("p_")]
        # Average per row x condition
        agg = df.groupby(["row","condition"], as_index=False)[prob_cols].mean()

        # One figure per condition
        for cond in agg["condition"].unique():
            sub = agg[agg["condition"] == cond].sort_values("row")
            x = np.arange(len(sub["row"]))
            plt.figure(figsize=(10,5))
            width = 0.25
            plt.bar(x - width, sub["p_weak"],  width=width, label="weak")
            plt.bar(x,          sub["p_mild"],  width=width, label="mild")
            plt.bar(x + width,  sub["p_strong"],width=width, label="strong")
            plt.xticks(x, sub["row"])
            plt.ylabel("Mean predicted probability")
            plt.title(f"Predicted class probabilities by group (row) — {cond}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(figs / f"bar_means_pred_probs_{cond}.png", dpi=200)
            plt.close()

    print("Saved figures to", figs)

if __name__ == "__main__":
    main()
