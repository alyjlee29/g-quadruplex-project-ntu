"""
visualize_by_looping.py
Group samples into biology-defined LOOP patterns inferred from sample_id
and generate two families of bar charts for both NaCl and KCl:
  1) Mean predicted probabilities per group (requires predictions.csv)
  2) Mean measured intensity per group (no ML required)

Inputs:
  data_processed/averaged_by_well_NaCl_KCl.csv
  models/3class/predictions.csv (only for probability charts)

Outputs:
  figures/bar_loopgroups_pred_probs_<COND>.png
  data_processed/loop_group_means_<COND>.csv
  figures/bar_loopgroups_intensity_<COND>.png
  data_processed/loop_group_intensity_means_<COND>.csv
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Parsing rules (regex) ----------
RE_SINGLE = re.compile(r"^L([123])([ATCG])$")              # L1A, L2C, ... (point mutation in loop)
RE_DOUBLET = re.compile(r"^L([123])(AA|AT|TA|CC|CT|TC)$")  # L1AA, L2CT, L3TA, ...
RE_LENGTH = re.compile(r"^(\d{3})(AF|TF|A|T)?$")           # 111T, 121T, 222A, 333C, 211AF, 111TF
RE_FLANK  = re.compile(r"^RF(\d+)([ACGT])?$")              # RF11C, RF222C (flank variant codes)
RE_CTRL   = re.compile(r"^CTRL\d+$", re.IGNORECASE)

def classify_loop_group(sample_id: str) -> dict:
    """Return dict with display group and a family key for ordering.

    Examples:
      111T     -> {group: "111T", family: "Len"}
      121T     -> {group: "121T", family: "Len"}
      L1A      -> {group: "L1A",  family: "SingleSwap"}
      L2CT     -> {group: "L2CT", family: "Doublet"}
      RF11C    -> {group: "RF11C",family: "Flank"}
    """
    sid = str(sample_id).strip()
    u = sid.upper()

    if RE_CTRL.match(u):
        return {"group":"Control", "family":"Control"}

    m = RE_FLANK.match(u)
    if m:
        length_code = m.group(1)  # e.g., 11, 222
        base = m.group(2) or ""
        return {"group": f"RF{length_code}{base}", "family": "Flank"}

    m = RE_SINGLE.match(u)
    if m:
        loop_idx = m.group(1)  # 1/2/3
        base = m.group(2)
        return {"group": f"L{loop_idx}{base}", "family": "SingleSwap"}

    m = RE_DOUBLET.match(u)
    if m:
        loop_idx = m.group(1)
        motif = m.group(2)     # AA/AT/TA/CC/CT/TC
        return {"group": f"L{loop_idx}{motif}", "family": "Doublet"}

    m = RE_LENGTH.match(u)
    if m:
        len_code = m.group(1)       # 3-digit loop-length code
        bias = m.group(2) or ""     # AF/TF/A/T or none
        return {"group": f"{len_code}{bias}", "family": "Len"}

    return {"group":"Other", "family":"Other"}

# ---------- Plotting ----------
def _order_tuple_for_group(group: str, family: str):
    # Natural ordering across families: SingleSwap -> Doublet -> Len -> Flank -> Control -> Other
    fam_rank = {"SingleSwap":0, "Doublet":1, "Len":2, "Flank":3, "Control":4}
    base_rank = fam_rank.get(family, 9)
    if family in ("SingleSwap","Doublet"):
        # groups like L1A, L2CT
        m = re.match(r"^L([123])", group)
        loop_order = int(m.group(1)) if m else 9
        return (base_rank, loop_order, group)
    if family == "Len":
        # groups like 111T, 121T, 222A, 333C, 211AF
        m = re.match(r"^(\d{3})([A-Z]+)?$", group)
        if m:
            return (base_rank, int(m.group(1)), m.group(2) or "", group)
        return (base_rank, group)
    return (base_rank, group)

def barplot_loop_groups(df: pd.DataFrame, cond: str, out_png: Path, out_csv: Path):
    # df must have: group, family, p_weak, p_mild, p_strong
    gmeans = (df.groupby(["group","family"], as_index=False)[["p_weak","p_mild","p_strong"]].mean())
    gmeans = gmeans.sort_values(by=["group","family"], key=lambda s: s.map(lambda x: _order_tuple_for_group(gmeans.loc[s.index, "group"], gmeans.loc[s.index, "family"])) if s.name=="group" else s)
    gmeans.to_csv(out_csv, index=False)

    x = np.arange(len(gmeans))
    width = 0.25
    plt.figure(figsize=(12,6))
    plt.bar(x - width, gmeans["p_weak"],  width=width, label="weak")
    plt.bar(x,          gmeans["p_mild"], width=width, label="mild")
    plt.bar(x + width,  gmeans["p_strong"], width=width, label="strong")
    plt.xticks(x, gmeans["group"], rotation=20, ha="right")
    plt.ylabel("Mean predicted probability")
    plt.title(f"Predicted class probabilities by LOOP group — {cond}")
    plt.legend(title="Class")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[looping] Wrote {out_png} and {out_csv}")

def barplot_intensity_groups(df: pd.DataFrame, cond: str, out_png: Path, out_csv: Path):
    # df must have: group, family, intensity
    gmeans = (df.groupby(["group","family"], as_index=False)[["intensity"]].mean())
    # Sort using the same natural order
    gmeans = gmeans.sort_values(by=["group","family"], key=lambda s: s.map(lambda x: _order_tuple_for_group(gmeans.loc[s.index, "group"], gmeans.loc[s.index, "family"])) if s.name=="group" else s)
    gmeans.to_csv(out_csv, index=False)

    x = np.arange(len(gmeans))
    plt.figure(figsize=(12,6))
    plt.bar(x, gmeans["intensity"], width=0.6, color="#4C78A8")
    plt.xticks(x, gmeans["group"], rotation=20, ha="right")
    plt.ylabel("Mean measured intensity")
    plt.title(f"Measured intensity by LOOP group — {cond}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[looping] Wrote {out_png} and {out_csv}")

def main():
    root = Path(".")
    proc = root / "data_processed"
    figs = root / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    # Load per-well table and predictions
    plate_path = proc / "averaged_by_well_NaCl_KCl.csv"
    pred_path  = root / "models" / "3class" / "predictions.csv"

    if not plate_path.exists():
        raise FileNotFoundError(f"{plate_path} not found. Run scripts/preprocess.py first.")
    if not pred_path.exists():
        raise FileNotFoundError(f"{pred_path} not found. Run scripts/train_model.py first.")

    plate = pd.read_csv(plate_path)[["sample_id","NaCl_avg","KCl_avg"]]
    # Aggregate to per-sample mean intensity to avoid multiple wells biasing groups
    plate_means = (plate.groupby("sample_id", as_index=False)[["NaCl_avg","KCl_avg"]].mean())

    # Classification for each sample_id
    groups_plate = plate_means["sample_id"].astype(str).apply(classify_loop_group).apply(pd.Series)
    plate_means = pd.concat([plate_means, groups_plate], axis=1)

    # Build long intensity by condition
    long_intensity = []
    for cond, col in [("NaCl","NaCl_avg"),("KCl","KCl_avg")]:
        tmp = plate_means[["sample_id","group","family", col]].copy()
        tmp = tmp.rename(columns={col:"intensity"})
        tmp["condition"] = cond
        long_intensity.append(tmp)
    long_intensity = pd.concat(long_intensity, ignore_index=True)
    long_intensity = long_intensity[long_intensity["group"] != "Control"]

    # If predictions exist, also create probability bars with improved labels
    if pred_path.exists():
        preds = pd.read_csv(pred_path)  # sample_id, condition, p_weak, p_mild, p_strong, predicted_class
        groups_pred = preds["sample_id"].astype(str).apply(classify_loop_group).apply(pd.Series)
        preds = pd.concat([preds, groups_pred], axis=1)
        preds = preds[preds["group"] != "Control"]
        for cond in sorted(preds["condition"].dropna().unique()):
            sub = preds[preds["condition"] == cond].copy()
            prob_cols = [c for c in sub.columns if c.startswith("p_")]
            if not prob_cols:
                print(f"[looping] No probability columns for {cond}; skipping.")
                continue
            out_png = figs / f"bar_loopgroups_pred_probs_{cond}.png"
            out_csv = proc / f"loop_group_means_{cond}.csv"
            barplot_loop_groups(sub[["group","family"] + prob_cols], cond, out_png, out_csv)

    # Always produce intensity bars
    for cond in sorted(long_intensity["condition"].dropna().unique()):
        sub = long_intensity[long_intensity["condition"] == cond].copy()
        out_png = figs / f"bar_loopgroups_intensity_{cond}.png"
        out_csv = proc / f"loop_group_intensity_means_{cond}.csv"
        barplot_intensity_groups(sub[["group","family","intensity"]], cond, out_png, out_csv)

if __name__ == "__main__":
    main()
