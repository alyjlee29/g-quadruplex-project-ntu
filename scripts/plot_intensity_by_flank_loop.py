"""
plot_intensity_by_flank_loop.py
--------------------------------
Reproduce intensity bar charts grouped by flanking length (0,3,6,9-mer) and loop length code
(111,222,333,444,555), split by loop nucleotide family (T-loop vs A-loop), for NaCl and KCl.

Inputs:
  data_processed/averaged_by_well_NaCl_KCl.csv   # from preprocess.py (controls flagged)

Outputs:
  figures/intensity_by_flank_loop_NaCl.png
  figures/intensity_by_flank_loop_KCl.png
  data_processed/intensity_by_flank_loop_summary.csv

Notes:
  • A-loop/T-loop classification requires each of L1,L2,L3 to be composed entirely of A or T.
  • Flanking length must be symmetric and exactly in {0,3,6,9}. Others are excluded (count reported).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- G4 dissection helpers -----------------

def find_gtracts(seq: str):
    """Return the first four contiguous G-runs (>=3 Gs) as (start,end) 0-based, end exclusive."""
    s = seq.upper().replace("U","T")
    idx = []
    i = 0
    while i <= len(s)-3:
        if s[i:i+3] == "GGG":
            j = i+3
            while j < len(s) and s[j] == "G":
                j += 1
            idx.append((i,j))
            i = j
        else:
            i += 1
    return idx[:4]

def dissect(seq: str):
    """Return 5' flank, G1, L1, G2, L2, G3, L3, G4, 3' flank. Empty strings if not found."""
    seq = str(seq or "").upper().replace("U", "T")
    g = find_gtracts(seq)
    if len(g) < 4:
        return "", "", "", "", "", "", "", "", ""
    five = seq[:g[0][0]]
    G1s,G1e = g[0]; G2s,G2e = g[1]; G3s,G3e = g[2]; G4s,G4e = g[3]
    G1 = seq[G1s:G1e]; G2 = seq[G2s:G2e]; G3 = seq[G3s:G3e]; G4 = seq[G4s:G4e]
    L1 = seq[G1e:G2s]; L2 = seq[G2e:G3s]; L3 = seq[G3e:G4s]
    three = seq[G4e:]
    return five, G1, L1, G2, L2, G3, L3, G4, three

def loop_code(L1, L2, L3):
    return f"{len(L1)}{len(L2)}{len(L3)}"

def loop_family(L1, L2, L3, frac_thresh=0.8):
    """
    'T loop' if each loop is >= frac_thresh T;
    'A loop' if each loop is >= frac_thresh A;
    else None (mixed).
    """
    loops = [L1, L2, L3]
    def frac_of(s, base):
        return (s.count(base) / len(s)) if s else 0.0
    if all(len(x) > 0 and frac_of(x, "T") >= frac_thresh for x in loops):
        return "T loop"
    if all(len(x) > 0 and frac_of(x, "A") >= frac_thresh for x in loops):
        return "A loop"
    return None
def bin_flanking_lengths(five: str, three: str):
    """
    Return a robust flanking bin in {0,3,6,9}.
    - Allow small left/right asymmetry (<= 2 nt).
    - Bin by the *average* flank length to nearest 0/3/6/9 (clip to [0,9]).
    - Return None if flanks are wildly asymmetric.
    """
    n5 = len(five or "")
    n3 = len(three or "")
    # reject only if very asymmetric
    if abs(n5 - n3) > 2:
        return None
    avg = (n5 + n3) / 2.0
    # bin to nearest of 0, 3, 6, 9
    candidates = np.array([0, 3, 6, 9], dtype=float)
    fl = int(candidates[np.argmin(np.abs(candidates - avg))])
    # guard: if avg is far from any bin (>1.5 nt), treat as unknown
    if abs(avg - fl) > 1.5:
        return None
    return fl

# ----------------- Main aggregation & plotting -----------------

def aggregate(df):
    """
    df: tidy table with columns [sample_id, sequence, NaCl_avg, KCl_avg, is_control]
    Returns long-format aggregated mean & std by:
      condition in {NaCl,KCl} × family in {'T loop','A loop'} × flank_len in {0,3,6,9} × code in {111..555}
    """
    rows = []
    excluded = {"not_four_tracts":0, "non_AT_loops":0, "bad_flank_len":0}
    for r in df.itertuples(index=False):
        five, G1, L1, G2, L2, G3, L3, G4, three = dissect(r.sequence)
        if not L1:  # no proper G4 segmentation
            excluded["not_four_tracts"] += 1
            continue
        fam = loop_family(L1, L2, L3)
        if fam is None:
            excluded["non_AT_loops"] += 1
            continue
        flank = bin_flanking_lengths(five, three)
        if flank is None:
            excluded["bad_flank_len"] += 1
            continue
        code = loop_code(L1, L2, L3)  # '111', '222', ...
        if code not in {"111","222","333","444","555"}:
            # keep only the five canonical loop codes for this figure
            continue

        for cond, val in [("NaCl", getattr(r, "NaCl_avg", np.nan)),
                          ("KCl", getattr(r, "KCl_avg", np.nan))]:
            if pd.isna(val): 
                continue
            rows.append({"condition": cond,
                         "family": fam,
                         "flank_len": flank,
                         "code": code,
                         "intensity": float(val)})

    agg = (pd.DataFrame(rows)
             .groupby(["condition","family","flank_len","code"], as_index=False)
             .agg(mean_intensity=("intensity","mean"),
                  sd_intensity=("intensity","std"),
                  n=("intensity","size")))
    return agg, excluded

# Replace plot_panels() with this version:

def plot_panels(agg, cond, out_png):
    fam_order = ["T loop","A loop"]
    flank_order = [0,3,6,9]
    code_order = ["111","222","333","444","555"]

    subset = agg[agg["condition"]==cond].copy()
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 7), sharey=False)
    for i, fam in enumerate(fam_order):
        for j, fl in enumerate(flank_order):
            ax = axes[i, j]
            sub = subset[(subset["family"]==fam) & (subset["flank_len"]==fl)]
            ax.set_title(f"{fl}-mer flanking", fontsize=11)
            ax.set_xlabel("Loop length code")
            if j == 0:
                ax.set_ylabel(f"{fam}\nIntensity (a.u.)")
            else:
                ax.set_ylabel("Intensity (a.u.)")
            ax.grid(axis="y", alpha=0.25, linestyle="--")
            if sub.empty:
                ax.text(0.5, 0.5, "no samples", ha="center", va="center", alpha=0.6)
                ax.set_xticks(range(len(code_order))); ax.set_xticklabels(code_order)
                continue
            frame = pd.DataFrame({"code": code_order}).merge(
                sub[["code","mean_intensity","sd_intensity","n"]],
                on="code", how="left"
            )
            # plot only codes that exist
            frame = frame.dropna(subset=["mean_intensity"])
            x = np.arange(len(frame))
            ax.bar(x, frame["mean_intensity"], yerr=frame["sd_intensity"].fillna(0.0), capsize=3)
            ax.set_xticks(x); ax.set_xticklabels(frame["code"])
    fig.suptitle(f"Intensity by flanking length and loop code — {cond}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    root = Path(".")
    proc = root / "data_processed"
    figs = root / "figures"

    src = proc / "averaged_by_well_NaCl_KCl.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Run scripts/preprocess.py first.")
    df = pd.read_csv(src)

    # Basic hygiene
    df = df[~df["sample_id"].astype(str).str.upper().isin(["CTRL0001","CTRL0002"])].copy()
    for col in ("NaCl_avg","KCl_avg"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    agg, excluded = aggregate(df)
     
    print("[diag] Excluded:", excluded)
    if not agg.empty:
        print("[diag] Cells per condition/family/flank:")
        print(agg.groupby(["condition","family","flank_len"])["code"].nunique())
    # Save the aggregated numbers for slide tables
    out_csv = proc / "intensity_by_flank_loop_summary.csv"
    agg.sort_values(["condition","family","flank_len","code"]).to_csv(out_csv, index=False)
    print(f"[plot] Wrote summary: {out_csv}")
    print("[plot] Excluded counts:", excluded)

    # Draw one figure per condition
    if not agg.empty:
        for cond in sorted(agg["condition"].unique()):
            out_png = figs / f"intensity_by_flank_loop_{cond}.png"
            plot_panels(agg, cond, out_png)
            print(f"[plot] Wrote: {out_png}")
    else:
        print("[plot] No data matched the (flank_len, loop code) criteria.")

if __name__ == "__main__":
    main()
