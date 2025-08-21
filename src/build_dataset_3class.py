import argparse, sys, pandas as pd, numpy as np, re
from pathlib import Path
from collections import Counter

# --------- CONFIG ---------
WEAK_MAX = 80.0
STRONG_MIN = 135.0
CONTROL_NAMES = {"CTRL0001","CTRL0002"}  # case-insensitive

# --------- HELPERS ---------
def read_sequence_list(seq_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_excel(seq_path, sheet_name=0, header=None).iloc[:, :2].copy()
        df.columns = ["sample_id","sequence"]
    except Exception:
        df = pd.read_excel(seq_path, sheet_name=0).iloc[:, :2].copy()
        df.columns = ["sample_id","sequence"]
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    df = df[df["sequence"].str.fullmatch(r"[ACGT]+", na=False)]
    return df.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)

def parse_g4_sections(seq: str):
    s = (seq or "").upper().replace("U","T").strip()
    gidx = []
    i = 0
    while i <= len(s) - 3:
        if s[i:i+3] == "GGG":
            j = i + 3
            while j < len(s) and s[j] == "G":
                j += 1
            gidx.append((i, j))
            i = j
        else:
            i += 1
    if len(gidx) < 4:
        return "", [], ["","",""], ""
    gidx = gidx[:4]
    five = s[:gidx[0][0]]
    three = s[gidx[-1][1]:]
    loops = [s[gidx[k][1]:gidx[k+1][0]] for k in range(3)]
    gtracts = [s[a:b] for a,b in gidx]
    return five, gtracts, loops, three

def loop_feats(loop: str, prefix: str):
    c = Counter(loop); n = len(loop)
    feats = {
        f"{prefix}_len": n,
        f"{prefix}_A_frac": (c["A"]/n) if n else 0.0,
        f"{prefix}_C_frac": (c["C"]/n) if n else 0.0,
        f"{prefix}_G_frac": (c["G"]/n) if n else 0.0,
        f"{prefix}_T_frac": (c["T"]/n) if n else 0.0,
    }
    dinucs = [a+b for a in "ACGT" for b in "ACGT"]
    denom = max(1, n-1)
    km2 = Counter(loop[i:i+2] for i in range(max(0, n-1)))
    for d in dinucs:
        feats[f"{prefix}_2mer_{d}_frac"] = km2.get(d, 0) / denom
    return feats

def flank_feats(five: str, three: str):
    def gc(s): return (s.count("G")+s.count("C"))/len(s) if s else 0.0
    return {
        "five_len": len(five),
        "three_len": len(three),
        "five_gc": gc(five),
        "three_gc": gc(three),
        "flank_variant": ("CGACGAGTA" if five.startswith("CGACGAGTA")
                          else ("CGACGAGGA" if five.startswith("CGACGAGGA") else "other"))
    }

def gtract_feats(gtracts):
    if not gtracts:
        return {"gtract_min":0,"gtract_max":0,"gtract_mean":0.0,"gtract_var":0.0}
    lens = [len(g) for g in gtracts]
    return {
        "gtract_min": int(min(lens)),
        "gtract_max": int(max(lens)),
        "gtract_mean": float(np.mean(lens)),
        "gtract_var": float(np.var(lens)),
    }

def featurize_sequence(sample_id: str, seq: str) -> dict:
    five, gtracts, loops, three = parse_g4_sections(seq)
    feats = {"sample_id": sample_id, "sequence": seq}
    for i, loop in enumerate(loops, start=1):
        feats.update(loop_feats(loop, f"L{i}"))
    Ls = [len(loops[0] or ""), len(loops[1] or ""), len(loops[2] or "")]
    feats["loop_code"] = f"{Ls[0]}{Ls[1]}{Ls[2]}"
    feats["L_sum"] = sum(Ls)
    feats["L_var"] = float(np.var(Ls)) if Ls else 0.0
    feats["L_symmetry_L1eqL3"] = int(Ls[0] == Ls[2]) if Ls else 0
    feats["L_max"] = max(Ls) if Ls else 0
    feats["L_min"] = min(Ls) if Ls else 0
    feats.update(flank_feats(five, three))
    feats.update(gtract_feats(gtracts))
    return feats

def label_3class(intensity: float) -> str:
    if pd.isna(intensity):
        return np.nan
    if intensity <= WEAK_MAX:
        return "weak"
    elif intensity >= STRONG_MIN:
        return "strong"
    else:
        return "mild"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_matrix", required=True, help="Path to NaCl_model_matrix.csv")
    p.add_argument("--seq_list", required=True, help="Path to 'G4 with flanking sequence list.xlsx'")
    p.add_argument("--out_csv", required=True, help="Output CSV path for 3-class features")
    args = p.parse_args()

    model_df = pd.read_csv(args.model_matrix)
    model_df.columns = [c.strip().lower().replace(" ", "_") for c in model_df.columns]

    # sample_id column
    sample_col = None
    for key in ("sample","sample_id","name","id"):
        if key in model_df.columns:
            sample_col = key; break
    if sample_col is None:
        sys.exit("Could not find a 'sample' or 'sample_id' column in model matrix.")
    model_df["sample_id"] = model_df[sample_col].astype(str).str.strip()

    # intensity column
    intensity_col = None
    for cand in ("mean_intensity","intensity_mean","intensity","rfu_mean","rfu"):
        if cand in model_df.columns:
            intensity_col = cand; break
    if intensity_col is None:
        num_cols = model_df.select_dtypes(include=[np.number]).columns.tolist()
        cand = [c for c in num_cols if "intensity" in c or "rfu" in c]
        if cand: intensity_col = cand[0]
    if intensity_col is None:
        sys.exit("Could not locate an intensity column in model matrix.")

    # sequences
    if "sequence" not in model_df.columns:
        seq_df = read_sequence_list(Path(args.seq_list))
        model_df = model_df.merge(seq_df, on="sample_id", how="left", validate="many_to_one")
    else:
        model_df["sequence"] = model_df["sequence"].astype(str).str.strip().str.upper()

    # exclude controls
    sid_up = model_df["sample_id"].str.upper()
    model_df = model_df[~sid_up.isin(CONTROL_NAMES)].copy()

    # label
    model_df["label_3class"] = model_df[intensity_col].apply(label_3class)

    # optional QC flag if present
    if "sd_intensity" in model_df.columns and "n_wells" in model_df.columns:
        cv = model_df["sd_intensity"] / model_df[intensity_col].replace(0, np.nan)
        model_df["qc_fail"] = ((model_df["n_wells"]>=2) & (cv>0.30) & ~((model_df[intensity_col]<1.0) & (model_df["sd_intensity"]<=1.0))).astype(int)
    else:
        model_df["qc_fail"] = 0

    # features
    seq_unique = model_df[["sample_id","sequence"]].drop_duplicates()
    feats = pd.DataFrame([featurize_sequence(r.sample_id, r.sequence) for r in seq_unique.itertuples(index=False)])
    out = model_df.merge(feats, on=["sample_id","sequence"], how="left", validate="many_to_one")

    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} with shape {out.shape}")

if __name__ == "__main__":
    main()
