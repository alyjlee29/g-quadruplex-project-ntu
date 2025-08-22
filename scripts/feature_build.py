"""
feature_build.py  — robust version
Dissects sequences into G4 sections and encodes A/T/C/G→1/2/3/4 with per-section padding.
Builds a long table (NaCl + KCl) with 3-class labels.

Input:
  data_processed/averaged_by_well_NaCl_KCl.csv

Output:
  data_processed/model_matrix_3class.csv
"""

from pathlib import Path
import pandas as pd, numpy as np, re, sys

WEAK_MAX   = 80.0
STRONG_MIN = 135.0
BASE_MAP   = {"A":1, "T":2, "C":3, "G":4}

def label_3class(x: float) -> str:
    if pd.isna(x): return np.nan
    if x <= WEAK_MAX: return "weak"
    if x >= STRONG_MIN: return "strong"
    return "mild"

def sanitize_seq(x) -> str:
    """Return a safe ATCG string (uppercased; U→T; non-ATCG removed). Empty if missing."""
    if isinstance(x, str):
        s = x.strip().upper().replace("U","T")
    else:
        s = ""
    # keep only A/T/C/G
    s = re.sub(r"[^ATCG]", "", s)
    return s

def find_gtracts(s: str):
    """Indices of first four G-runs (≥3 Gs)."""
    idx = []
    i = 0
    n = len(s)
    while i <= n - 3:
        if s[i:i+3] == "GGG":
            j = i + 3
            while j < n and s[j] == "G":
                j += 1
            idx.append((i, j))
            i = j
        else:
            i += 1
    return idx[:4]

def dissect(seq: str):
    """Return [five, G1, L1, G2, L2, G3, L3, G4, three]. If <4 G-tracts, returns empty sections."""
    gidx = find_gtracts(seq)
    if len(gidx) < 4:
        return ["","","","","","","","",""]
    five = seq[:gidx[0][0]]
    G1s, G1e = gidx[0]
    G2s, G2e = gidx[1]
    G3s, G3e = gidx[2]
    G4s, G4e = gidx[3]
    G1 = seq[G1s:G1e]; G2 = seq[G2s:G2e]; G3 = seq[G3s:G3e]; G4 = seq[G4s:G4e]
    L1 = seq[G1e:G2s]; L2 = seq[G2e:G3s]; L3 = seq[G3e:G4s]
    three = seq[G4e:]
    return [five, G1, L1, G2, L2, G3, L3, G4, three]

def encode_section(s: str, maxlen: int):
    arr = [BASE_MAP.get(ch, 0) for ch in s]
    if len(arr) < maxlen:
        arr = arr + [0] * (maxlen - len(arr))
    else:
        arr = arr[:maxlen]
    return arr

def main():
    project_root = Path(".")
    proc = project_root / "data_processed"
    src  = proc / "averaged_by_well_NaCl_KCl.csv"
    out_csv = proc / "model_matrix_3class.csv"

    if not src.exists():
        sys.exit(f"ERROR: {src} not found. Run scripts/preprocess.py first.")

    df = pd.read_csv(src)

    # ---- sanitize/validate sequences ----
    if "sequence" not in df.columns:
        sys.exit("ERROR: 'sequence' column not found in averaged_by_well_NaCl_KCl.csv")

    df["sequence_raw"] = df["sequence"]
    df["sequence"] = df["sequence"].apply(sanitize_seq)

    n_total = len(df)
    n_missing = (df["sequence"] == "").sum()
    if n_missing:
        # Show which sample_ids are missing sequence (often controls or name mismatches)
        missing_ids = df.loc[df["sequence"] == "", "sample_id"].dropna().astype(str).unique()[:20]
        print(f"[feature_build] WARNING: {n_missing}/{n_total} rows lack a valid ATCG sequence.")
        print("Examples (first 20):", list(missing_ids))

    # Drop rows without a valid sequence (controls or unmapped names)
    df = df[df["sequence"] != ""].copy()
    if df.empty:
        sys.exit("ERROR: No rows left after sequence sanitization. Check your sequence list join.")

    # ---- build long table by condition with numeric intensity ----
    for col in ("NaCl_avg","KCl_avg"):
        if col not in df.columns:
            sys.exit(f"ERROR: '{col}' missing in averaged_by_well_NaCl_KCl.csv")

    df["NaCl_avg"] = pd.to_numeric(df["NaCl_avg"], errors="coerce")
    df["KCl_avg"]  = pd.to_numeric(df["KCl_avg"],  errors="coerce")

    long = []
    for cond, col in [("NaCl","NaCl_avg"), ("KCl","KCl_avg")]:
        tmp = df[["sample_id","sequence",col]].copy()
        tmp["condition"] = cond
        tmp = tmp.rename(columns={col:"intensity"})
        long.append(tmp)
    mm = pd.concat(long, ignore_index=True)

    # labels
    mm["label_3class"] = mm["intensity"].apply(label_3class)

    # ---- per-section max lengths (from cleaned sequences) ----
    sections = ["five","G1","L1","G2","L2","G3","L3","G4","three"]
    maxlen = {sec:0 for sec in sections}
    parts_cache = []

    for seq in mm["sequence"]:
        parts = dissect(seq)
        parts_cache.append(parts)
        for sec, s in zip(sections, parts):
            if len(s) > maxlen[sec]:
                maxlen[sec] = len(s)

    # ---- encode to fixed length ----
    rows = []
    for (idx, r) in enumerate(mm.itertuples(index=False)):
        parts = parts_cache[idx]
        row = {
            "sample_id": r.sample_id,
            "condition": r.condition,
            "intensity": r.intensity,
            "label_3class": r.label_3class
        }
        for sec, s in zip(sections, parts):
            enc = encode_section(s, maxlen[sec])
            for i, v in enumerate(enc, start=1):
                row[f"{sec}_{i}"] = v
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)

    print(f"[feature_build] Wrote {out_csv} with shape {out.shape}")
    print("[feature_build] Per-section max lengths:", maxlen)
    if n_missing:
        print("[feature_build] NOTE: Rows with empty sequences were dropped to avoid encoding errors.")

if __name__ == "__main__":
    main()
