
"""
feature_build.py
----------------
Dissect sequences into G4 sections and encode nucleotides A,T,C,G -> 1,2,3,4.
Pad each section to the **maximum length seen** for that section so every sample
has a fixed-length vector (as in the slide). Also adds the cation condition as
a categorical feature (NaCl or KCl). Creates **3-class labels** using thresholds:
weak ≤ 80, mild (80–135), strong ≥ 135.

Input:
- data_processed/averaged_by_well_NaCl_KCl.csv (from preprocess.py)

Output:
- data_processed/model_matrix_3class.csv (features + label_3class + sample_id + condition)
"""
from pathlib import Path
import pandas as pd, numpy as np, re

WEAK_MAX = 80.0
STRONG_MIN = 135.0

# Map bases to integers
BASE_MAP = {"A":1, "T":2, "C":3, "G":4}

def label_3class(x: float) -> str:
    if pd.isna(x): return np.nan
    if x <= WEAK_MAX: return "weak"
    if x >= STRONG_MIN: return "strong"
    return "mild"

def find_gtracts(s: str):
    """Return list of (start,end) for the first four G-runs (>=3 Gs)"""
    s = s.upper().replace("U","T")
    idx = []
    i = 0
    while i <= len(s)-3:
        if s[i:i+3] == "GGG":
            j = i+3
            while j < len(s) and s[j]=="G":
                j += 1
            idx.append((i,j))
            i = j
        else:
            i += 1
    return idx[:4]

def dissect(seq: str):
    """Return sections: five, G1, L1, G2, L2, G3, L3, G4, three"""
    seq = (seq or "").upper().replace("U","T")
    gidx = find_gtracts(seq)
    if len(gidx) < 4:
        return ["", "", "", "", "", "", "", "", ""]  # pad empty sections
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
        arr = arr + [0]*(maxlen-len(arr))
    else:
        arr = arr[:maxlen]
    return arr

def main():
    project_root = Path(".")
    proc = project_root / "data_processed"
    src = proc / "averaged_by_well_NaCl_KCl.csv"
    out_csv = proc / "model_matrix_3class.csv"

    df = pd.read_csv(src)
    # Build long-format by condition
    long = []
    for cond, col in [("NaCl","NaCl_avg"), ("KCl","KCl_avg")]:
        tmp = df[["sample_id","sequence",col]].copy()
        tmp["condition"] = cond
        tmp = tmp.rename(columns={col:"intensity"})
        long.append(tmp)
    mm = pd.concat(long, ignore_index=True)
    mm["label_3class"] = mm["intensity"].apply(label_3class)

    # Determine per-section max lengths
    sections = ["five","G1","L1","G2","L2","G3","L3","G4","three"]
    lengths = {sec:0 for sec in sections}
    split = {}
    for r in mm.itertuples(index=False):
        parts = dissect(r.sequence)
        split[id(r)] = parts
        for sec, s in zip(sections, parts):
            lengths[sec] = max(lengths[sec], len(s))

    # Encode
    feat_rows = []
    for r in mm.itertuples(index=False):
        parts = split[id(r)]
        row = {
            "sample_id": r.sample_id,
            "condition": r.condition,
            "intensity": r.intensity,
            "label_3class": r.label_3class
        }
        for sec, s in zip(sections, parts):
            enc = encode_section(s, lengths[sec])
            for i, val in enumerate(enc, start=1):
                row[f"{sec}_{i}"] = val
        feat_rows.append(row)

    out = pd.DataFrame(feat_rows)
    out.to_csv(out_csv, index=False)
    print("Wrote", out_csv, "with shape", out.shape)
    print("Per-section max lengths:", lengths)

if __name__ == "__main__":
    main()
