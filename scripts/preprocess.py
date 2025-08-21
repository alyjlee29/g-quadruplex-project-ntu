
"""
preprocess.py
-------------
Merge NaCl/KCl replicates using explicit rules, attach sequences, and emit tidy tables
ready for feature building and ML. Also save grouped lists by plate row.

Inputs (expected):
- data_raw/20250814_test.xlsx
- data_raw/G4_with_flanking_sequence_list.xlsx

Outputs (written to data_processed/):
- averaged_by_well_NaCl_KCl.csv     : well-level NaCl_avg, KCl_avg, sample_id, sequence
- grouped_by_row_summary.csv        : mean per plate row (A–H) for NaCl/KCl
- grouped_samples_by_row.csv        : (row, col, well, sample_id, sequence)
- grouped_samples.json              : {"A": [{well, sample_id, sequence}, ...], ...}
"""

from pathlib import Path
import pandas as pd, numpy as np, json

WEIRD_NACL_ROW = "D"   # row D in NaCl_1 is abnormal and will be ignored

def read_plate_map(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="Data processing")
    df.columns = [str(c).strip() for c in df.columns]
    if "Sample" not in df.columns:
        for c in df.columns:
            if str(c).strip().lower() == "sample":
                df = df.rename(columns={c:"Sample"}); break
    num_cols = [c for c in df.columns if str(c).isdigit() and 1 <= int(str(c)) <= 12]
    plate = df[["Sample"] + num_cols].dropna(how="all")
    plate = plate[plate["Sample"].astype(str).str.match(r"^[A-H]$", na=False)]
    long = plate.melt(id_vars=["Sample"], var_name="col", value_name="sample_id")
    long["row"] = long["Sample"].astype(str)
    long["col"] = long["col"].astype(int)
    long["well"] = long["row"] + long["col"].astype(str)
    return long[["well","row","col","sample_id"]]

def detect_sheet(xls: pd.ExcelFile, keys):
    for k in keys:
        for s in xls.sheet_names:
            if k.lower() == s.lower():
                return s
    for k in keys:
        for s in xls.sheet_names:
            if k.lower() in s.lower():
                return s
    return None

def read_measure_sheet(xlsx_path: Path, sheet_name: str, value_name: str) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    header_row = None
    for i in range(min(200, len(raw))):
        if str(raw.iat[i,0]).strip().lower() == "rfu":
            header_row = i; break
    if header_row is None:
        raise ValueError(f"Could not find 'RFU' header in sheet {sheet_name}")
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=header_row)
    df = df.rename(columns={df.columns[0]:"row"})
    num_cols = [c for c in df.columns if str(c).isdigit()]
    df = df[["row"] + num_cols].copy()
    long = df.melt(id_vars=["row"], var_name="col", value_name=value_name)
    long["row"] = long["row"].astype(str).str.strip().str.upper()
    long["col"] = pd.to_numeric(long["col"], errors="coerce").astype("Int64")
    long = long.dropna(subset=["col"])
    long["well"] = long["row"] + long["col"].astype(str)
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    return long[["well","row","col",value_name]]

def main():
    project_root = Path(".")
    raw = project_root / "data_raw"
    proc = project_root / "data_processed"
    proc.mkdir(parents=True, exist_ok=True)

    plate_xlsx = raw / "20250814_test.xlsx"
    seq_xlsx = raw / "G4_with_flanking_sequence_list.xlsx"
    xls = pd.ExcelFile(plate_xlsx)

    # Plate map
    plate = read_plate_map(plate_xlsx)

    # Sheet detection (be tolerant to naming)
    nacl1 = detect_sheet(xls, ["replicate 1_ nacl","nacl 1","rep 1 nacl","replicate1_nacl","NaCl 1"])
    nacl2 = detect_sheet(xls, ["replicate 2_ nacl","nacl 2","rep 2 nacl","replicate2_nacl","NaCl 2"])
    kcl1  = detect_sheet(xls, ["replicate 1_ kcl","kcl 1","rep 1 kcl","replicate1_kcl","KCl 1"])
    kcl2  = detect_sheet(xls, ["replicate 2_ kcl","kcl 2","rep 2 kcl","replicate2_kcl","KCl 2"])

    # Read replicate sheets
    df_n1 = read_measure_sheet(plate_xlsx, nacl1, "rep_na1") if nacl1 else None
    df_n2 = read_measure_sheet(plate_xlsx, nacl2, "rep_na2") if nacl2 else None
    df_k1 = read_measure_sheet(plate_xlsx, kcl1,  "rep_k1") if kcl1 else None
    df_k2 = read_measure_sheet(plate_xlsx, kcl2,  "rep_k2") if kcl2 else None

    # NaCl merge (drop row D from replicate 1)
    base = plate[["well","row","col"]].copy()
    if df_n1 is not None:
        df_n1 = df_n1.loc[~(df_n1["row"].str.upper()==WEIRD_NACL_ROW)].copy()
        base = base.merge(df_n1, on=["well","row","col"], how="left")
    if df_n2 is not None:
        base = base.merge(df_n2, on=["well","row","col"], how="left")
    base["NaCl_avg"] = base[["rep_na1","rep_na2"]].mean(axis=1, skipna=True)

    # KCl merge (simple average)
    if df_k1 is not None:
        base = base.merge(df_k1, on=["well","row","col"], how="left")
    if df_k2 is not None:
        base = base.merge(df_k2, on=["well","row","col"], how="left")
    base["KCl_avg"] = base[["rep_k1","rep_k2"]].mean(axis=1, skipna=True)

    # Attach sample_id, sequence
    base = base.merge(plate[["well","sample_id"]], on="well", how="left")
    seq_df = pd.read_excel(seq_xlsx, sheet_name=0, header=None).iloc[:, :2].copy()
    seq_df.columns = ["sample_id","sequence"]
    seq_df["sample_id"] = seq_df["sample_id"].astype(str).str.strip()
    seq_df["sequence"] = seq_df["sequence"].astype(str).str.strip().str.upper()
    base = base.merge(seq_df, on="sample_id", how="left")
    base["is_control"] = base["sample_id"].astype(str).str.upper().isin(["CTRL0001","CTRL0002"]).astype(int)

    # Save tidy
    tidy_path = proc / "averaged_by_well_NaCl_KCl.csv"
    base[["row","col","well","sample_id","sequence","NaCl_avg","KCl_avg","is_control"]].to_csv(tidy_path, index=False)

    # Group summaries
    group_summary = base.groupby("row", as_index=False).agg(
        NaCl_mean=("NaCl_avg","mean"),
        KCl_mean=("KCl_avg","mean"),
        n_wells=("well","size")
    )
    group_summary.to_csv(proc / "grouped_by_row_summary.csv", index=False)

    # Grouped lists
    plate_seq = base[["row","col","well","sample_id","sequence"]].sort_values(["row","col"])
    plate_seq.to_csv(proc / "grouped_samples_by_row.csv", index=False)
    grouped = (plate_seq.groupby("row")
               .apply(lambda g: [{"well": f"{r.row}{r.col}", "sample_id": r.sample_id, "sequence": r.sequence}
                                 for r in g.itertuples(index=False)])
               .to_dict())
    with open(proc / "grouped_samples.json","w") as f:
        json.dump(grouped, f, indent=2)

    print("Wrote:", tidy_path, "and grouped summaries to data_processed/")

if __name__ == "__main__":
    main()
