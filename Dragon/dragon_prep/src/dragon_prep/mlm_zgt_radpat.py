from pathlib import Path

import pandas as pd

# paths
data_dir = Path("/mnt/data/radpat")
report_paths = [
    data_dir / "radiology_part_1.csv",
    data_dir / "radiology_part_2.csv",
    data_dir / "pathology.csv",
]
save_path = Path("/mnt/data/joeran/mlm_zgt_radpat.json")

# load data
report_parts = []
report_column_names = ("PLATTETEXT", "TRANSTEXT", "VERSLAG")
for report_path in report_paths:
    df = pd.read_csv(report_path, sep=";", encoding="utf-8")
    for column in report_column_names:
        if column in df.columns:
            df = df.rename(columns={column: "text"})
            break
    for column in ["ONDERZNR", "RONTVERRID"]:
        if column in df.columns:
            df = df.rename(columns={column: "uid"})
            break
    print(f"Loaded {len(df)} reports (parts) from {report_path}, {len(df.uid.unique())} studies.")
    report_parts.append(df[["uid", "text"]])

# combine reports
print(f"Combining {len(report_parts)} report sources")
reports = pd.concat(report_parts, axis=0, ignore_index=True)
print(f"Have {len(reports)} report (parts) from {len(reports.uid.unique())} studies.")

# show info
print(reports.info())

# remove duplicates
study_ids_all = reports.uid.unique()
reports_duplicated = reports.duplicated(subset=["text"])
print(f"Removing {reports_duplicated.sum()} duplicate report (parts) from {len(reports[reports_duplicated].uid.unique())} studies.")
reports = reports.drop_duplicates(subset=["text"])
study_ids_filtered = reports.uid.unique()
print(f"Have {len(reports)} report (parts) from {len(study_ids_filtered)} studies after filtering.")
print(f"{len(set(study_ids_all) - set(study_ids_filtered))} studies were removed altogether.")

# show info
print(reports.info())

# train/validation/test splits
print("Splitting into train/validation/test")
reports = reports.sample(frac=1, random_state=42)
reports_train = reports.iloc[: int(0.8 * len(reports))]
reports_val = reports.iloc[int(0.8 * len(reports)):int(0.9 * len(reports))]
reports_test = reports.iloc[int(0.9 * len(reports)):]

# save
reports.to_json(save_path, orient="records", lines=True)
reports_train.to_json(save_path.with_name(save_path.stem + "_train.json"), orient="records", lines=True)
reports_val.to_json(save_path.with_name(save_path.stem + "_val.json"), orient="records", lines=True)
reports_test.to_json(save_path.with_name(save_path.stem + "_test.json"), orient="records", lines=True)
