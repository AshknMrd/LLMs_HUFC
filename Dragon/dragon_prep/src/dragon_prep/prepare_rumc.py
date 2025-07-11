#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
from pathlib import Path

import pandas as pd

from dragon_prep.utils import (apply_anon_annotations, num_patients,
                               read_marksheet)


def prepare_rumc_prostate_radiology_reports(input_dir: Path) -> pd.DataFrame:
    # read PI-CAI marksheet
    df = read_marksheet(input_dir / "PICAI-PubPrivTrain-patient-level-marksheet_v2.xlsx")
    df["uid"] = df.apply(lambda row: f"{row['patient_id']}_{row['study_id']}", axis=1)

    # select cases from RUMC
    df = df[df["center"] == "RUMC"]

    # drop rows without a radiology report
    df = df.rename(columns={"radiology_report": "text"})
    df = df.dropna(subset=["text"])
    print(f"Have {len(df)} cases ({num_patients(df)} patients) for RUMC")

    # exclude cases with a non-Dutch report
    with open(input_dir / "radiology/excluded_cases_not_dutch.json") as f:
        excluded_cases = json.load(f)
    mask = df["uid"].isin(excluded_cases)
    df = df[~mask]
    print(f"Excluded {mask.sum()} cases with a non-Dutch report")
    print(f"Have {len(df)} cases ({num_patients(df)} patients) for RUMC with a Dutch report")

    df_reports = pd.read_json(input_dir / "radiology/all.jsonl", lines=True)
    df_reports = df_reports.apply(apply_anon_annotations, axis=1)
    df_reports["uid"] = df_reports.meta.apply(lambda meta: meta["uid"])

    # merge with radiology reports
    for i, row in df.iterrows():
        df.loc[i, "text"] = df_reports[df_reports["uid"] == row["uid"]]["text"].values[0]

    return df


def prepare_rumc_lung_pathology_reports(input_dir: Path) -> pd.DataFrame:
    # read marksheet
    df = pd.read_excel(input_dir / "ignite_reports_with_ct_number.xlsx")
    df["uid"] = df["C/T-number"]
    df = df.rename(columns={"Report": "text", "Anonymous ID": "patient_id"})
    print(f"Have {len(df)} cases from {num_patients(df)} patients")

    # read reports with PHI annotations
    df_reports = pd.read_json(input_dir / "all.jsonl", lines=True)

    # remove false positives from the PHI annotations
    for i, row in df_reports.iterrows():
        labels = row["label"]
        for i, (start, end, lbl) in enumerate(labels):
            selected_text = row["text"][start:end]
            if selected_text in [
                "ALK, BRAF", "KIT, KRAS", "KRAS, BRAF", "ALK, ARAF", "ARAF, BRAF", "ALK, AR", "ALK",
                "KRAS, NRAS", "NRAS, BRAF", "TRK, BRAF", "ALK FISH", "KDR, KIT", "AR, ALK",
                "Fish-FISH ALK", "KRAS", "KRAS, MPL", "ALK, ROS", "FISH ALK", "BRAF, KIT"
            ]:
                labels[i] = (start, end, "REMOVE")
        labels = [label for label in labels if label[2] != "REMOVE"]
        df_reports.at[i, "label"] = labels

    # apply the annotations
    df_reports = df_reports.apply(apply_anon_annotations, axis=1)

    # merge with radiology reports
    for i, row in df.iterrows():
        df.loc[i, "text"] = df_reports[df_reports["uid"] == row["uid"]]["text"].values[0]

    return df
