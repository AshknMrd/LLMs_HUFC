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

from pathlib import Path

import pandas as pd

from dragon_prep.utils import apply_anon_annotations, num_patients


def prepare_avl_radiology_reports(input_dir: Path) -> pd.DataFrame:
    # read AVL marksheet
    df_annot_consecutive = pd.read_csv(input_dir / "annot_consecutive.csv", dtype=str)
    df_annot_random = pd.read_csv(input_dir / "annot_random.csv", dtype=str)
    print(f"Loaded {len(df_annot_consecutive)} consecutive radiology reports ({num_patients(df_annot_consecutive)} patients) for AvL")
    print(f"Loaded {len(df_annot_random)} random radiology reports ({num_patients(df_annot_random)} patients) for AvL")
    for df in [df_annot_consecutive, df_annot_random]:
        df["uid"] = df.apply(lambda row: row["patient_id"] + "_" + row["date"], axis=1)
        df["study_id"] = df["date"]

    # merge dataframes
    df = pd.concat((df_annot_consecutive, df_annot_random), ignore_index=True)

    # drop duplicates (due to both consecutive and random radiology reports being present in the dataset, which overlap for some cases)
    df = df.drop_duplicates(subset=["uid"])
    df = df.drop_duplicates(subset=["text"])
    print(f"Have {len(df)} radiology reports ({num_patients(df)} patients) after excluding duplicates for AvL")

    # exclude cases that are not a prostate MRI detection study
    mask = (df["joeran_checken"] == "Exclude no detection study")
    print(f"Excluding {mask.sum()} cases ({num_patients(df[mask])}) that are not a prostate MRI detection study")
    df = df[~mask]
    print(f"Have {len(df)} radiology reports ({num_patients(df)} patients) after excluding non-detection studies for AvL")

    # read reports with manual anonymization annotations
    df_reports_consecutive = pd.read_json(input_dir / "data_radiology_consecutive_mapped.jsonl", lines=True)
    df_reports_random = pd.read_json(input_dir / "data_radiology_random_mapped.jsonl", lines=True)
    df_reports = pd.concat((df_reports_consecutive, df_reports_random), ignore_index=True)
    df_reports = df_reports.apply(apply_anon_annotations, axis=1)
    df_reports = df_reports.set_index("uid")

    # merge with radiology reports
    df["text"] = df["uid"].map(df_reports["text"].to_dict())

    # cleanup pirads lesions notation
    df["pirads_lesions"] = df["pirads_lesions"].str.replace(".", ",")

    return df


def prepare_avl_pathology_reports(
    input_dir: Path,
) -> pd.DataFrame:
    # read marksheet
    df = pd.read_json(input_dir / "data_pathology_mapped.jsonl", lines=True)
    df["patient_id"] = df["meta"].apply(lambda meta: meta["PATIENTNR"])
    df["study_id"] = df["meta"].apply(lambda meta: meta["uid"])
    df["uid"] = df["meta"].apply(lambda meta: f"{meta['PATIENTNR']}_{meta['uid']}")
    print(f"Have {len(df)} pathology cases ({num_patients(df)} patients) from AVL")

    # apply manual anonymization
    df = df.apply(apply_anon_annotations, axis=1)

    return df
