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

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dragon_prep.utils import (apply_anon_annotations, num_patients,
                               prepare_for_anon, read_anon,
                               split_and_save_data)


def preprocess_reports(
    task_name: str,
    input_dir: Path,
    output_dir: Path,
):
    # read marksheets
    df = pd.read_excel(input_dir / "pancreas_overview_v2.xlsx", dtype=str)
    df["uid"] = df["base_studyuid"]
    df = df.rename(columns={"archiveID": "patient_id"})
    df_diagnosis_groups = pd.read_excel(input_dir / "pancreas_overview_v2.xlsx", sheet_name="Sheet2")
    df_reports = pd.read_json(input_dir / "all.jsonl", lines=True)
    print(f"Have {len(df)} cases ({num_patients(df)} patients) in the dataset")

    # strip whitespace
    for col in df.columns:
        df[col] = df[col].str.strip()

    # select cases with diagnosis annotation (missing diagnosis means the baseline study is not there yet)
    df = df.loc[df["diag_rad"].notna()]
    assert len(df) == 2035
    print(f"Have {len(df)} cases ({num_patients(df)} patients) with diagnosis annotation")

    # check labels
    diagnosis_group_mapping = {
        value.strip(): diagnosis_group
        for diagnosis_group in ["PDAC", "Other pancreatic disease", "Normal pancreas"]
        for value in df_diagnosis_groups[diagnosis_group].values
        if not (isinstance(value, float) and np.isnan(value))
    }
    assert set(df["diag_rad"].unique()) == set(diagnosis_group_mapping)

    # add reports to df
    df_reports["StudyInstanceUID"] = df_reports.meta.apply(lambda x: x["uid"])
    df_reports = df_reports.apply(apply_anon_annotations, axis=1)
    df_reports = df_reports.set_index("StudyInstanceUID")
    df["text"] = df.base_studyuid.map(df_reports.text.to_dict())

    # exclude cases without reports (some cases were annotated in a different workflow)
    assert df.text.isna().sum() == 41, f"Unexpected number of cases without reports: {df.text.isna().sum()}"
    df = df.loc[df.text.notna()]
    print(f"Have {len(df)} cases ({num_patients(df)} patients) with reports")

    # exclude cases with a non-diagnostic report (these were annotated in a different workflow)
    with open(input_dir / "excluded_cases_no_diagnostic_report.json") as f:
        subject_list_exclude = json.load(f)
    mask = df["uid"].isin(subject_list_exclude)
    print(f"Excluding {mask.sum()} cases ({num_patients(df[mask])} patients) without a diagnostic report")
    df = df[~mask]
    print(f"Have {len(df)} cases ({num_patients(df)} patients) left")

    # convert diagnosis to diagnosis group
    df["single_label_multi_class_classification_target"] = df["diag_rad"].map(diagnosis_group_mapping)

    # prepare for anonynimization
    assert df.text.apply(lambda text: "<DATUM>" in text).sum() > 800, f"Unexpected number of <DATUM> tags in RUMC reports: {df.text.apply(lambda text: '<DATUM>' in text).sum()}"
    prepare_for_anon(df=df, output_dir=output_dir, task_name=task_name, tag_phi=False, apply_hips=True)


def prepare_reports(
    task_name: str,
    output_dir: Path,
    test_split_size: float = 0.3,
):
    # read anonynimized data
    df = read_anon(output_dir / "anon" / task_name / "nlp-dataset.json")

    # make test and cross-validation splits
    split_and_save_data(
        df=df,
        output_dir=output_dir,
        task_name=task_name,
        test_split_size=test_split_size,
        split_by="patient_id",
    )


if __name__ == "__main__":
    # create the parser
    parser = argparse.ArgumentParser(description="Script for preparing reports")
    parser.add_argument("--task_name", type=str, default="Task009_pdac_diagnosis_clf",
                        help="Name of the task")
    parser.add_argument("-i", "--input", type=Path, default=Path("/input"),
                        help="Path to the input data")
    parser.add_argument("-o", "--output", type=Path, default=Path("/output"),
                        help="Folder to store the prepared reports in")
    parser.add_argument("--test_split_size", type=float, default=0.3,
                        help="Fraction of the dataset to use for testing")
    args = parser.parse_args()

    # run preprocessing
    preprocess_reports(
        task_name=args.task_name,
        input_dir=args.input,
        output_dir=args.output,
    )
    prepare_reports(
        task_name=args.task_name,
        output_dir=args.output,
        test_split_size=args.test_split_size,
    )
