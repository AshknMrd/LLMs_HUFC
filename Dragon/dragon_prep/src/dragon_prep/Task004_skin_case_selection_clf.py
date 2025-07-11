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
from pathlib import Path

import pandas as pd

from dragon_prep.utils import (apply_anon_annotations, num_patients,
                               prepare_for_anon, read_anon,
                               split_and_save_data)


def preprocess_reports(
    task_name: str,
    input_dir: Path,
    output_dir: Path,
):
    # read reports and labels (the PHI labels are in the "label" field, the reports are in the "text" field
    # and the task labels are in the meta->label field)
    # the labels are in the format [[start1, end1, label1], [start2, end2, label2], ...], for example [[75, 86, "<LABEL>"]]
    df = pd.read_json(input_dir / "all_checked_with_patient_ids_anon_annot_manual_fixed.jsonl", lines=True)

    # select both sets of labels
    df = df.rename(columns={"label": "label_phi"})
    df["label_task"] = df.meta.apply(lambda meta: meta["label"])

    # extract the metadata
    for key in df["meta"].iloc[0].keys():
        df[key] = df["meta"].apply(lambda x: x[key])
    df = df.drop(["meta", "label"], axis=1)

    # set the unique case identifier
    df["uid"] = df.apply(lambda row: f"{row['pid']}_{row['id']}", axis=1)
    print(f"Loaded {len(df)} cases ({num_patients(df)} patients)")

    # make label
    df["single_label_binary_classification_target"] = df["label_task"].apply(
        lambda labels: any(lbl == "excluded" for _, _, lbl in labels)
    )

    # perform anonymization
    df = df.apply(apply_anon_annotations, axis=1)
    assert df.text.apply(lambda text: "<PERSOON>" in text).sum() > 100, f"Unexpected number of <PERSOON> tags in reports: {df.text.apply(lambda text: '<PERSOON>' in text).sum()}"
    assert df.text.apply(lambda text: "<DATUM>" in text).sum() > 500, f"Unexpected number of <DATUM> tags in reports: {df.text.apply(lambda text: '<DATUM>' in text).sum()}"
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
        split_by="pid",
    )


if __name__ == "__main__":
    # create the parser
    parser = argparse.ArgumentParser(description="Script for preparing reports")
    parser.add_argument("--task_name", type=str, default="Task004_skin_case_selection_clf",
                        help="Name of the task")
    parser.add_argument("-i", "--input", type=Path, default=Path("/input"),
                        help="Path to the input data")
    parser.add_argument("-o", "--output", type=Path, default=Path("/output"),
                        help="Folder to store the prepared reports in")
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
    )
