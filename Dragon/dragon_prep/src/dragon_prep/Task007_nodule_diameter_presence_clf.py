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

import pandas as pd

from dragon_prep.utils import prepare_for_anon, read_anon, split_and_save_data


def read_dataset(
    marksheet_path: Path | str,
    reports_path: Path | str,
):
    """
    Read dataset from marksheet and reports.
    """
    # read marksheet (Excel sheet), with PatientID	AccessionNumber	StudyInstanceUID	StudyDescription	StudyDate	hospital	contains nodules	max_diameter	comment
    df = pd.read_excel(marksheet_path, dtype=str)

    # read reports (JSONlines), with PatientID	AccessionNumber	StudyInstanceUID	StudyDescription	StudyDate	text
    with open(reports_path, "r") as f:
        reports = [json.loads(line) for line in f]

    # add reports to marksheet
    assert len(df) == len(reports)
    assert len(df) == len(df["StudyInstanceUID"].unique())
    df["text"] = ""
    for report in reports:
        accession_number = report["StudyInstanceUID"]
        text = report["text"]
        df.loc[df["StudyInstanceUID"] == accession_number, "text"] = text

    if any(df["text"] == ""):
        print(f"WARNING: {sum(df['text'] == '')} reports were not found in {reports_path}")

    return df


def preprocess_reports(
    task_name: str,
    input_dir: Path,
    output_dir: Path,
):
    # paths
    development_annotations_path = input_dir / "annotations_development.xlsx"
    test_annotations_path = input_dir / "annotations_test.xlsx"
    sampled_development_reports_path = input_dir / "sampled_reports_development.json"
    sampled_test_reports_path = input_dir / "sampled_reports_test.json"

    # read marksheet and reports
    df_dev = read_dataset(
        marksheet_path=development_annotations_path,
        reports_path=sampled_development_reports_path,
    )
    df_test = read_dataset(
        marksheet_path=test_annotations_path,
        reports_path=sampled_test_reports_path,
    )
    df_dev["uid"] = df_dev["StudyInstanceUID"]
    df_test["uid"] = df_test["StudyInstanceUID"]

    # select cases with nodules
    df_dev = df_dev[df_dev["contains nodules"] == "True"]
    df_test = df_test[df_test["contains nodules"] == "True"]

    # set labels
    df_dev["single_label_binary_classification_target"] = (df_dev["max_diameter"].astype(float) > 0)
    df_test["single_label_binary_classification_target"] = (df_test["max_diameter"].astype(float) > 0)

    assert df_dev["single_label_binary_classification_target"].sum() == 186
    assert df_test["single_label_binary_classification_target"].sum() == 32

    # prepare for anonynimization
    prepare_for_anon(df_dev=df_dev, df_test=df_test, output_dir=output_dir, task_name=task_name)


def prepare_reports(
    task_name: str,
    output_dir: Path,
):
    # read anonynimized data
    df_dev = read_anon(output_dir / "anon" / task_name / "nlp-development-dataset.json")
    df_test = read_anon(output_dir / "anon" / task_name / "nlp-test-dataset.json")

    # make test and cross-validation splits
    split_and_save_data(
        df=df_dev,
        df_test=df_test,
        output_dir=output_dir,
        task_name=task_name,
        split_by="PatientID",
    )


if __name__ == "__main__":
    # create the parser
    parser = argparse.ArgumentParser(description="Script for preparing reports")
    parser.add_argument("--task_name", type=str, default="Task007_nodule_diameter_presence_clf",
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
