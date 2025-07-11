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

from dragon_prep.utils import (num_patients, prepare_for_anon, read_anon,
                               split_and_save_data)


def preprocess_reports(
    task_name: str,
    marksheet_path: Path | str,
    output_dir: Path,
):
    # read marksheet
    df = pd.read_csv(marksheet_path)
    print(f"Have {len(df)} reports from {num_patients(df)} patients.")

    # prepare marksheet
    df["uid"] = [f"{task_name}_case{idx}" for idx in range(len(df))]
    df["label"] = df["label"].astype(bool)
    df = df.rename(columns={"report": "text", "label": "single_label_binary_classification_target"})

    # prepare for anonynimization
    prepare_for_anon(df=df, output_dir=output_dir, task_name=task_name)


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
        split_by="patient",
    )


if __name__ == "__main__":
    # create the parser
    parser = argparse.ArgumentParser(description="Script for preparing reports")
    parser.add_argument("--task_name", type=str, default="Task001_adhesion_clf",
                        help="Name of the task")
    parser.add_argument("-i", "--input", type=Path, default="/input/reports.csv",
                        help="Path to the input data")
    parser.add_argument("-o", "--output", type=Path, default=Path("/output"),
                        help="Folder to store the prepared reports in")
    parser.add_argument("--test_split_size", type=float, default=0.3,
                        help="Fraction of the dataset to use for testing")
    args = parser.parse_args()

    # run preprocessing
    preprocess_reports(
        task_name=args.task_name,
        marksheet_path=args.input,
        output_dir=args.output,
    )
    prepare_reports(
        task_name=args.task_name,
        output_dir=args.output,
        test_split_size=args.test_split_size,
    )
