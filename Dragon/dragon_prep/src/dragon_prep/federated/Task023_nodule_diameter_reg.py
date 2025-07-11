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

from dragon_prep.Task023_nodule_diameter_reg import preprocess_reports
from dragon_prep.utils import read_anon, split_and_save_data


def prepare_reports(
    task_name: str,
    output_dir: Path,
):
    # read anonynimized data
    df_dev = read_anon(output_dir / "anon" / task_name / "nlp-development-dataset.json")
    df_test = read_anon(output_dir / "anon" / task_name / "nlp-test-dataset.json")

    # make test and cross-validation splits
    df_dev["text_parts"] = df_dev.apply(lambda row: [row["hospital"], row["text"]], axis=1)
    df_dev = df_dev.drop(columns=["hospital", "text"])
    df_test["text_parts"] = df_test.apply(lambda row: [row["hospital"], row["text"]], axis=1)
    df_test = df_test.drop(columns=["hospital", "text"])
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
    parser.add_argument("--task_name", type=str, default="Task023_nodule_diameter_reg",
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
