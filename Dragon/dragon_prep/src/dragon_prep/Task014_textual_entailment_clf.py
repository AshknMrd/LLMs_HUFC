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

from dragon_prep.utils import split_and_save_data


def preprocess_reports(
    task_name: str,
    input_dir: Path,
    output_dir: Path,
):
    # read marksheets
    df_train = pd.read_csv(input_dir / "MEDNLI_train_dutch_dl.txt", sep="\t", header=None, names=["premise", "hypothesis", "label"])
    df_dev = pd.read_csv(input_dir / "MEDNLI_dev_dutch_dl.txt", sep="\t", header=None, names=["premise", "hypothesis", "label"])
    df_test = pd.read_csv(input_dir / "MEDNLI_test_dutch_dl.txt", sep="\t", header=None, names=["premise", "hypothesis", "label"])

    # add uid
    df_train["uid"] = (df_train.index).astype(str)
    df_dev["uid"] = (df_dev.index + len(df_train)).astype(str)
    df_test["uid"] = (df_test.index + len(df_train) + len(df_dev)).astype(str)

    # combine splits
    df_dev = pd.concat([df_train, df_dev], axis=0, ignore_index=True)

    # strip whitespace
    df_dev["premise"] = df_dev["premise"].str.strip()
    df_test["premise"] = df_test["premise"].str.strip()
    df_dev["hypothesis"] = df_dev["hypothesis"].str.strip()
    df_test["hypothesis"] = df_test["hypothesis"].str.strip()
    df_dev["label"] = df_dev["label"].str.strip()
    df_test["label"] = df_test["label"].str.strip()

    # rename columns
    df_dev["text_parts"] = df_dev.apply(lambda row: [row["premise"], row["hypothesis"]], axis=1)
    df_test["text_parts"] = df_test.apply(lambda row: [row["premise"], row["hypothesis"]], axis=1)
    df_dev.rename(columns={"label": "single_label_multi_class_classification_target"}, inplace=True)
    df_test.rename(columns={"label": "single_label_multi_class_classification_target"}, inplace=True)

    # make test and cross-validation splits
    split_and_save_data(
        df=df_dev,
        df_test=df_test,
        output_dir=output_dir,
        task_name=task_name,
        split_by="premise",
    )


if __name__ == "__main__":
    # create the parser
    parser = argparse.ArgumentParser(description="Script for preparing reports")
    parser.add_argument("--task_name", type=str, default="Task014_textual_entailment_clf",
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
