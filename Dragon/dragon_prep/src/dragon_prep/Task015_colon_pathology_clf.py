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

from dragon_prep.utils import (num_patients, prepare_for_anon, read_anon,
                               split_and_save_data)


def preprocess_reports(
    task_name: str,
    input_dir: Path,
    output_dir: Path,
):
    # read marksheets
    df = pd.read_excel(input_dir / "colon_rumc_gt.xlsx")
    print(f"Found {len(df)} slides from {len(df.block.unique())} blocks of {num_patients(df)} patients.")

    # for all patiens, check if the report is available
    reports = {}
    for pa in df["pa"].unique():
        patient_df = df[df["pa"] == pa]
        patient_df = patient_df.dropna(subset=["Conclusion"])
        conclusions = patient_df["Conclusion"].unique()
        if len(conclusions) > 1:
            raise ValueError(f"Patient {pa} has multiple conclusions: {conclusions}")
        elif len(conclusions) == 0:
            print(f"Patient {pa} has no conclusion.")
        else:
            reports[pa] = conclusions[0]

    # select patients with a report
    df = df[df["pa"].isin(reports.keys())]
    print(f"Have {len(df)} slides from {len(df.block.unique())} blocks of {num_patients(df)} patients with a report.")

    # exclude patients with excluded annotations
    df = df[df["exclude"] != 1]
    print(f"Have {len(df)} slides from {len(df.block.unique())} blocks of {num_patients(df)} patients after excluding.")

    # select only the relevant columns:
    df = df[["pa", "block", "block_nr", "biopsy", "cancer", "hgd", "hyperplastic", "lgd", "ni", "serrated"]]

    # drop duplicates
    df = df.drop_duplicates()
    print(f"Have {len(df)} slides from {len(df.block.unique())} blocks of {num_patients(df)} patients after dropping duplicates.")

    # sanity check: all blocks are unique
    assert len(df) == len(df["block"].unique())

    # add report
    df["text"] = df["pa"].map(reports)

    # fix PHI tag spelling
    df["text"] = df["text"].str.replace("<T-NUMMER>", "<RAPPORT_ID.T_NUMMER>")

    # rename `block` to `uid`
    df = df.rename(columns={"block": "uid"})

    # exclude non-Dutch reports
    with open(input_dir / "excluded_cases_not_dutch.json") as f:
        excluded_cases = json.load(f)
    mask = df["uid"].isin(excluded_cases)
    num_cases_excluded = df[mask].uid.apply(lambda x: "_".join(x.split("_")[:-1])).nunique()
    df = df[~mask]
    print(f"Excluded {mask.sum()} blocks ({num_cases_excluded} cases) with a non-Dutch report")
    print(f"Have {len(df)} blocks ({num_patients(df)} patients) for RUMC with a Dutch report")

    # for each label, confirm it's binary and convert to int
    label_names = ["biopsy", "cancer", "hgd", "hyperplastic", "lgd", "ni", "serrated"]
    for label in label_names:
        assert len(df[label].unique()) == 2
        df[label] = df[label].astype(bool)
        print(f"Have {df[label].sum()}/{len(df)} blocks with {label}.")

    # convert block_nr to roman numerals
    df["block_nr"] = df["block_nr"].map({0: "NA", 1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X", 11: "XI", 12: "XII", 13: "XIII", 14: "XIV", 15: "XV"})

    # collect algorithm targets
    df["multi_label_binary_classification_target"] = df.apply(lambda row: row[label_names].tolist(), axis=1)

    # prepare for anonynimization
    prepare_for_anon(df=df, output_dir=output_dir, task_name=task_name)


def prepare_reports(
    task_name: str,
    output_dir: Path,
    test_split_size: float = 0.3,
):
    # read anonynimized data
    df = read_anon(output_dir / "anon" / task_name / "nlp-dataset.json")

    # collect algorithm inputs
    df["text_parts"] = df.apply(lambda row: [row["block_nr"], row["text"]], axis=1)
    df = df.drop(columns=["text"])

    # make test and cross-validation splits
    split_and_save_data(
        df=df,
        output_dir=output_dir,
        task_name=task_name,
        test_split_size=test_split_size,
        split_by="pa",
    )


if __name__ == "__main__":
    # create the parser
    parser = argparse.ArgumentParser(description="Script for preparing reports")
    parser.add_argument("-i", "--input", type=Path, default=Path("/input"),
                        help="Path to the input data")
    parser.add_argument("-o", "--output", type=Path, default=Path("/output"),
                        help="Folder to store the prepared reports in")
    parser.add_argument("--test_split_size", type=float, default=0.3,
                        help="Fraction of the dataset to use for testing")
    parser.add_argument("--task_name", type=str, default="Task015_colon_pathology_clf",
                        help="Name of the task")
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
