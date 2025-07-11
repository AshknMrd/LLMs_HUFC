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
    # read test set annotations
    df_test = pd.read_csv(input_dir / "annotations.csv", index_col=0)
    df_test["uid"] = df_test["StudyInstanceUID"]

    # read reports into a dictionary with StudyInstanceUID as key
    path = input_dir / "MarNavarro.jsonl"
    reports = {}
    patient_id_map: dict[str, str] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                line = json.loads(line)
                reports[line["metadata"]["StudyInstanceUID"]] = line["text"]
                patient_id_map[line["metadata"]["StudyInstanceUID"]] = line["metadata"]["PatientID"]

    # add reports to test set dataframe
    df_test["text"] = df_test["StudyInstanceUID"].map(reports)

    # add patient ID
    df_test["patient_id"] = df_test["StudyInstanceUID"].map(patient_id_map)
    print(f"Have {len(df_test)} cases ({num_patients(df_test)} patients) for testing.")

    assert df_test.duplicated(subset=["text"]).sum() == 0, "Duplicate reports found in test set"

    # prepare labels
    df_test.loc[df_test["left_annotation"] == "5", "left_annotation"] = "p"  # hip prosthesis
    df_test.loc[df_test["right_annotation"] == "5", "right_annotation"] = "p"  # hip prosthesis
    df_test["multi_label_multi_class_classification_target"] = df_test.apply(
        lambda row: row[["left_annotation", "right_annotation"]].tolist(), axis=1
    )

    # read development set annotations (with columns ,Unnamed: 0,StudyInstanceUID,Left_model,Right_model)
    df_dev = pd.concat([
        pd.read_csv(input_dir / "openai_final_all.csv", index_col=0),
        pd.read_csv(input_dir / "openai_test_all.csv", index_col=0),
    ], ignore_index=True)

    # extract the patient ID and StudyInstanceUID from the filename
    df_dev["patient_id"] = df_dev["StudyInstanceUID"].str.split("-").str[0].astype(int)
    df_dev["StudyInstanceUID"] = df_dev["StudyInstanceUID"].str.split("-").str[1]
    print(f"Have {len(df_dev)} cases ({num_patients(df_dev)} patients) for development.")

    # remove patients present in the test set
    df_dev = df_dev[~df_dev.patient_id.isin(df_test.patient_id)]
    print(f"Have {len(df_dev)} cases ({num_patients(df_dev)} patients) for development after removing patients present in the test set.")

    # drop rows with duplicate StudyInstanceUID
    df_dev = df_dev.drop_duplicates(subset=["StudyInstanceUID"])
    print(f"Have {len(df_dev)} cases ({num_patients(df_dev)} patients) for development after removing duplicates.")

    # set uid to StudyInstanceUID
    df_dev["uid"] = df_dev["StudyInstanceUID"]

    # remove rows with missing annotations
    df_dev = df_dev[(df_dev["Left_model"] != "False") & (df_dev["Right_model"] != "False")]
    print(f"Have {len(df_dev)} cases ({num_patients(df_dev)} patients) for development after removing missing annotations.")

    # add reports to development set dataframe
    df_dev["text"] = df_dev["StudyInstanceUID"].map(reports)
    df_dev["multi_label_multi_class_classification_target"] = df_dev.apply(
        lambda row: row[["Left_model", "Right_model"]].tolist(), axis=1
    )

    # drop overlapping reports
    df_dev = df_dev.drop_duplicates(subset=["text"])
    print(f"Have {len(df_dev)} cases ({num_patients(df_dev)} patients) for development after removing overlapping reports.")

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
        test_split_size=0,
        split_by="StudyInstanceUID",
    )


if __name__ == "__main__":
    # create the parser
    parser = argparse.ArgumentParser(description="Script for preparing reports")
    parser.add_argument("--task_name", type=str, default="Task018_osteoarthritis_clf",
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
