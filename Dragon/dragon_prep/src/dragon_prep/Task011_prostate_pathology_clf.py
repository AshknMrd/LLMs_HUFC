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
import re
from pathlib import Path

import numpy as np
import pandas as pd

from dragon_prep.prepare_avl import prepare_avl_pathology_reports
from dragon_prep.prepare_umcg import prepare_umcg_pathology_reports
from dragon_prep.utils import (apply_anon_annotations, parse_scores,
                               prepare_for_anon, read_anon,
                               split_and_save_data)


def count_isup_2345(s: str | float) -> int:
    scores = parse_scores(s)
    if isinstance(scores, list):
        return sum(np.array(scores) >= 2)
    elif isinstance(scores, float) and np.isnan(scores):
        return 0
    else:
        raise ValueError(f"Unexpected value {scores}")


def calculate_gleason_score(input_string: str) -> int:
    # Check for no malignancy
    no_malignancy_phrases = ["geen", "zonder"]
    if any(phrase in input_string.lower() for phrase in no_malignancy_phrases):
        return 0

    # Find all numbers in the input string
    numbers = [int(num) for num in re.findall(r'\d+', input_string)]

    # Calculate the Gleason score based on the number of digits found
    if len(numbers) == 1:
        return numbers[0]
    elif len(numbers) == 2:
        return sum(numbers)
    elif len(numbers) == 3:
        # Find the two lowest numbers and sum them
        sorted_numbers = sorted(numbers)
        if sorted_numbers[0] + sorted_numbers[1] == sorted_numbers[2]:
            return sorted_numbers[2]
        else:
            # Invalid Gleason score
            return -1
    else:
        # If there are no numbers or the format is unexpected, return an error or zero
        raise ValueError(f"Invalid Gleason score: {input_string}, numbers: {numbers}")


def count_isup_2345_umcg(row: pd.Series) -> int | None:
    text = row["text"]
    num_isup_2345 = 0
    for start, end, label in row["label"]:
        selected_text = text[start:end]
        total_score = calculate_gleason_score(selected_text)
        num_lesions = {"lesion 1": 1, "lesion 2": 1, "lesion 3": 1, "lesion 4": 1, "lesion 5": 1, "2 lesions": 2, "3 lesions": 3, "4+ lesions": 4}[label]
        if total_score >= 7:
            num_isup_2345 += num_lesions
        elif total_score == -1:
            print(f"Invalid Gleason score for {row.uid}: {selected_text}")
            return None

    return num_isup_2345


def num_patients(df: pd.Series):
    return len(df["patient_id"].unique())


def prepare_rumc_reports(
    input_dir: Path,
) -> pd.DataFrame:
    # read marksheet
    df = pd.read_json(input_dir / "PICAI-PubPrivTrain-patient-level-marksheet_v2_with_pathology_study_ids.jsonl", lines=True)
    df["uid"] = df.apply(lambda row: row["pathology_study_id"] if row["pathology_study_id"] else f"{row['patient_id']}_{row['study_id']}", axis=1)

    # select cases from RUMC
    df = df[df["center"] == "RUMC"]
    print(f"Have {len(df)} cases ({num_patients(df)} patients) from RUMC")

    # prepare data for scoring
    # drop rows without a pathology report
    df = df.dropna(subset=["pathology_report"])
    df["text"] = df["pathology_report"]
    print(f"Have {len(df)} cases ({num_patients(df)} patients) from RUMC with pathology reports")

    # drop duplicates (this can happen if the same pathology study was matched with multiple radiology studies)
    df = df.drop_duplicates(subset=["pathology_report"])
    print(f"Have {len(df)} cases ({num_patients(df)} patients) from RUMC after dropping duplicates")

    # compute the number of ISUP >= 2 lesions (which is equivalent to Gleason score >= 7)
    df["label"] = df["lesion_ISUP"].apply(lambda scores: count_isup_2345(scores))

    # update reports with checked PHI tags
    df_reports = pd.read_json(input_dir / "pathology/all.jsonl", lines=True)
    df_reports = df_reports.apply(apply_anon_annotations, axis=1)
    df_reports["uid"] = df_reports.meta.apply(lambda meta: meta["uid"])

    # merge with radiology reports
    for i, row in df.iterrows():
        df.loc[i, "text"] = df_reports[df_reports["uid"] == row["uid"]]["text"].values[0]

    return df


def prepare_umcg_reports(
    input_dir: Path,
) -> pd.DataFrame:
    df = prepare_umcg_pathology_reports(input_dir=input_dir)

    # parse dataset
    umcg_entries = []
    for subject_id in df["subject_id"].unique():
        # select entries for this study
        study_df: pd.DataFrame = df[df["subject_id"] == subject_id]
        study_df = study_df.drop_duplicates(subset=["location_description"])

        # parse and count Gleason score >= 7 scores (which is equivalent to ISUP >= 2)
        scores = [
            (int(row.gg1) + int(row.gg2))
            for _, row in study_df.iterrows()
            if row.gg1 != "NA"
        ]

        umcg_entries.append({
            "label": sum([score >= 7 for score in scores]),
            "lesion_GS": ",".join([f"{row.gg1}+{row.gg2}" for _, row in study_df.iterrows()]).replace("NA+NA", "N/A"),
            **study_df.iloc[0].to_dict(),
        })

    df = pd.DataFrame(umcg_entries)
    df["uid"] = df["subject_id"]

    return df


def prepare_avl_reports(
    input_dir: Path,
) -> pd.DataFrame:
    # read marksheet
    df = prepare_avl_pathology_reports(input_dir)

    # filter out excluded cases
    mask = df["label"].apply(lambda labels: any([lbl == "exclude" for (_, _, lbl) in labels]))
    print(f"Excluding {len(df[mask])} cases ({num_patients(df[mask])} patients) from AVL")
    df = df[~mask]
    print(f"Have {len(df)} cases ({num_patients(df)} patients) from AVL after filtering")

    # filter out anonymization labels
    df["label"] = df["label"].apply(lambda labels: [(start, end, label) for (start, end, label) in labels if not ("<" in label and ">" in label)])

    # select labels for ISUP >= 2 counts
    df["label"] = df.apply(count_isup_2345_umcg, axis=1)

    # exclude cases with invalid Gleason scores
    mask = df["label"].isna()
    print(f"Excluding {len(df[mask])} cases ({num_patients(df[mask])} patients) from AVL due to invalid Gleason scores")
    df = df[~mask]
    print(f"Have {len(df)} cases ({num_patients(df)} patients) from AVL after filtering")

    return df


def preprocess_reports(
    task_name: str,
    input_dir: Path,
    output_dir: Path,
):
    # read RUMC marksheet
    df_rumc = prepare_rumc_reports(input_dir / "rumc/prostate")
    print(f"Have {len(df_rumc)} cases ({num_patients(df_rumc)} patients) from RUMC")

    # read UMCG marksheet
    df_umcg = prepare_umcg_reports(input_dir / "umcg/prostate/pathology")
    print(f"Have {len(df_umcg)} cases ({num_patients(df_umcg)} patients) from UMCG")

    # read AvL marksheet
    df_avl = prepare_avl_reports(input_dir / "avl/prostate/pathology")
    print(f"Have {len(df_avl)} cases ({num_patients(df_avl)} patients) from AVL")

    # merge dataframes
    cols = ["uid", "patient_id", "study_id", "label", "text"]
    assert set(df_rumc.patient_id) & set(df_umcg.patient_id) == set()
    assert set(df_rumc.patient_id) & set(df_avl.patient_id) == set()
    assert set(df_umcg.patient_id) & set(df_avl.patient_id) == set()
    df = pd.concat((df_rumc[cols], df_umcg[cols], df_avl[cols]), ignore_index=True)
    print(f"Have {len(df)} cases ({num_patients(df)} patients) in total")

    # exclude cases with label 4, as there are only 7 of them
    assert len(df[df["label"] == 4]) == 7, f"Unexpected number of cases with label 4, found {len(df[df['label'] == 4])}"
    df = df[df["label"] != 4]
    print(f"Have {len(df)} cases ({num_patients(df)} patients) after excluding label 4")

    # prepare labels
    df["label"] = df["label"].astype(str)
    df.rename(columns={"label": "single_label_multi_class_classification_target"}, inplace=True)

    # prepare for anonynimization
    assert df.text.apply(lambda text: "<PERSOON>" in text).sum() > 600, f"Unexpected number of <PERSOON> tags in RUMC reports: {df.text.apply(lambda text: '<PERSOON>' in text).sum()}"
    assert df.text.apply(lambda text: "<PERSOONAFKORTING>" in text).sum() > 1500, \
        f"Unexpected number of <PERSOONAFKORTING> tags in RUMC reports: {df.text.apply(lambda text: '<PERSOONAFKORTING>' in text).sum()}"
    assert df.text.apply(lambda text: "<DATUM>" in text).sum() > 1500, f"Unexpected number of <DATUM> tags in RUMC reports: {df.text.apply(lambda text: '<DATUM>' in text).sum()}"
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
    )


if __name__ == "__main__":
    # create the parser
    parser = argparse.ArgumentParser(description="Script for preparing reports")
    parser.add_argument("--task_name", type=str, default="Task011_prostate_pathology_clf",
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
