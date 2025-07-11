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

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

try:
    from report_anonymizer.model.anonymizer_functions import Anonymizer
except ImportError:
    print("Anonymizer not found, will not be able to run the anonymization pipeline.")

INPUT_COLUMN_NAMES = [
    "text",
    "text_parts",
]

TARGET_COLUMN_NAMES = [
    "single_label_multi_class_classification_target",
    "single_label_regression_target",
    "multi_label_regression_target",
    "named_entity_recognition_target",
    "multi_label_multi_class_classification_target",
    "single_label_binary_classification_target",
    "multi_label_binary_classification_target",
    "multi_label_named_entity_recognition_target",
    "text_target",
    "custom_target",
]


def read_marksheet(path: Path | str) -> pd.DataFrame:
    extension = Path(path).suffix
    kwargs = {
        "dtype": str,
        "keep_default_na": False,
        "na_values": ["", " "],
    }
    if extension == ".csv":
        return pd.read_csv(path, **kwargs)
    elif extension == ".xlsx":
        return pd.read_excel(path, engine="openpyxl", **kwargs)
    else:
        raise ValueError(f"Unknown extension: {extension}")


def parse_scores(scores):
    """Parse the PI-RADS scores from the lesion PI-RADS string.

    Args:
        lesion_PIRADS (str): The lesion PI-RADS string.

    Returns:
        list: The PI-RADS scores.
    """
    if isinstance(scores, str):
        scores = scores.replace(".", ",").split(",")
        scores = [int(score) for score in scores if score != "N/A"]
        return scores
    elif isinstance(scores, float) and np.isnan(scores):
        return np.nan
    else:
        raise ValueError(f"Unknown type for scores: {type(scores)}")


def validate_dataframes(
    df: pd.DataFrame,
    input_name: str,
    label_names: Iterable[str],
    df_test: Optional[pd.DataFrame] = None,
):
    """Validate the dataframes."""
    # check if the required columns are present
    required_columns = ["uid", input_name, *label_names]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in dataframe.")
        if df_test is not None and column not in df_test.columns:
            raise ValueError(f"Column {column} not found in dataframe.")

    if df["uid"].dtype not in (str, object):
        raise ValueError(f"The uid should be a string, got {df['uid'].dtype}.")

    # check if the uid is unique (between both df and df_test, if provided)
    all_uid = list(df["uid"].values) + ([] if df_test is None else list(df_test["uid"].values))
    if len(all_uid) != len(set(all_uid)):
        raise ValueError("The uid is not unique.")


def make_cv_splits(
    df: pd.DataFrame,
    seed: int = 42,
    folds: int = 5,
    test_split_size: Optional[float] = None,
    df_test: Optional[pd.DataFrame] = None,
    split_by: str = "patient_id",
) -> Dict[str, pd.DataFrame]:
    """Make train, val and test splits."""
    dataframes: Dict[str, pd.DataFrame] = {}

    # shuffle the patient ids in a reproducible way
    split_ids = df[split_by].unique()
    np.random.seed(seed)
    np.random.shuffle(split_ids)

    # make the test split
    test_patient_ids = []
    if df_test is not None:
        dataframes["test"] = df_test
    elif test_split_size is None:
        raise ValueError("Either `test_split_size` or `df_test` should be provided.")
    elif test_split_size > 0:
        # make and save the test split
        test_patient_ids = split_ids[:int(test_split_size * len(split_ids))]
        print(f"Selected {len(test_patient_ids)} unique {split_by} for the test split.")
        df_test = df[df[split_by].isin(test_patient_ids)]
        dataframes["test"] = df_test

    # make the train and val splits
    df_dev = df[~df[split_by].isin(test_patient_ids)]
    dataframes["dev"] = df_dev
    dev_patient_ids = df_dev[split_by].unique()
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = kf.split(dev_patient_ids)
    for i, (train_idx, val_idx) in enumerate(splits):
        # select the train and val patient ids
        train_patient_ids = dev_patient_ids[train_idx]
        val_patient_ids = dev_patient_ids[val_idx]

        # select the train and val dataframes
        df_train = df_dev[df_dev[split_by].isin(train_patient_ids)]
        df_val = df_dev[df_dev[split_by].isin(val_patient_ids)]

        # collect results
        dataframes[f"train_fold{i}"] = df_train
        dataframes[f"val_fold{i}"] = df_val

    return dataframes


def save_dataframes(
    dataframes: Dict[str, pd.DataFrame],
    output_dir: Path,
    task_name: str,
    input_name: str,
    label_names: Iterable[str],
    folds: int = 5,
    task_details: Optional[Dict[str, Any]] = None,
) -> None:
    df_test = dataframes.get("test", None)
    columns = ["uid", input_name, *label_names]

    for fold in range(folds):
        # make the output directory
        save_dir = output_dir / "algorithm-input" / f"{task_name}-fold{fold}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # save the train and val dataframes
        dataframes[f"train_fold{fold}"][columns].to_json(save_dir / "nlp-training-dataset.json", orient="records", indent=2)
        dataframes[f"val_fold{fold}"][columns].to_json(save_dir / "nlp-validation-dataset.json", orient="records", indent=2)

        # save the test dataframe for inference
        if df_test is not None:
            df_test[["uid", input_name]].to_json(save_dir / "nlp-test-dataset.json", orient="records", indent=2)

        if task_details is not None:
            # save the dataset details
            task_nr = int(task_name.split("_")[0].split("Task")[1])
            task_nr = task_nr * 10 + fold
            task_details["jobid"] = task_nr
            with open(save_dir / "nlp-task-configuration.json", "w") as f:
                json.dump(task_details, f, indent=2)

    # save the test dataframe for testing
    if df_test is not None:
        path = output_dir / "test-set" / f"{task_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        df_test[columns].to_json(path, orient="records", indent=2)


def prepare_for_anon(
    output_dir: Path,
    task_name: str,
    df: Optional[pd.DataFrame] = None,
    df_dev: Optional[pd.DataFrame] = None,
    df_test: Optional[pd.DataFrame] = None,
    tag_phi: bool = True,
    apply_hips: bool = True,
):
    """
    Save the dataframes.
    """
    if apply_hips is False and tag_phi is True:
        raise ValueError("Cannot tag PHI without applying HIPS.")

    save_dir_for_anon = Path(output_dir, "for-anon", task_name)
    save_dir_for_anon.mkdir(parents=True, exist_ok=True)
    save_dir_anon = Path(output_dir, "anon", task_name)
    save_dir_anon.mkdir(parents=True, exist_ok=True)
    df_paths = []

    # save dataframes
    for df, name in [
        (df, "nlp-dataset"),
        (df_dev, "nlp-development-dataset"),
        (df_test, "nlp-test-dataset"),
    ]:
        if df is not None:
            # place metadata in "meta" dictionary
            df["meta"] = df.apply(lambda row: {key: row[key] for key in df.columns if key != "text"}, axis=1)
            df = df[["text", "meta"]]

            # save the dataframe
            path_for_anon = save_dir_for_anon / f"{name}.json"
            path_anon = save_dir_anon / f"{name}.json"
            df.to_json(path_for_anon, orient="records", indent=2)
            df_paths.append({"path_for_anon": path_for_anon, "path_anon": path_anon})

            if tag_phi:
                # anonymize the reports
                cmd = [
                    "anonymize_reports",
                    "--input", path_for_anon.as_posix(),
                    "--output", path_anon.as_posix(),
                ]
                subprocess.check_call(cmd)
            elif apply_hips:
                anonymizer = Anonymizer()
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    report = row["text"]

                    # apply HIPS
                    md5_hash = hashlib.md5(report.encode())
                    seed = int(md5_hash.hexdigest(), 16) % 2 ** 32
                    report_anon = anonymizer.HideInPlainSight.apply_hips(report=report, seed=seed)

                    # save
                    df.at[i, "text"] = report_anon

                # sanity check
                for i, row in df.iterrows():
                    report = row["text"]
                    remaining_phi_tags = re.findall(r"<[a-zA-Z0-9\.\-\_]{1,50}>", report)
                    for res in remaining_phi_tags:
                        if res not in ["<st0>", "</st0>", "<st1>", "</st1>", "<st2>", "</st2>", "<beschermd>", "</beschermd>"]:
                            raise ValueError(f"Found remaining PHI tag: {res} in '{report}'")

                # save the reports
                df.to_json(path_anon, orient="records", indent=2)

    return df_paths


def read_anon(path: Path | str) -> pd.DataFrame:
    """Read the anonymized dataset."""
    # read the dataframe
    df = pd.read_json(path)

    # extract the metadata
    for key in df["meta"].iloc[0].keys():
        df[key] = df["meta"].apply(lambda x: x[key])
    df = df.drop("meta", axis=1)

    return df


def split_and_save_data(
    df: pd.DataFrame,
    task_name: str,
    output_dir: Path | str | None = None,
    seed: int = 42,
    folds: int = 5,
    test_split_size: float | None = None,
    df_test: pd.DataFrame | None = None,
    split_by: str = "patient_id",
    recommended_truncation_side: str = "left",
    anonymize_uid: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int], Dict[str, Any]]:
    """Make train, val and test splits.

    Args:
        df (pd.DataFrame): DataFrame with the data. Should contain columns "uid" (str), "text" (str) OR "text_parts" (array of strings), `split_by` (str) and targets.
            the targets may be one or multiple columns of:
            - "single_label_multi_class_classification_target": must be a string for each case.
            - "single_label_regression_target": must be a float for each case.
            - "multi_label_regression_target": must be an array of floats for each case.
            - "named_entity_recognition_target": must be an array of strings for each case.
            - "multi_label_named_entity_recognition_target": must be an array of arrays of strings for each case.
            - "multi_label_multi_class_classification_target": must be an array of strings for each case.
            - "single_label_binary_classification_target": must be an int for each case.
            - "multi_label_binary_classification_target": must be an array of ints (each either 0 or 1) for each case.
            - "text_target": must be a string for each case.
            - "custom_target": custom target column (experimental).
        task_name (str): The name of the task. The splits will be saved in a directory with this name.
        output_dir (Path or str, optional): The output directory. Defaults to None (don't save anything).
        seed (int, optional): The random seed. Defaults to 42.
        folds (int, optional): The number of folds. Defaults to 5.
        test_split_size (float, optional): The size of the test split. Defaults to 0.3 if df_test is not provided.
        df_test (pd.DataFrame, optional): The dataframe with the test data. Defaults to None (make subset from `df`).
        split_by (str, optional): The column to use for splitting the data between test/train/val. Defaults to "patient_id".
    """
    # input validation
    output_dir = Path(output_dir) if output_dir is not None else None
    if test_split_size is not None and test_split_size > 0 and df_test is not None:
        raise ValueError("Cannot specify both `test_split_size > 0` and `df_test`.")
    elif test_split_size is None:
        test_split_size = 0.3 if df_test is None else 0

    # select the label names
    label_names = [col for col in TARGET_COLUMN_NAMES if col in df.columns]
    if len(label_names) == 0:
        raise ValueError(f"Could not find any label column. Allowed are: {TARGET_COLUMN_NAMES}.")
    elif len(label_names) > 1:
        raise ValueError(f"Found multiple label columns. Allowed is one of: {TARGET_COLUMN_NAMES}.")

    # select the input name
    input_names = [col for col in INPUT_COLUMN_NAMES if col in df.columns]
    if len(input_names) == 0:
        raise ValueError(f"Could not find input column. Allowed is one of: {INPUT_COLUMN_NAMES}.")
    elif len(input_names) > 1:
        raise ValueError(f"Found multiple input columns. Allowed is one of: {INPUT_COLUMN_NAMES}.")
    input_name = input_names[0]

    # validate dataframes
    validate_dataframes(
        df=df,
        df_test=df_test,
        input_name=input_name,
        label_names=label_names,
    )

    if anonymize_uid:
        # override the uid
        df["uid"] = [f"{task_name}_case{idx}" for idx in range(len(df))]
        if df_test is not None:
            df_test["uid"] = [f"{task_name}_test_case{idx}" for idx in range(len(df_test))]

    # make the splits
    dataframes = make_cv_splits(
        df=df,
        seed=seed,
        folds=folds,
        test_split_size=test_split_size,
        df_test=df_test,
        split_by=split_by,
    )

    # make dataset details
    task_details = make_task_details(
        task_name=task_name,
        input_name=input_name,
        label_name=label_names[0],
        recommended_truncation_side=recommended_truncation_side,
        jobid=0,
    )

    if output_dir is not None:
        # save the splits
        save_dataframes(
            dataframes=dataframes,
            output_dir=output_dir,
            task_name=task_name,
            input_name=input_name,
            label_names=label_names,
            folds=folds,
            task_details=task_details,
        )

    # collect results and return
    dataframes = {k: v[["uid", split_by, *input_names, *label_names]] for k, v in dataframes.items()}
    split_sizes = {k: len(v) for k, v in dataframes.items()}

    print(f"{task_name} split sizes: {split_sizes}")

    return dataframes, split_sizes, task_details


def make_task_details(
    task_name: str,
    input_name: str,
    label_name: str,
    recommended_truncation_side: str = "right",
    jobid: int = -1,
):
    """Make the dataset details."""

    task_details = {
        "jobid": int(jobid),
        "task_name": str(task_name),
        "input_name": str(input_name),
        "label_name": label_name,
        "recommended_truncation_side": str(recommended_truncation_side),
        "version": "1.0",
    }

    return task_details


def apply_anon_annotations(
    row: pd.Series,
    label_name: Optional[str] = None,
) -> pd.Series:
    """Apply manual anonymization."""
    if label_name is None:
        if "label" in row:
            label_name = "label"
        elif "labels" in row:
            label_name = "labels"
        elif "label_phi" in row:
            label_name = "label_phi"
        else:  # pragma: no cover
            raise ValueError("Could not find the label column.")

    labels = row[label_name]
    for start, end, lbl in labels:
        if "<" in lbl and ">" in lbl:
            selected_text = row["text"][start:end]

            # anonymize the text
            row["text"] = row["text"][:start] + lbl + row["text"][end:]

            # shift the labels
            label_shift = len(lbl) - len(selected_text)
            for i, (s, e, l) in enumerate(labels):
                if s > start:
                    s += label_shift
                if e > start:
                    e += label_shift
                labels[i] = (s, e, l)

    return row


def num_patients(df: pd.DataFrame) -> int:
    """
    Get the number of unique patients in a dataframe
    """
    for column_name in ["patient_id", "PatientID", "pid", "patient", "anon_patientid", "MDN", "pa"]:
        if column_name in df.columns:
            return len(df[column_name].unique())
    raise ValueError("Could not find a patient_id, PatientID, or pid column.")
