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
import hashlib
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from dragon_prep.ner import doccano_to_tags
from dragon_prep.utils import (num_patients, prepare_for_anon, read_anon,
                               split_and_save_data)

try:
    from report_anonymizer.model.anonymizer_functions import Anonymizer
except ImportError:
    print("Anonymizer not found, will not be able to run the full pipeline.")


def preprocess_reports(
    task_name: str,
    input_dir: Path,
    output_dir: Path,
):
    # read marksheet (the PHI labels are in the "label" field, the reports are in the "text" field
    # and the task labels are in the meta->label field)
    df = pd.read_json(input_dir / "all_checked_with_phi_annot.jsonl", lines=True)

    # select both sets of labels
    df = df.rename(columns={"label": "label_phi"})
    df["label_task"] = df.meta.apply(lambda meta: meta["label"])

    # extract the metadata
    for key in df["meta"].iloc[0].keys():
        df[key] = df["meta"].apply(lambda x: x[key])
    df = df.drop(["meta", "label"], axis=1)

    print(f"Loaded {len(df)} cases ({num_patients(df)} patients)")

    # exclude cases with "exclude" label
    df = df[~df["label_task"].apply(lambda labels: any(lbl == "exclude" for _, _, lbl in labels))]
    print(f"Have {len(df)} included cases ({num_patients(df)} patients)")

    # exclude single report with naald 7
    mask = df["label_task"].apply(lambda label: any("7-locatie naald" == lbl for _, _, lbl in label))
    assert mask.sum() == 1
    df = df[~mask]
    print(f"Have {len(df)} cases ({num_patients(df)} patients) after excluding single report with naald 7")

    # exclude duplicate cases
    df = df.drop_duplicates(subset=["text"])
    print(f"Have {len(df)} cases ({num_patients(df)} patients) after dropping duplicates")

    # perform anonymization
    df_paths = prepare_for_anon(df=df[~mask], output_dir=output_dir, task_name=task_name, tag_phi=False, apply_hips=False)
    anonymizer = Anonymizer()
    for paths in df_paths:
        # read the reports
        df = pd.read_json(paths["path_for_anon"])
        for i, row in tqdm(df.iterrows(), total=len(df)):
            report = row["text"]
            labels_task = row["meta"]["label_task"]

            # apply PHI annotations
            phi_labels = row["meta"]["label_phi"]
            sorted_labels = sorted(phi_labels, key=lambda x: x[0], reverse=True)
            for start_idx, end_idx, tag in sorted_labels:
                report = report[:start_idx] + tag + report[end_idx:]
                shift = len(tag) - (end_idx - start_idx)
                labels_task = [
                    (start + (shift if start > start_idx else 0), end + (shift if start >= start_idx else 0), label)
                    for (start, end, label) in labels_task
                ]

            # apply HIPS
            md5_hash = hashlib.md5(report.encode())
            seed = int(md5_hash.hexdigest(), 16) % 2 ** 32
            report_anon, labels_anon = anonymizer.HideInPlainSight.apply_hips(report=report, seed=seed, ner_labels=labels_task)
            df.at[i, "text"] = report_anon
            row["meta"]["label_task"] = labels_anon

        # save the reports
        df.to_json(paths["path_anon"], orient="records", indent=2)


def prepare_reports(
    task_name: str,
    output_dir: Path,
    test_split_size: float = 0.3,
):
    # read anonynimized data
    df = read_anon(output_dir / "anon" / task_name / "nlp-dataset.json")
    df = df.rename(columns={"label_task": "label"})

    # tokenize reports
    data = df.to_dict(orient="records")
    data = doccano_to_tags(data)

    # restructure data
    df = pd.DataFrame(data)
    df = df.rename(columns={"label": "multi_label_named_entity_recognition_target"})
    df = df.drop(columns=["text"])

    # make test and cross-validation splits
    split_and_save_data(
        df=df,
        output_dir=output_dir,
        task_name=task_name,
        test_split_size=test_split_size,
        split_by="PatientID",
    )


if __name__ == "__main__":
    # create the parser
    parser = argparse.ArgumentParser(description="Script for preparing reports")
    parser.add_argument("--task_name", type=str, default="Task027_prostate_biopsy_ner",
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
