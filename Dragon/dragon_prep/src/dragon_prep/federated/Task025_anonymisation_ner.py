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
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from dragon_prep.ner import doccano_to_bio_tags
from dragon_prep.Task025_anonymisation_ner import (
    combine_phi_labels, preprocess_reports_avl,
    preprocess_reports_rumc_bcc_pathology,
    preprocess_reports_rumc_lung_pathology,
    preprocess_reports_rumc_prostate_biopsy_procedure,
    preprocess_reports_rumc_prostate_pathology,
    preprocess_reports_rumc_thorax_abdomen_ct)
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
    # validate input
    cols = ["uid", "PatientID", "text", "label"]

    # read reports and labels from RUMC
    df_rumc_ct = preprocess_reports_rumc_thorax_abdomen_ct(
        input_dir=input_dir / "rumc/anonymisation/ct-thorax-abdomen",
    )
    df_rumc_prostate_pathology = preprocess_reports_rumc_prostate_pathology(
        input_dir=input_dir / "rumc/anonymisation/pathology-prostate",
    )
    df_rumc_bcc = preprocess_reports_rumc_bcc_pathology(
        input_dir=input_dir / "rumc/bcc",
    )
    df_rumc_prostate_biopsy = preprocess_reports_rumc_prostate_biopsy_procedure(
        input_dir=input_dir / "rumc/prostate-biopsy",
    )
    df_rumc_lung_pathology = preprocess_reports_rumc_lung_pathology(
        input_dir=input_dir / "rumc/pathology-lung",
    )
    df_rumc = pd.concat([
        df_rumc_ct[cols],
        df_rumc_prostate_pathology[cols],
        df_rumc_bcc[cols],
        df_rumc_prostate_biopsy[cols],
        df_rumc_lung_pathology[cols],
    ], ignore_index=True)
    print(f"Have {len(df_rumc)} reports ({num_patients(df_rumc)} patients) for RUMC after combining datasets")
    df_rumc["center"] = "RUMC"

    # read reports and labels from AvL
    df_avl = preprocess_reports_avl(
        input_dir=input_dir / "avl/prostate",
    )
    df_avl["center"] = "AVL"

    # combine the datasets
    cols = cols + ["center"]
    df = pd.concat([df_rumc[cols], df_avl[cols]], ignore_index=True)

    print(f"Have {len(df)} reports ({num_patients(df)} patients) after combining datasets")

    # combine equivalent labels
    df["label"] = df.label.apply(lambda x: [[
        int(start), int(end), label.replace("<NAAM>", "<PERSOON>").replace("<TNUMMER>", "<RAPPORT_ID>")
    ] for start, end, label in x])

    # print label statistics
    all_labels = [label[2] for labels in df.label for label in labels]
    print(f"Have {len(all_labels)} labels in total, with {len(set(all_labels))} unique labels")
    print(pd.Series(all_labels).value_counts())

    # perform anonynimization
    df = df[cols]
    df_paths = prepare_for_anon(df=df, output_dir=output_dir, task_name=task_name, tag_phi=False, apply_hips=False)
    anonymizer = Anonymizer()
    for paths in df_paths:
        # read the reports
        df = pd.read_json(paths["path_for_anon"])
        for i, row in tqdm(df.iterrows(), total=len(df)):
            report = row["text"]

            # apply PHI annotations
            phi_labels_orig = row["meta"]["label"]
            sorted_labels = sorted(phi_labels_orig, key=lambda x: x[0], reverse=True)
            phi_labels_tags = phi_labels_orig
            for start_idx, end_idx, tag in sorted_labels:
                report = report[:start_idx] + tag + report[end_idx:]
                shift = len(tag) - (end_idx - start_idx)
                phi_labels_tags = [
                    (start + (shift if start > start_idx else 0), end + (shift if start >= start_idx else 0), label)
                    for (start, end, label) in phi_labels_tags
                ]

            # verify the PHI annotations and shifted labels
            for start_idx, end_idx, tag in phi_labels_tags:
                assert report[start_idx:end_idx] == tag, f"Expected '{tag}' at {start_idx}:{end_idx} in '{report[start_idx-10:start_idx]}|{report[start_idx:end_idx]}|{report[end_idx:end_idx+10]}'"

            # apply HIPS
            md5_hash = hashlib.md5(report.encode())
            seed = int(md5_hash.hexdigest(), 16) % 2 ** 32
            report_anon, phi_labels_anon = anonymizer.HideInPlainSight.apply_hips(report=report, seed=seed, ner_labels=phi_labels_tags)

            # save
            df.at[i, "text"] = report_anon
            row["meta"]["label"] = phi_labels_anon

        # sanity check
        for i, row in df.iterrows():
            report = row["text"]
            remaining_phi_tags = re.findall(r"<[a-zA-Z0-9\.\-\_]{1,50}>", report)
            for res in remaining_phi_tags:
                if res not in ["<st0>", "</st0>", "<st1>", "</st1>", "<st2>", "</st2>", "<beschermd>", "</beschermd>"]:
                    raise ValueError(f"Found remaining PHI tag: {res} in '{report}'")

        # save the reports
        df.to_json(paths["path_anon"], orient="records", indent=2)


def prepare_reports(
    task_name: str,
    output_dir: Path,
    test_split_size: float = 0.3,
):
    # read anonynimized data
    df = read_anon(output_dir / "anon" / task_name / "nlp-dataset.json")

    # merge particular labels (after anonymisation using HIPS)
    df["label"] = df.label.apply(lambda x: [
        [int(start), int(end), combine_phi_labels(label)]
        for start, end, label in x
    ])

    # print label statistics
    all_labels = [label[2] for labels in df.label for label in labels]
    print(f"Have {len(all_labels)} labels in total, with {len(set(all_labels))} unique labels")
    print(pd.Series(all_labels).value_counts())

    # tokenize reports
    data = df.to_dict(orient="records")
    data = doccano_to_bio_tags(data)

    # restructure data
    df = pd.DataFrame(data)
    df.rename(columns={"labels": "named_entity_recognition_target", "text": "text_parts"}, inplace=True)

    # make test and cross-validation splits
    df["text_parts"] = df.apply(lambda row: [row["center"]] + list(row["text_parts"]), axis=1)
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
    parser.add_argument("--task_name", type=str, default="Task025_anonymisation_ner",
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
