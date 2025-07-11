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
from dragon_prep.utils import (num_patients, prepare_for_anon, read_anon,
                               split_and_save_data)

try:
    from report_anonymizer.model.anonymizer_functions import Anonymizer
except ImportError:
    print("Anonymizer not found, will not be able to run the full pipeline.")


def combine_phi_labels(label: str) -> str:
    # combine ill-differentiated labels
    label = label.replace(
        "<TELEFOONNUMMER>", "<PHINUMMER>").replace(
        "<ZIEKENHUIS>", "<PLAATS>").replace(
        "<PATIENTNUMMER>", "<PHINUMMER>").replace(
        "<ZNUMMER>", "<PHINUMMER>").replace(
        "<PERSOONAFKORTING>", "<PERSOON>"
        )

    # combine similar labels
    label = re.sub(r"<RAPPORT[_-]ID\.(T|R|C|DPA|RPA)[_-]NUMMER>", "<RAPPORT_ID>", label)

    # rename
    label = label.replace("<STUDIE-NAAM>", "<STUDIE_NAAM>")

    return label


def preprocess_reports_rumc_thorax_abdomen_ct(
    input_dir: Path,
) -> pd.DataFrame:
    # read reports and labels
    # the labels are in the format [[start1, end1, label1], [start2, end2, label2], ...], for example [[75, 86, "<DATUM>"]]
    df = pd.read_json(input_dir / "anon_ground_truth_v11.jsonl", lines=True)

    # prepare metadata
    df["PatientID"] = df.meta.apply(lambda x: x["PatientID"])
    df["StudyInstanceUID"] = df.meta.apply(lambda x: x["StudyInstanceUID"])
    df["uid"] = df.apply(lambda row: row["PatientID"] + "_" + row["StudyInstanceUID"], axis=1)
    df = df.rename(columns={"labels": "label"})
    print(f"Loaded {len(df)} reports ({num_patients(df)} patients) for RUMC CT thorax abdomen")

    # exclude duplicate reports
    df = df.drop_duplicates(subset=["text"])
    print(f"Have {len(df)} reports ({num_patients(df)} patients) after excluding duplicates")

    return df


def preprocess_reports_rumc_prostate_pathology(
    input_dir: Path,
) -> pd.DataFrame:
    # read reports and labels
    # the labels are in the format [[start1, end1, label1], [start2, end2, label2], ...], for example [[75, 86, "<DATUM>"]]
    df = pd.read_json(input_dir / "all_checked.jsonl", lines=True)

    # prepare metadata
    df["PatientID"] = df.meta.apply(lambda x: x["MDN:"])
    df["StudyID"] = df.meta.apply(lambda x: x["PA-NR.:"])
    df["uid"] = df.apply(lambda row: f"{row['PatientID']}_{row['StudyID']}", axis=1)
    print(f"Loaded {len(df)} reports ({num_patients(df)} patients) for RUMC prostate pathology")

    # exclude duplicate reports
    df = df.drop_duplicates(subset=["text"])
    print(f"Have {len(df)} reports ({num_patients(df)} patients) after excluding duplicates")

    return df


def preprocess_reports_rumc_bcc_pathology(
    input_dir: Path,
) -> pd.DataFrame:
    # read reports and labels (the PHI labels are in the "label" field, the reports are in the "text" field
    # and the task labels are in the meta->label field)
    # the labels are in the format [[start1, end1, label1], [start2, end2, label2], ...], for example [[75, 86, "<LABEL>"]]
    df = pd.read_json(input_dir / "all_checked_with_patient_ids_anon_annot_manual_fixed.jsonl", lines=True)

    # select both sets of labels
    df = df.rename(columns={"label": "label_phi"})
    df["label_task"] = df.meta.apply(lambda meta: meta["label"])

    # extract the metadata
    for key in df["meta"].iloc[0].keys():
        df[key] = df["meta"].apply(lambda x: x[key])
    df = df.drop(["meta", "label"], axis=1)

    # set the unique case identifier
    df["uid"] = df.apply(lambda row: f"{row['pid']}_{row['id']}", axis=1)
    print(f"Loaded {len(df)} reports ({num_patients(df)} patients) for RUMC BCC pathology")

    # exclude duplicate reports
    df = df.drop_duplicates(subset=["text"])
    print(f"Have {len(df)} reports ({num_patients(df)} patients) after excluding duplicates")

    df = df.rename(columns={"label_phi": "label", "pid": "PatientID"})

    return df


def preprocess_reports_rumc_prostate_biopsy_procedure(
    input_dir: Path,
) -> pd.DataFrame:
    # read reports and labels (the PHI labels are in the "label" field, the reports are in the "text" field
    # and the task labels are in the meta->label field)
    # the labels are in the format [[start1, end1, label1], [start2, end2, label2], ...], for example [[75, 86, "<LABEL>"]]
    df = pd.read_json(input_dir / "all_checked_with_phi_annot.jsonl", lines=True)

    # select both sets of labels
    df = df.rename(columns={"label": "label_phi"})
    df["label_task"] = df.meta.apply(lambda meta: meta["label"])

    # extract the metadata
    for key in df["meta"].iloc[0].keys():
        df[key] = df["meta"].apply(lambda x: x[key])
    df = df.drop(["meta", "label"], axis=1)

    print(f"Loaded {len(df)} reports ({num_patients(df)} patients) for RUMC prostate biopsy")

    # exclude duplicate reports
    df = df.drop_duplicates(subset=["text"])
    print(f"Have {len(df)} reports ({num_patients(df)} patients) after excluding duplicates")

    df = df.rename(columns={"label_phi": "label"})

    return df


def preprocess_reports_rumc_lung_pathology(
    input_dir: Path,
) -> pd.DataFrame:
    # read reports and labels
    # the labels are in the format [[start1, end1, label1], [start2, end2, label2], ...], for example [[75, 86, "<DATUM>"]]
    df = pd.read_json(input_dir / "all_checked.jsonl", lines=True)
    anon_id_mapping = pd.read_csv(input_dir / "tnumber_to_anon_pid.csv")

    # add patient ID to the dataframe
    df["PatientID"] = None
    for i, row in df.iterrows():
        patient_df = anon_id_mapping[anon_id_mapping["C/T-number"] == row["uid"]]
        if len(patient_df) != 1:
            raise ValueError("Have inexact match for patient ID.")
        patient_id = patient_df.iloc[0]["Anonymous patient ID"]
        df.at[i, "PatientID"] = patient_id

    print(f"Loaded {len(df)} reports ({num_patients(df)} patients) for RUMC lung pathology")

    # drop unnecessary columns
    df = df.drop(["meta", "Comments", "id"], axis=1)

    # exclude duplicate reports
    df = df.drop_duplicates(subset=["text"])
    print(f"Have {len(df)} reports ({num_patients(df)} patients) after excluding duplicates")

    return df


def preprocess_reports_avl(
    input_dir: Path,
) -> pd.DataFrame:
    # read reports and labels from AvL
    # the labels are in the format [[start1, end1, label1], [start2, end2, label2], ...], for example [[75, 86, "<DATUM>"]]
    df_path = pd.read_json(input_dir / "pathology/data_pathology_mapped.jsonl", lines=True)
    df_rad_consecutive = pd.read_json(input_dir / "radiology/data_radiology_consecutive_mapped.jsonl", lines=True)
    df_rad_random = pd.read_json(input_dir / "radiology/data_radiology_random_mapped.jsonl", lines=True)

    # print statistics
    print(f"Loaded {len(df_path)} pathology reports ({num_patients(df_path)} patients) for AvL")
    print(f"Loaded {len(df_rad_consecutive)} consecutive radiology reports ({num_patients(df_rad_consecutive)} patients) for AvL")
    print(f"Loaded {len(df_rad_random)} random radiology reports ({num_patients(df_rad_random)} patients) for AvL")

    # combine the datasets
    cols = ["uid", "PatientID", "text", "label"]
    df_rad = pd.concat([df_rad_consecutive[cols], df_rad_random[cols]], ignore_index=True)
    print(f"Have {len(df_rad)} radiology reports ({num_patients(df_rad)} patients) for AvL")

    df_rad = df_rad.drop_duplicates(subset=["uid"])
    print(f"Have {len(df_rad)} radiology reports ({num_patients(df_rad)} patients) after excluding duplicate studies for AvL")

    df_rad = df_rad.drop_duplicates(subset=["text"])
    print(f"Have {len(df_rad)} radiology reports ({num_patients(df_rad)} patients) after excluding duplicates for AvL")

    df = pd.concat([df_path[cols], df_rad[cols]], ignore_index=True)
    print(f"Have {len(df)} reports ({num_patients(df)} patients) for AvL")

    # drop duplicates (due to both consecutive and random radiology reports being present in the dataset, which overlap for some cases)
    assert df["uid"].is_unique, "UIDs are not unique in the dataset for AvL"
    assert df["text"].is_unique, "Texts are not unique in the dataset for AvL"

    # exclude excluded reports
    df = df[~df.label.apply(lambda x: any([label[2] in ["exclude"] for label in x]))]

    # select labels for anonymization task
    excluded_labels = ["lesion 1", "lesion 2", "lesion 3", "lesion 4", "2 lesions", "3 lesions", "4+ lesions"]
    df["label"] = df.label.apply(lambda x: [label for label in x if label[2] not in excluded_labels])

    return df


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

    # read reports and labels from AvL
    df_avl = preprocess_reports_avl(
        input_dir=input_dir / "avl/prostate",
    )

    # combine the datasets
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
    parser.add_argument("--input", type=Path, default=Path("/input"),
                        help="Path to the input data")
    parser.add_argument("--task_name", type=str, default="Task025_anonymisation_ner",
                        help="Name of the task")
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
