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

import numpy as np
import pandas as pd

from dragon_prep.prepare_avl import prepare_avl_radiology_reports
from dragon_prep.prepare_rumc import prepare_rumc_prostate_radiology_reports
from dragon_prep.prepare_umcg import prepare_umcg_radiology_reports
from dragon_prep.utils import (num_patients, parse_scores, prepare_for_anon,
                               read_anon, split_and_save_data)


def preprocess_reports_rumc(
    input_dir: Path,
) -> pd.DataFrame:
    # read PI-CAI marksheet
    df = prepare_rumc_prostate_radiology_reports(input_dir)

    # compute the number of PI-RADS >= 3 lesions
    df["label"] = df.lesion_PIRADS.apply(lambda scores: sum(np.array(parse_scores(scores)) >= 3))

    return df


def preprocess_reports_umcg(
    input_dir: Path,
) -> pd.DataFrame:
    # read UMCG marksheet
    df = prepare_umcg_radiology_reports(input_dir)
    df["label"] = df["num_pirads_345"]

    return df


def preprocess_reports_avl(
    input_dir: Path,
) -> pd.DataFrame:
    # read AVL marksheet
    df = prepare_avl_radiology_reports(input_dir)

    # prepare labels
    df = df[df.pirads_lesions != "nr"]  # exclude cases without PI-RADS scores ("not reported")
    print(f"Have {len(df)} radiology reports ({num_patients(df)} patients) with PI-RADS scores for AVL")

    # exclude cases with ambiguous PI-RADS scores
    mask = (
        (df["joeran_checken"] == "Exclude for PI-RADS ambiguity")
        | (df["pirads_lesions"] == "wtf")
        | (df["pirads_lesions"].apply(lambda scores: isinstance(scores, str) and ("/" in scores or "???" in scores or "tot" in scores or "-" in scores)))
    )
    print(f"Excluding {mask.sum()} cases ({num_patients(df[mask])}) with ambiguous PI-RADS scores")
    df = df[~mask]
    print(f"Have {len(df)} radiology reports ({num_patients(df)} patients) after excluding")

    # compute the number of PI-RADS >= 3 lesions
    df["label"] = df["pirads_lesions"].apply(lambda scores: sum(np.array(parse_scores(scores)) >= 3))

    return df


def preprocess_reports(
    task_name: str,
    input_dir: Path,
    output_dir: Path,
):
    # read PI-CAI marksheet for RUMC reports
    df_rumc = preprocess_reports_rumc(input_dir / "rumc/prostate")
    print(f"Have {len(df_rumc)} radiology reports ({num_patients(df_rumc)} patients) from RUMC")

    # # read UMCG marksheet for UMCG reports
    df_umcg = preprocess_reports_umcg(input_dir / "umcg/prostate/radiology")
    print(f"Have {len(df_umcg)} radiology reports ({num_patients(df_umcg)} patients) from UMCG")

    # read AVL marksheet for AVL reports
    df_avl = preprocess_reports_avl(input_dir / "avl/prostate/radiology")
    print(f"Have {len(df_avl)} radiology reports ({num_patients(df_avl)} patients) from AVL")

    # merge dataframes
    cols = ["uid", "patient_id", "study_id", "label", "text"]
    assert set(df_rumc.patient_id) & set(df_umcg.patient_id) == set()
    assert set(df_rumc.patient_id) & set(df_avl.patient_id) == set()
    assert set(df_umcg.patient_id) & set(df_avl.patient_id) == set()
    assert df_rumc.text.apply(lambda text: "<PERSOON>" in text).sum() > 1200, f"Unexpected number of <PERSOON> tags in RUMC reports: {df_rumc.text.apply(lambda text: '<PERSOON>' in text).sum()}"
    assert df_rumc.text.apply(lambda text: "<DATUM>" in text).sum() > 1200, f"Unexpected number of <DATUM> tags in RUMC reports: {df_rumc.text.apply(lambda text: '<DATUM>' in text).sum()}"
    assert df_umcg.text.apply(lambda text: "<DATUM>" in text).sum() > 150, f"Unexpected number of <DATUM> tags in UMCG reports: {df_umcg.text.apply(lambda text: '<DATUM>' in text).sum()}"
    assert df_umcg.text.apply(lambda text: "<TIJD>" in text).sum() > 150, f"Unexpected number of <TIJD> tags in UMCG reports: {df_umcg.text.apply(lambda text: '<TIJD>' in text).sum()}"
    assert df_avl.text.apply(lambda text: "<PERSOON>" in text).sum() > 200, f"Unexpected number of <PERSOON> tags in AVL reports: {df_avl.text.apply(lambda text: '<PERSOON>' in text).sum()}"
    assert df_avl.text.apply(lambda text: "<DATUM>" in text).sum() > 200, f"Unexpected number of <DATUM> tags in AVL reports: {df_avl.text.apply(lambda text: '<DATUM>' in text).sum()}"
    df = pd.concat((df_rumc[cols], df_umcg[cols], df_avl[cols]), ignore_index=True)
    print(f"Have {len(df)} radiology reports ({num_patients(df)} patients) in total")

    # prepare labels
    df["label"] = df["label"].astype(str)
    df = df.rename(columns={"label": "single_label_multi_class_classification_target"})

    # prepare for anonynimization
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
    parser.add_argument("--task_name", type=str, default="Task010_prostate_radiology_clf",
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
