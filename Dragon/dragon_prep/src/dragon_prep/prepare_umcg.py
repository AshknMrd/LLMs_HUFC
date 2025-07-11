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

from pathlib import Path

import pandas as pd

from dragon_prep.utils import apply_anon_annotations, read_marksheet


def prepare_umcg_radiology_reports(input_dir: Path) -> pd.DataFrame:
    # read UMCG marksheet
    df = read_marksheet(input_dir / "radiologie_rapporten_JB_mapped.csv")

    # rename columns
    df.rename({
        "RadiologyReport": "text",
        "psa_density": "psad",
        "volume": "prostate_volume",
    }, axis="columns", inplace=True)

    df["subject_id"] = df.apply(lambda row: f"{row['patient_id']}_{row['study_id']}", axis=1)
    df["label"] = None  # placeholder

    # exclude non-detection studies: visit0395406
    df = df[df.study_id != "visit0395406"]

    # parse dataset
    umcg_entries = []
    for subject_id in df["subject_id"].unique():
        study_df = df[df["subject_id"] == subject_id]
        study_df = study_df.drop_duplicates(subset=["roiID"])

        # parse and count PI-RADS >= 3 score
        scores = [
            int(score) for score in study_df.pirads
            if score != "NA"
        ]

        umcg_entries.append({
            "num_pirads_345": sum([score >= 3 for score in scores]),
            "lesion_PIRADS": ",".join(study_df.pirads.values),
            **study_df.iloc[0].to_dict(),
        })

    df = pd.DataFrame(umcg_entries)
    df["uid"] = df["subject_id"]

    df_reports = pd.read_json(input_dir / "all_checked.jsonl", lines=True)
    df_reports = df_reports.apply(apply_anon_annotations, axis=1)
    df_reports["uid"] = df_reports.meta.apply(lambda meta: meta["uid"])
    df_reports = df_reports.set_index("uid")

    # merge with radiology reports
    df["text"] = df["uid"].map(df_reports.text.to_dict())

    # clean up notation
    df["lesion_PIRADS"] = df["lesion_PIRADS"].str.replace("NA", "N/A")

    return df


def prepare_umcg_pathology_reports(input_dir: Path) -> pd.DataFrame:
    df = read_marksheet(input_dir / "pathologie_rapporten_JB_mapped.csv")

    # rename columns
    df.rename({
        "PathologyReport": "text",
    }, axis="columns", inplace=True)

    df["subject_id"] = df.apply(lambda row: f"{row['patient_id']}_{row['study_id']}", axis=1)

    # verify dataset
    assert len(set(df.text.values)) == len(set(df["subject_id"])) == 227
    assert len(set(df.patient_id.values)) == 220

    # load reports with PHI tags
    df_reports = pd.read_json(input_dir / "all.jsonl", lines=True)
    df_reports = df_reports.apply(apply_anon_annotations, axis=1)
    df_reports["subject_id"] = df_reports.meta.apply(lambda meta: meta["subject_id"])
    df_reports = df_reports.set_index("subject_id")

    # merge with pathology reports
    df["text"] = df["subject_id"].map(df_reports.text.to_dict())

    return df
