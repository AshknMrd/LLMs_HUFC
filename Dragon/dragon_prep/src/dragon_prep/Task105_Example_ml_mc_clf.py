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
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from dragon_prep.synthetic_data_utils import NOTES
from dragon_prep.utils import split_and_save_data

# Diagnostic Report components
symptoms = ["fever", "cough", "fatigue", "shortness of breath", "headache", "dizziness", "chest pain", "nausea", "joint pain"]
findings = ["inflamed tonsils", "chest congestion", "swollen lymph nodes", "irregular heart rate", "high blood pressure"]

# Disease diagnosis based on symptoms and findings
diagnosis_criteria = {
    "Flu": ["fever", "cough", "inflamed tonsils"],
    "COVID-19": ["fever", "cough", "shortness of breath", "chest congestion"],
    "Common Cold": ["cough", "inflamed tonsils"],
    "Pneumonia": ["fever", "chest congestion"],
    "Heart Disease": ["dizziness", "chest pain", "irregular heart rate"],
    "Migraine": ["headache", "dizziness", "nausea"],
    "Arthritis": ["joint pain"],
}

# Treatment recommendation based on findings
treatment_mapping = {
    "inflamed tonsils": "antibiotics",
    "high blood pressure": "beta blockers",
    "headache": "pain killers",
    "chest pain": "pain killers",
}


def generate_sample(idx: int) -> dict[str, Any]:
    np.random.seed(idx)
    report = ""

    # generate random report pieces
    num_symptoms = np.random.randint(1, 4+1)
    selected_symptoms = np.random.choice(symptoms, size=num_symptoms, replace=False)
    selected_finding = np.random.choice(findings, size=2, replace=False)
    report = f"Patient reports {', '.join(selected_symptoms)}. On examination, {selected_finding[0]} and {selected_finding[1]} was noted."

    # Disease diagnosis based on symptoms and findings
    diagnosed_disease = "Unknown"
    for disease, criteria in diagnosis_criteria.items():
        if all(x in report for x in criteria):
            diagnosed_disease = disease
            break

    # Treatment recommendation based on symptoms and findings
    treatment = "Unknown"
    for finding in list(selected_finding) + list(selected_symptoms):
        if finding in treatment_mapping:
            treatment = treatment_mapping[finding]
            break

    # for a few reports, make the report very long
    if idx % 25 == 0:
        report += " ".join([np.random.choice(NOTES) for _ in range(100)])

    return {"uid": f"Task105_case{idx}", "text": report, "multi_label_multi_class_classification_target": [diagnosed_disease, treatment]}


def main(
    output_dir: Path | str = "/output",
    num_examples: int = 750,
    task_name: str = "Task105_Example_ml_mc_clf",
) -> None:
    """
    Generate an example classification dataset for NLP
    """
    # generate the data
    data = []
    for idx in tqdm(range(num_examples), desc="Generating data"):
        data.append(generate_sample(idx))
    df = pd.DataFrame(data)

    # make test and cross-validation splits
    split_and_save_data(
        df=df,
        task_name=task_name,
        output_dir=output_dir,
        split_by="uid",
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_dir', type=str, default="/output",
                        help="Directory to store data.")
    parser.add_argument('--num_examples', type=int, default=750,
                        help="Number of examples to generate.")
    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        num_examples=args.num_examples,
    )
