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

from dragon_prep.synthetic_data_utils import (DIAGNOSES, NOTES, PREFIXES,
                                              SYMPTOMS)
from dragon_prep.utils import split_and_save_data


def generate_sample(idx: int, max_num_lesions: int = 10) -> dict[str, Any]:
    np.random.seed(idx)
    report: list[str] = []
    labels: list[list[str]] = []

    # generate random report pieces
    prefix: str = np.random.choice(PREFIXES)
    symptom: str = np.random.choice(SYMPTOMS)
    diagnosis: str = np.random.choice(DIAGNOSES)
    note: str = np.random.choice(NOTES)

    # add pieces to the report
    for s in [prefix, symptom, diagnosis]:
        for word in s.split():
            report.append(word)
            labels.append([])

    # generate random sentences
    structure_counter = 0
    for i in range(np.random.randint(1, 5+1)):
        # randomly decide what kind of structure is described
        object_type = np.random.choice(["lesion", "normal", "other"], p=[0.4, 0.4, 0.2])

        # select a random size for the object
        selected_size = np.random.uniform(1, 20)
        selected_size_str = f"{selected_size:.1f}" if selected_size < 10 else f"{selected_size:.0f}"
        unit = np.random.choice(["mm", "cm"], p=[0.8, 0.2])

        # give a roman numeral to the structure, which could be multiple structures
        num_structures = np.random.randint(1, 3+1)
        roman_numerals = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV"][
            structure_counter:structure_counter+num_structures
        ]

        # restrict the number of lesions in a report to `max_num_lesions`
        if object_type == "lesion" and structure_counter >= max_num_lesions:
            object_type = "normal"
        elif object_type == "lesion" and (structure_counter + num_structures) >= max_num_lesions:
            num_structures = max_num_lesions - structure_counter

        # add pieces to the report
        report.append("+".join(roman_numerals))
        labels.append([])

        s = f"{prefix} {object_type} of"
        for word in s.split():
            report.append(word)
            labels.append([])

        # add measurement to the report
        s = f"{selected_size_str} {unit}"
        for i, word in enumerate(s.split()):
            if object_type == "lesion":
                report.append(word)
                labels.append([f"{'B' if i == 0 else 'I'}-{idx}-lesion" for idx in range(structure_counter, structure_counter + num_structures)])
            else:
                report.append(word)
                labels.append([])

        # end the sentence
        report.append(".")
        labels.append([])

        # update the structure counter
        structure_counter += num_structures

    # add a note
    for word in note.split():
        report.append(word)
        labels.append([])

    # for a few reports, make the report very long
    if idx % 10 == 0:
        long_note = " ".join([np.random.choice(NOTES) for _ in range(100)])
        for word in long_note.split():
            report.append(word)
            labels.append([])

    # add "O" labels for all words that are not part of a named entity
    labels = [label if label else ["O"] for label in labels]

    return {"uid": f"Task109_case{idx}", "text_parts": report, "multi_label_named_entity_recognition_target": labels}


def main(
    output_dir: Path | str = "/output",
    num_examples: int = 500,
    task_name: str = "Task109_Example_ml_ner",
) -> None:
    """
    Generate an example multi-label regression dataset for NLP
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
        recommended_truncation_side="right",
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_dir', type=str, default="/output",
                        help="Directory to store data.")
    parser.add_argument('--num_examples', type=int, default=500,
                        help="Number of examples to generate.")
    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        num_examples=args.num_examples,
    )
