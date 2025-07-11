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
                                              SYMPTOMS, add_noise)
from dragon_prep.utils import split_and_save_data


def generate_sample(idx: int, noise: bool = False) -> dict[str, Any]:
    np.random.seed(idx)
    label = {
        idx: np.nan
        for idx in range(1, 5+1)
    }
    current_lesion_idx = 1
    report = ""

    # generate random report pieces
    prefix = np.random.choice(PREFIXES)
    symptom = np.random.choice(SYMPTOMS)
    diagnosis = np.random.choice(DIAGNOSES)
    note = np.random.choice(NOTES)

    # add pieces to the report
    report += f"{prefix} {symptom}, {diagnosis}. "

    # generate random sentences
    for i in range(np.random.randint(1, 5+1)):
        # randomly decide what kind of structure is described
        object_type = np.random.choice(["lesion", "normal", "other"], p=[0.4, 0.4, 0.2])

        # select a random size for the object
        selected_size = np.random.uniform(1, 20)
        selected_size_str = f"{selected_size:.1f}" if selected_size < 10 else f"{selected_size:.0f}"
        unit = np.random.choice(["mm", "cm"], p=[0.8, 0.2])
        physical_size = selected_size * 10 if unit == "cm" else selected_size

        # give a roman numeral to the structure
        roman_numeral = ["I", "II", "III", "IV", "V"][i]

        # generate a sentence
        prefix = roman_numeral + ": " + np.random.choice(PREFIXES)
        maybe_space = " " if np.random.rand() < 0.5 else ""
        if object_type == "lesion":
            report += f"{prefix} lesion of {selected_size_str}{maybe_space}{unit}. "
        elif object_type == "normal":
            report += f"{prefix} normal of {selected_size_str}{maybe_space}{unit}. "
        elif object_type == "other":
            report += f"{prefix} other structure of size {selected_size_str}{maybe_space}{unit}. "
        else:
            raise ValueError(f"Unknown object type {object_type}.")

        # update the label
        label[current_lesion_idx] = round(physical_size, 0)
        current_lesion_idx += 1

    # add a note
    report += f"{note} "

    # for a few reports, make the report very long
    if idx % 25 == 0:
        report += " ".join([np.random.choice(NOTES) for _ in range(100)])

    if noise:
        # introduce noise:
        # - randomly remove some words (p = 0.025)
        # - randomly swap some words (p = 0.025)
        # - reconstruct report while randomly removing some spaces (p = 0.025)
        report = add_noise(report)

    return {"uid": f"Task107_case{idx}", "text": report, "multi_label_regression_target": list(label.values())}


def main(
    output_dir: Path | str = "/output",
    num_examples: int = 1500,
    task_name: str = "Task107_Example_ml_reg",
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
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_dir', type=str, default="/output",
                        help="Directory to store data.")
    parser.add_argument('--num_examples', type=int, default=1500,
                        help="Number of examples to generate.")
    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        num_examples=args.num_examples,
    )
