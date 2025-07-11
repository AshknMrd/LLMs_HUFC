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


def add_words(report: list[str], labels: list[str], sentence: str, label: str) -> tuple[list[str], list[str]]:
    words = sentence.split()
    word_labels = [f"B-{label}"] + [f"I-{label}"] * (len(words) - 1)
    for word, lbl in zip(words, word_labels):
        report.append(word)
        labels.append(lbl)
    return report, labels


def generate_sample(idx: int) -> dict[str, Any]:
    np.random.seed(idx)
    report: list[str] = []
    labels: list[str] = []

    # generate random report pieces
    prefix: str = np.random.choice(PREFIXES)
    symptom: str = np.random.choice(SYMPTOMS)
    diagnosis: str = np.random.choice(DIAGNOSES)
    note: str = np.random.choice(NOTES)

    # add pieces to the report
    for part, lbl in [
        (prefix, "PREFIX"),
        (symptom, "SYMPTOM"),
        (diagnosis, "DIAGNOSIS"),
    ]:
        report, labels = add_words(report, labels, part, lbl)

    # generate random sentences
    for i in range(np.random.randint(5, 15+1)):
        # randomly decide what kind of structure is described
        object_type = np.random.choice(["lesion", "normal", "other"], p=[0.4, 0.4, 0.2])

        # select a random size for the object
        selected_size = np.random.uniform(1, 20)
        selected_size_str = f"{selected_size:.1f}" if selected_size < 10 else f"{selected_size:.0f}"
        unit = np.random.choice(["mm", "cm"], p=[0.8, 0.2])

        # give a roman numeral to the structure
        roman_numeral = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV"][i]

        # add pieces to the report
        report.append(roman_numeral)
        labels.append("B-ROMAN_NUMERAL")

        s = f"{prefix} {object_type} of {selected_size_str} {unit}. "
        for word in s.split():
            report.append(word)
            labels.append("B-STRUCTURE" if word == object_type else "O")

    # add a note
    for part, lbl in [
        (note, "NOTE"),
    ]:
        report, labels = add_words(report, labels, part, lbl)

    # for a few reports, make the report very long
    if idx % 10 == 0:
        long_note = " ".join([np.random.choice(NOTES) for _ in range(100)])
        report, labels = add_words(report, labels, long_note, "NOTE")

    return {"uid": f"Task108_case{idx}", "text_parts": report, "named_entity_recognition_target": labels}


def main(
    output_dir: Path | str = "/output",
    num_examples: int = 100,
    task_name: str = "Task108_Example_sl_ner",
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
    parser.add_argument('--num_examples', type=int, default=100,
                        help="Number of examples to generate.")
    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        num_examples=args.num_examples,
    )
