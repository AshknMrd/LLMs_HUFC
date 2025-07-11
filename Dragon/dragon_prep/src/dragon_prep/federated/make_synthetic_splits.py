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
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def main(
    input_dir: Path,
    output_dir: Path,
    seed: int = 42,
):
    algorithm_input_dir = input_dir / "algorithm-input"
    test_labeled_dir = input_dir / "test-set"
    for input_task_dir in tqdm(algorithm_input_dir.glob("Task*_Example_*-fold0"), desc="Preparing synthetic data"):
        if not input_task_dir.is_dir():
            continue

        # create output task dir
        output_task_dir = output_dir / "algorithm-input" / input_task_dir.name
        debug_test_dir = output_dir / "test-set"
        task_name = input_task_dir.name.split("-fold0")[0]

        # read data
        with open(input_task_dir / "nlp-training-dataset.json") as f:
            df_train = pd.DataFrame(json.load(f))
        with open(input_task_dir / "nlp-validation-dataset.json") as f:
            df_val = pd.DataFrame(json.load(f))
        with open(input_task_dir / "nlp-test-dataset.json") as f:
            df_test = pd.DataFrame(json.load(f))
        with open(test_labeled_dir / f"{task_name}.json") as f:
            df_test_labeled = pd.DataFrame(json.load(f))
        with open(input_task_dir / "nlp-task-configuration.json") as f:
            task_config = json.load(f)

        if task_config["label_name"] not in [
            "single_label_binary_classification_target",
            "multi_label_binary_classification_target",
            "multi_label_regression_target",
            "named_entity_recognition_target",
        ]:
            print(f"Skipping task {task_name} with label {task_config['label_name']}")
            continue

        # add center
        n_centers = max(2, len(df_train) // 100)
        np.random.seed(seed)
        for df in [df_train, df_val, df_test]:
            tqdm.write(f"Adding {n_centers} centers to {len(df)} entries for task {task_name}")
            df["center"] = np.random.randint(0, n_centers, len(df)).astype(str)

        df_test_labeled["center"] = df_test["center"]

        # write data
        output_task_dir.mkdir(exist_ok=True, parents=True)
        df_train.to_json(output_task_dir / "nlp-training-dataset.json", orient="records", indent=2)
        df_val.to_json(output_task_dir / "nlp-validation-dataset.json", orient="records", indent=2)
        df_test[["uid", "center", task_config["input_name"]]].to_json(output_task_dir / "nlp-test-dataset.json", orient="records", indent=2)
        df_test_labeled.to_json(debug_test_dir / f"{task_name}.json", orient="records", indent=2)
        with open(output_task_dir / "nlp-task-configuration.json", "w") as f:
            json.dump(task_config, f, indent=2)


if __name__ == "__main__":
    # create the parser
    parser = argparse.ArgumentParser(description="Script for preparing reports")
    parser.add_argument("-i", "--input", type=Path, default=Path("/input"),
                        help="Path to the input data")
    parser.add_argument("-o", "--output", type=Path, default=Path("/output"),
                        help="Folder to store the prepared reports in")
    args = parser.parse_args()

    # run preprocessing
    main(
        input_dir=args.input,
        output_dir=args.output,
    )
