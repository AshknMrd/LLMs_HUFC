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

import pandas as pd
from tqdm import tqdm


def main(
    federated_data_dir: Path,
    reference_data_dir: Path,
    debug_input_dir: Path,
    debug_test_dir: Path,
):
    for input_task_dir in tqdm(federated_data_dir.glob("*-fold0"), desc="Making debug splits"):
        if not input_task_dir.is_dir():
            continue

        # create output task dir
        output_task_dir = debug_input_dir / input_task_dir.name
        output_task_dir.mkdir(exist_ok=True, parents=True)
        task_name = input_task_dir.name.split("-fold0")[0]

        # read data
        data = []
        with open(input_task_dir / "nlp-training-dataset.json") as f:
            data += json.load(f)
        with open(input_task_dir / "nlp-validation-dataset.json") as f:
            data += json.load(f)
        with open(input_task_dir / "nlp-task-configuration.json") as f:
            task_config = json.load(f)
        task_config["jobid"] += 2000
        df_dev = pd.DataFrame(data)

        # read reference data
        reference_data = []
        with open(reference_data_dir / input_task_dir.name / "nlp-training-dataset.json") as f:
            reference_data += json.load(f)
        with open(reference_data_dir / input_task_dir.name / "nlp-validation-dataset.json") as f:
            reference_data += json.load(f)
        reference_data = pd.DataFrame(reference_data)

        # extract center
        df_dev["center"] = df_dev["text_parts"].apply(lambda parts: parts[0])
        if len(df_dev.iloc[0]["text_parts"]) == 2:
            df_dev["text"] = df_dev["text_parts"].apply(lambda parts: parts[1])
            df_dev = df_dev.drop(columns=["text_parts"])
            task_config["input_name"] = "text"
        else:
            df_dev["text_parts"] = df_dev["text_parts"].apply(lambda parts: parts[1:])

        # verify splits are correct
        for col in df_dev.columns:
            if col == "center":
                continue
            mask = (df_dev[col] != reference_data[col])
            if mask.mean() > 0.05:
                print(f"Warning: Column {col} differs for {mask.mean():.1%} of entries")
            if mask.mean() > 0.1:
                raise ValueError(f"Column {col} differs for {mask.mean():.1%} of entries")

        # copy reports
        df_dev[task_config["input_name"]] = reference_data[task_config["input_name"]]
        df_dev[task_config["label_name"]] = reference_data[task_config["label_name"]]

        # split data
        df_test = df_dev.sample(frac=0.3, random_state=42)
        df_dev = df_dev.drop(df_test.index)
        df_train = df_dev.sample(frac=0.8, random_state=42)
        df_val = df_dev.drop(df_train.index)

        # write data
        df_train.to_json(output_task_dir / "nlp-training-dataset.json", orient="records", indent=2)
        df_val.to_json(output_task_dir / "nlp-validation-dataset.json", orient="records", indent=2)
        df_test[["uid", "center", task_config["input_name"]]].to_json(output_task_dir / "nlp-test-dataset.json", orient="records", indent=2)
        df_test.to_json(debug_test_dir / f"{task_name}.json", orient="records", indent=2)
        with open(output_task_dir / "nlp-task-configuration.json", "w") as f:
            json.dump(task_config, f, indent=2)


if __name__ == "__main__":
    # create the parser
    parser = argparse.ArgumentParser(description="Script for preparing reports")

    parser.add_argument("-i", "--federated-algorithm-input", type=Path, default=Path("/input/federated/algorithm-input"),
                        help="Path to the input data")
    parser.add_argument("-r", "--original-algorithm-input", type=Path, default=Path("/input/algorithm-input"),
                        help="Path to the reference input data")
    parser.add_argument("-o", "--debug-input-dir", type=Path, default=Path("/output/federated/debug-input"),
                        help="Folder to store the prepared reports in")
    parser.add_argument("-t", "--debug-test-dir", type=Path, default=Path("/output/federated/debug-test-set"),
                        help="Folder to store the prepared reports in")
    args = parser.parse_args()

    # run preprocessing
    main(
        federated_data_dir=args.federated_algorithm_input,
        reference_data_dir=args.original_algorithm_input,
        debug_input_dir=args.debug_input_dir,
        debug_test_dir=args.debug_test_dir,
    )
