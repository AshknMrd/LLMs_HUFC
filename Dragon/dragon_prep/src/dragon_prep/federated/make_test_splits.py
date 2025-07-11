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
    federated_test_dir: Path,
    reference_test_dir: Path,
):
    files = sorted(federated_data_dir.glob("*-fold*"))
    for input_task_dir in tqdm(files, desc="Making debug splits"):
        if not input_task_dir.is_dir():
            continue

        task_name = input_task_dir.name.split("-fold")[0]

        # read data
        with open(input_task_dir / "nlp-training-dataset.json") as f:
            df_train = pd.DataFrame(json.load(f))
        with open(input_task_dir / "nlp-validation-dataset.json") as f:
            df_val = pd.DataFrame(json.load(f))
        with open(input_task_dir / "nlp-test-dataset.json") as f:
            df_test = pd.DataFrame(json.load(f))
        with open(federated_test_dir / f"{task_name}.json") as f:
            df_test_labeled = pd.DataFrame(json.load(f))
        with open(input_task_dir / "nlp-task-configuration.json") as f:
            task_config = json.load(f)

        # read reference data
        with open(reference_data_dir / input_task_dir.name / "nlp-training-dataset.json") as f:
            df_train_reference = pd.DataFrame(json.load(f))
        with open(reference_data_dir / input_task_dir.name / "nlp-validation-dataset.json") as f:
            df_validation_reference = pd.DataFrame(json.load(f))
        with open(reference_data_dir / input_task_dir.name / "nlp-test-dataset.json") as f:
            df_test_reference = pd.DataFrame(json.load(f))
        with open(reference_test_dir / f"{task_name}.json") as f:
            df_test_labelled_reference = pd.DataFrame(json.load(f))

        # extract center
        for df, df_ref in [
            (df_train, df_train_reference),
            (df_val, df_validation_reference),
            (df_test, df_test_reference),
            (df_test_labeled, df_test_labelled_reference),
        ]:
            df["center"] = df["text_parts"].apply(lambda parts: parts[0])
            if len(df.iloc[0]["text_parts"]) == 2:
                df["text"] = df["text_parts"].apply(lambda parts: parts[1])
                df.drop(columns=["text_parts"], inplace=True)
                task_config["input_name"] = "text"
            else:
                df["text_parts"] = df["text_parts"].apply(lambda parts: parts[1:])

            # verify splits are correct
            for col in df.columns:
                if col == "center":
                    continue
                mask = (df[col] != df_ref[col])
                if mask.mean() > 0.05:
                    print(f"Warning: Column {col} differs for {mask.mean():.1%} of entries")
                if mask.mean() > 0.1:
                    raise ValueError(f"Column {col} differs for {mask.mean():.1%} of entries")

            # copy reports
            df[task_config["input_name"]] = df_ref[task_config["input_name"]]
            if task_config["label_name"] in df.columns:
                df[task_config["label_name"]] = df_ref[task_config["label_name"]]

        # write data
        df_train.to_json(input_task_dir / "nlp-training-dataset.json", orient="records", indent=2)
        df_val.to_json(input_task_dir / "nlp-validation-dataset.json", orient="records", indent=2)
        df_test.to_json(input_task_dir / "nlp-test-dataset.json", orient="records", indent=2)
        if "-fold4" in input_task_dir.name:
            df_test_labeled.to_json(federated_test_dir / f"{task_name}.json", orient="records", indent=2)
        with open(input_task_dir / "nlp-task-configuration.json", "w") as f:
            json.dump(task_config, f, indent=2)


if __name__ == "__main__":
    # create the parser
    parser = argparse.ArgumentParser(description="Script for preparing reports")
    parser.add_argument("-i", "--federated-algorithm-input", type=Path, default=Path("/input/federated/algorithm-input"),
                        help="Path to the input data")
    parser.add_argument("-r", "--original-algorithm-input", type=Path, default=Path("/input/algorithm-input"),
                        help="Path to the reference input data")
    parser.add_argument("-t", "--federated-test-dir", type=Path, default=Path("/input/federated/test-set"),
                        help="Path to the test data")
    parser.add_argument("-d", "--original-test-dir", type=Path, default=Path("/input/test-set"),
                        help="Path to the reference test data")
    args = parser.parse_args()

    # run preprocessing
    main(
        federated_data_dir=args.federated_algorithm_input,
        reference_data_dir=args.original_algorithm_input,
        federated_test_dir=args.federated_test_dir,
        reference_test_dir=args.original_test_dir,
    )
