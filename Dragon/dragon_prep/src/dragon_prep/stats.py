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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer


def main(
    input_dir: Path,
    output_dir: Path,
    model_name: str = "xlm-roberta-base",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for task_dir in tqdm(sorted(list(input_dir.glob("Task0*-fold0")))):
        task_name = task_dir.name.split("-")[0]

        # read data
        data = []
        with open(task_dir / "nlp-training-dataset.json") as f:
            data += json.load(f)
        with open(task_dir / "nlp-validation-dataset.json") as f:
            data += json.load(f)
        with open(task_dir / "nlp-task-configuration.json") as f:
            task_config = json.load(f)

        # tokenize reports and get lengths
        reports = [item[task_config["input_name"]] for item in data]
        lengths = []
        for report in reports:
            if isinstance(report, list):
                # have pre-tokenized text
                lengths.append(sum([len(tokenizer.encode(word, padding=False, truncation=False, return_tensors='np')[0]) - 2 for word in report]))
            else:
                lengths.append(len(tokenizer.encode(report, padding=False, truncation=False, return_tensors='np')[0]))

        # print statistics about report lengths: median (95% CI) [min, max]
        median = np.median(lengths)
        ci_2_5 = np.percentile(lengths, 2.5)
        ci_97_5 = np.percentile(lengths, 97.5)
        min_length = min(lengths)
        max_length = max(lengths)
        tqdm.write(f"{task_name} input length median {median:.0f} (95% CI: {ci_2_5:.0f}, {ci_97_5:.0f}) [min {min_length:.0f}, max {max_length:.0f}] "
                   f"tokens with {model_name} tokenizer (ranges are based on the development data).")

        # plot distribution of report lengths
        f, ax = plt.subplots(figsize=(10, 2.5))
        sns.histplot(lengths, bins=31, ax=ax)
        # ax.set_title(task_name.replace("_clf", "").replace("_reg", "").replace("_ner", "").replace("_", " "))
        ax.set_xlabel(f"Report length (tokens, {model_name} tokenizer)")
        ax.set_ylabel("Number of reports")
        f.tight_layout()
        f.savefig(output_dir / f"report_length_{task_name}.pdf")
        plt.close("all")

        # save report lengths to file
        stats = {
            "median": median,
            "ci_2_5": ci_2_5,
            "ci_97_5": ci_97_5,
            "min_length": min_length,
            "max_length": max_length,
            "num_reports": len(lengths),
        }
        with open(output_dir / f"report_length_{task_name}.json", "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    # create the parser
    parser = argparse.ArgumentParser(description="Script for preparing reports")
    parser.add_argument("-i", "--input", type=Path, default="/Users/joeranbosma/repos/dragon_data/preprocessed/algorithm-input",
                        help="Path to the input data")
    parser.add_argument("-o", "--output", type=Path, default=Path("/Users/joeranbosma/repos/dragon_data/preprocessed/statistics"),
                        help="Folder to store the results in")
    args = parser.parse_args()

    # run preprocessing
    main(
        input_dir=args.input,
        output_dir=args.output,
    )
