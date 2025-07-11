import os
from pathlib import Path
from typing import Any

import gcapi
from tqdm import tqdm

"""
Upload archive to Grand Challenge using the gcapi
First, create an archive: https://grand-challenge.org/archives/create/
"""

# paths
archive_dir = Path("/Users/joeranbosma/repos/dragon_data/preprocessed/algorithm-input")
token = os.getenv("GC_TOKEN")

# name of the archive
archive_slug = "dragon-test-dataset"

# initialise client
client = gcapi.Client(token=token)

# select archive
archive = client.archives.detail(slug=archive_slug)
archive_url = archive["api_url"]

for fold in range(5):
    dataset_names = [
        # First, the tasks most likely to time out or fail:
        "Task014",  # most samples = most likely to time out
        "Task028",  # most difficult = most likely to fail
        # Then each of the remaining task types:
        "Task001",
        "Task009",
        "Task015",
        "Task018",
        "Task023",
        "Task024",
        "Task025",
        # Then the rest:
        "Task002",
        "Task003",
        "Task004",
        "Task005",
        "Task006",
        "Task007",
        "Task008",
        "Task010",
        "Task011",
        "Task012",
        "Task013",
        "Task016",
        "Task017",
        "Task019",
        "Task020",
        "Task021",
        "Task022",
        "Task026",
        "Task027",
    ]
    datasets = [archive_dir.glob(f"{dataset}*-fold{fold}").__next__() for dataset in dataset_names]

    if len(datasets) != 28:
        raise Exception(f"Expected 28 datasets, found {len(datasets)}: {datasets}.")
    for path in datasets:
        if not path.exists():
            raise Exception(f"Path does not exist: {path}.")
    print(f"Uploading: {datasets}.")

    # create archive items
    sessions = {}

    for dataset_path in tqdm(datasets, desc="Creating archive items"):
        session = client.archive_items.create(
            values=[],
            archive=archive_url,
        )

        sessions[dataset_path] = session

    # upload training resources to each archive item
    sub_sessions: dict[Path, dict[str, Any]] = {}

    for dataset_path, session in tqdm(zip(datasets, sessions.values()), desc="Uploading archive items"):
        # upload elements to archive item
        for interface_name in [
            "nlp-training-dataset",
            "nlp-validation-dataset",
            "nlp-test-dataset",
            "nlp-task-configuration",
        ]:
            # construct path to file
            path = dataset_path / f"{interface_name}.json"
            assert path.exists()

            print(f"Uploading {path}..")
            sub_session = client.update_archive_item(
                archive_item_pk=session['pk'],
                values={
                    interface_name: [path],
                },
            )

            if dataset_path not in sub_sessions:
                sub_sessions[dataset_path] = {}
            sub_sessions[dataset_path][interface_name] = sub_session
