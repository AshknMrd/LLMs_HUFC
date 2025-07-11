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
archive_slug = "dragon-debug-train-dataset"

datasets = sorted(list(archive_dir.glob("Task1*_Example_*-fold0")), reverse=True)

if len(datasets) != 9:
    raise Exception(f"Expected 9 example datasets, found {len(datasets)}: {datasets}.")
print(f"Uploading: {datasets}.")

# initialise client
client = gcapi.Client(token=token)

# select archive
archive = client.archives.detail(slug=archive_slug)
archive_url = archive["api_url"]

# create archive items
sessions_save_path = Path("sessions.pkl")
if not sessions_save_path.exists():
    sessions = {}

    for dataset_path in tqdm(datasets, desc="Creating archive items"):
        session = client.archive_items.create(
            values=[],
            archive=archive_url,
        )

        sessions[dataset_path] = session

    with open("sessions.pkl", "wb") as fp:
        import pickle
        pickle.dump(sessions, fp)
else:
    with open("sessions.pkl", "rb") as fp:
        import pickle
        sessions = pickle.load(fp)

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
