import os
from pathlib import Path
from typing import Any

import gcapi
from tqdm import tqdm

"""
Upload archive to Grand Challenge using the gcapi
"""

# paths
archive_dir = Path("/Users/joeranbosma/repos/dragon_data/preprocessed/debug-input")
token = os.getenv("GC_TOKEN")

# initialise client
client = gcapi.Client(token=token)

fold = 0
config = [
    # First, the tasks most likely to time out or fail:
    ("Task014", "debug-dragon-task-014"),  # most samples = most likely to time out
    ("Task028", "debug-dragon-task-028"),  # most difficult = most likely to fail
    # Then each of the remaining task types:
    ("Task001", "debug-dragon-task-001"),
    ("Task009", "debug-dragon-task-009"),
    ("Task015", "debug-dragon-task-015"),
    ("Task018", "debug-dragon-task-018"),
    ("Task023", "debug-dragon-task-023"),
    ("Task024", "debug-dragon-task-024"),
    ("Task025", "debug-dragon-task-025"),
    # Then the rest:
    ("Task002", "debug-dragon-task-002"),
    ("Task003", "debug-dragon-task-003"),
    ("Task004", "debug-dragon-task-004"),
    ("Task005", "debug-dragon-task-005"),
    ("Task006", "debug-dragon-task-006"),
    ("Task007", "debug-dragon-task-007"),
    ("Task008", "debug-dragon-task-008"),
    ("Task010", "debug-dragon-task-010"),
    ("Task011", "debug-dragon-task-011"),
    ("Task012", "debug-dragon-task-012"),
    ("Task013", "debug-dragon-task-013"),
    ("Task016", "debug-dragon-task-016"),
    ("Task017", "debug-dragon-task-017"),
    ("Task019", "debug-dragon-task-019"),
    ("Task020", "debug-dragon-task-020"),
    ("Task021", "debug-dragon-task-021"),
    ("Task022", "debug-dragon-task-022"),
    ("Task026", "debug-dragon-task-026"),
    ("Task027", "debug-dragon-task-027"),
]

sessions = {}
for dataset_name, archive_slug in tqdm(config):
    # select archive
    archive = client.archives.detail(slug=archive_slug)
    archive_url = archive["api_url"]

    print(f"Creating item for: {dataset_name}.")

    # create archive item
    session = client.archive_items.create(
        values=[],
        archive=archive_url,
    )

    sessions[dataset_name] = session

    # upload training resources
    sub_sessions: dict[Path, dict[str, Any]] = {}

    # upload elements to archive item
    dataset_dir = archive_dir.glob(f"{dataset_name}*-fold{fold}").__next__()
    for interface_name in [
        "nlp-training-dataset",
        "nlp-validation-dataset",
        "nlp-test-dataset",
        "nlp-task-configuration",
    ]:
        # construct path to file
        path = dataset_dir / f"{interface_name}.json"
        assert path.exists()

        print(f"Uploading {path}..")
        sub_session = client.update_archive_item(
            archive_item_pk=session['pk'],
            values={
                interface_name: [path],
            },
        )

        if dataset_dir not in sub_sessions:
            sub_sessions[dataset_dir] = {}
        sub_sessions[dataset_dir][interface_name] = sub_session
