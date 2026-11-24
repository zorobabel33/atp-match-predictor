"""Fetch the actual champion's DVC training release and backfill its missing reference artifact."""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from ml.data import dvc_hash, fetch, validate_release
from ml.model import training_profile
from ml.registry import configure_tracking, get_champion, load_training_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backfill-reference", action="store_true",
                        help="add a missing artifact to the original approval run; never retrain")
    args = parser.parse_args()
    client, name = configure_tracking()
    champion = get_champion(client, name)
    if champion is None:
        raise ValueError("No approved champion")
    version = champion.tags["dataset_version"]
    if (champion.tags.get("data_source") != "dvc"
            or champion.tags.get("dvc_md5") != dvc_hash(version)):
        raise ValueError("Champion must match the pinned DVC release exactly")
    paths = fetch(version)
    train, _ = validate_release(paths)
    output = Path(".exercise-data") / "real" / version
    output.mkdir(parents=True, exist_ok=True)
    for key, path in paths.items():
        shutil.copyfile(path, output / f"{key}.csv")
    print(f"Fetched the verified {version} training release for {name} version {champion.version}")
    if not args.backfill_reference:
        return
    approval = champion.tags.get("approval_run_id")
    if not approval:
        raise ValueError("Missing original approval run; do not invent reference provenance")
    if any(p.path == "training-profile.json" for p in client.list_artifacts(approval)):
        load_training_profile(client, champion)
        print("The matching reference already exists; left unchanged")
        return
    with TemporaryDirectory() as folder:
        path = Path(folder) / "training-profile.json"
        path.write_text(json.dumps(training_profile(
            train, model_run_id=champion.run_id, dataset_version=version,
            dvc_md5=dvc_hash(version), data_source="dvc",
        )))
        client.log_artifact(approval, str(path))
    load_training_profile(client, champion)
    print("Backfilled and verified the original approval-run reference; champion alias unchanged")


if __name__ == "__main__":
    main()
