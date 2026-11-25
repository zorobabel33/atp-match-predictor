"""Prepare isolated, synthetic tennis traffic and two approved local monitoring models."""

import argparse
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

from ml.model import features, train_model, training_profile
from ml.registry import APPROVAL_TAG, APPROVED, CHAMPION
from tests.synthetic import make_split

NAME = "atp-match-winner-monitoring-demo"
ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "results" / "monitoring-demo"


def container_uri(uri):
    # MLflow on Windows adds the host drive (file:///C:/registry). The API reads this registry
    # inside its Linux container at /registry, so the stored URIs must not contain a drive.
    return re.sub(r"^file:///[A-Za-z]:/registry/", "file:///registry/", uri)


def store_container_uris(database):
    with closing(sqlite3.connect(database)) as connection, connection:
        for table, column, key in [("experiments", "artifact_location", "experiment_id"),
                                   ("runs", "artifact_uri", "run_uuid")]:
            rows = connection.execute(f"SELECT {key}, {column} FROM {table}").fetchall()
            connection.executemany(f"UPDATE {table} SET {column} = ? WHERE {key} = ?",
                                   [(container_uri(value), row_id) for row_id, value in rows])


def prepare():
    if REGISTRY.exists() and any(REGISTRY.iterdir()):
        raise ValueError("Demo registry already exists; use it or choose a fresh checkout")
    REGISTRY.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{REGISTRY / 'mlflow.db'}"
    client = MlflowClient(uri, uri)
    experiment = client.create_experiment("monitoring-demo", "file:///registry/artifacts")
    client.create_registered_model(NAME)
    for index, driver in [(1, "age"), (2, "rank")]:
        version = f"v{index}"
        train = make_split([2020 + index], 4000, driver=driver, seed=30 + index)
        evaluation = make_split([2021 + index], 1000, driver=driver, seed=40 + index)
        folder = ROOT / ".exercise-data" / version
        folder.mkdir(parents=True, exist_ok=True)
        train.to_csv(folder / "train.csv", index=False)
        evaluation.to_csv(folder / "evaluation.csv", index=False)
        run = client.create_run(experiment, tags={"data_source": "synthetic"})
        model = train_model(train)
        sample = features(train.head(3))
        mlflow.sklearn.save_model(model, REGISTRY / f"model-{version}", input_example=sample,
                                 signature=infer_signature(sample,
                                                           model.predict_proba(sample)[:, 1]),
                                 pip_requirements=str(ROOT / "app" / "requirements.txt"))
        artifact = REGISTRY / "artifacts" / run.info.run_id / "artifacts"
        artifact.mkdir(parents=True)
        (artifact / "training-profile.json").write_text(json.dumps(training_profile(
            train, model_run_id=run.info.run_id, dataset_version=version, data_source="synthetic",
        )))
        client.create_model_version(NAME, f"file:///registry/model-{version}",
                                    run_id=run.info.run_id, tags={
                                        APPROVAL_TAG: APPROVED, "dataset_version": version,
                                        "approval_run_id": run.info.run_id,
                                        "data_source": "synthetic"})
        client.set_terminated(run.info.run_id)
    client.set_registered_model_alias(NAME, CHAMPION, "1")
    store_container_uris(REGISTRY / "mlflow.db")
    # This is deliberately shifted synthetic traffic, not an additional DVC tennis release.
    shifted = make_split([2024], 868, driver="rank", seed=99)
    shifted["surface"] = "Grass"
    shifted["player_a_rank"] = 1
    shifted["player_a_hand"] = "L"
    folder = ROOT / ".exercise-data" / "v3"
    folder.mkdir(parents=True, exist_ok=True)
    shifted.to_csv(folder / "evaluation.csv", index=False)
    for path in [REGISTRY, *REGISTRY.rglob("*")]:
        path.chmod(0o777 if path.is_dir() else 0o666)
    print("Prepared synthetic models and drift traffic; production registry unchanged")


def switch_version(version):
    if not (REGISTRY / "mlflow.db").exists():
        raise ValueError("Prepare the local demo registry first")
    uri = f"sqlite:///{REGISTRY / 'mlflow.db'}"
    client = MlflowClient(uri, uri)
    model = client.get_model_version(NAME, version)
    if model.tags.get(APPROVAL_TAG) != APPROVED:
        raise ValueError("Demo rollback requires an approved version")
    client.set_registered_model_alias(NAME, CHAMPION, version)
    print(f"Local monitoring champion is now {version}; restart the demo API")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--switch-version", choices=["1", "2"])
    args = parser.parse_args()
    if args.switch_version:
        switch_version(args.switch_version)
    else:
        prepare()


if __name__ == "__main__":
    main()
