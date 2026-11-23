"""Create a mounted SQLite registry for container smoke tests, with no external services."""

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

from ml.model import features, train_model
from ml.registry import APPROVAL_TAG, APPROVED, CHAMPION
from tests.synthetic import make_split

MODEL_NAME = "atp-match-winner-ci"


def create_registry(output, container_dir=Path("/registry")):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty output directory for the test registry")
    uri = f"sqlite:///{(output / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    experiment = mlflow.create_experiment(
        "container-smoke", artifact_location=(output / "artifacts").as_uri(),
    )
    train = make_split([2021], 160, driver="rank", seed=42)
    model = train_model(train)
    sample = features(train.head(2))
    with mlflow.start_run(experiment_id=experiment) as run:
        mlflow.sklearn.save_model(
            model, output / "model", input_example=sample,
            signature=infer_signature(sample, model.predict_proba(sample)[:, 1]),
            pip_requirements=str(Path(__file__).resolve().parents[1] / "app/requirements.txt"),
        )
        run_id = run.info.run_id
    client = MlflowClient(uri, uri)
    client.create_registered_model(MODEL_NAME)
    # Paths in registry entries must resolve inside the container, not on the host runner.
    for _ in range(2):
        client.create_model_version(
            MODEL_NAME, (container_dir / "model").as_uri(), run_id=run_id,
            tags={APPROVAL_TAG: APPROVED, "dataset_version": "synthetic-ci",
                  "data_source": "synthetic"},
        )
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, "1")
    expected = {"model_name": MODEL_NAME, "model_version": "1", "run_id": run_id,
                "dataset_version": "synthetic-ci"}
    (output / "expected-model.json").write_text(json.dumps(expected, indent=2) + "\n")
    # mktemp directories are private by default. Make the synthetic fixture readable in Docker.
    for path in [output, *output.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    # MLflow 2 writes registered_model_meta beside local model artifacts during load_model.
    # Only this disposable synthetic model directory needs writes by the container's UID 10001.
    (output / "model").chmod(0o777)
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--container-dir", type=Path, default=Path("/registry"))
    args = parser.parse_args()
    if not args.container_dir.is_absolute():
        parser.error("--container-dir must be absolute")
    result = create_registry(args.output_dir, args.container_dir)
    print(f"Created {result['model_name']}: champion 1, newer approved version 2")


if __name__ == "__main__":
    main()
