"""Create an approved synthetic champion for local UI development without credentials."""

import argparse
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from ml.model import train_model
from ml.registry import APPROVAL_TAG, APPROVED, CHAMPION, record_candidate
from tests.synthetic import make_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/demo"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{(output / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    experiment = "courtside-ui-demo"
    name = "atp-match-winner-ui-demo"
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(output / "artifacts").as_uri())
    mlflow.set_experiment(experiment)
    client = MlflowClient(uri, uri)
    train = make_split([2021], 600, driver="rank", seed=42)
    model = train_model(train)
    with mlflow.start_run():
        source = record_candidate(model, train, {"data_source": "synthetic"})
        version = mlflow.register_model(source, name)
    for key, value in {APPROVAL_TAG: APPROVED, "dataset_version": "synthetic-demo",
                       "data_source": "synthetic"}.items():
        client.set_model_version_tag(name, version.version, key, value)
    client.set_registered_model_alias(name, CHAMPION, str(version.version))
    print("Synthetic model only: predictions are illustrative, not tennis forecasts.")
    print(f"MLFLOW_TRACKING_URI={uri}")
    print(f"MODEL_NAME={name}")


if __name__ == "__main__":
    main()
