"""Read and update the approved model used by production inference."""

import os
import time

import mlflow
import mlflow.sklearn
import numpy
import pandas
import sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from ml.model import features

CHAMPION = "champion"
APPROVAL_TAG = "approval"
APPROVED = "approved"
DEFAULT_MODEL_NAME = "atp-match-winner"
DEFAULT_EXPERIMENT = "atp-match-winner"


def configure_tracking():
    uri = os.getenv("MLFLOW_TRACKING_URI", "").strip().rstrip("/")
    if not uri:
        raise ValueError("Set MLFLOW_TRACKING_URI to the DagsHub endpoint or a local SQL store")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI") or uri)
    return MlflowClient(), os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)


def get_champion(client, name):
    try:
        registered_model = client.get_registered_model(name)
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise
    selected_version = registered_model.aliases.get(CHAMPION)
    if selected_version is None:
        return None
    version = client.get_model_version(name, str(selected_version))
    if version.tags.get(APPROVAL_TAG) != APPROVED:
        raise ValueError("The champion alias must point to an approved model")
    return version


def load_version(version):
    # Resolve the mutable alias once, then load an immutable version for this execution.
    return mlflow.sklearn.load_model(f"models:/{version.name}/{version.version}")


def record_candidate(estimator, train, metadata):
    sample = features(train.head(5))
    return mlflow.sklearn.log_model(
        estimator,
        artifact_path="model",
        signature=infer_signature(sample, estimator.predict_proba(sample)[:, 1]),
        input_example=sample,
        # Serving needs these libraries only, not the training tools (DVC).
        pip_requirements=[
            f"mlflow=={mlflow.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            f"pandas=={pandas.__version__}",
            f"numpy=={numpy.__version__}",
        ],
        metadata=metadata,
    ).model_uri


def register_approved(client, name, model_uri, tags):
    version = mlflow.register_model(model_uri, name, await_registration_for=90)
    tags = {**tags, APPROVAL_TAG: APPROVED}
    # Some tracking services briefly delay a newly registered model's availability.
    deadline = time.monotonic() + 10
    while True:
        try:
            for key, value in tags.items():
                client.set_model_version_tag(name, version.version, key, str(value))
            return str(version.version)
        except MlflowException as exc:
            if exc.error_code != "RESOURCE_DOES_NOT_EXIST" or time.monotonic() >= deadline:
                raise
            time.sleep(1)


def move_champion(client, name, expected, target):
    """Guard against a stale decision before changing the production selection."""
    current = get_champion(client, name)
    current_version = str(current.version) if current else None
    if current_version != expected:
        raise ValueError("The champion changed during this run. Retry on the current champion.")
    client.set_registered_model_alias(name, CHAMPION, target)
    selected = get_champion(client, name)
    if selected is None or str(selected.version) != target:
        raise ValueError("The production selection could not be verified")
