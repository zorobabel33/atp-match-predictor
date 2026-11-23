"""End-to-end pipeline decisions against a temporary local MLflow registry."""

import json

import mlflow
import pytest
from mlflow import MlflowClient

from ml import pipeline, policy
from ml.pipeline import main
from ml.predict import main as predict
from ml.registry import APPROVAL_TAG, CHAMPION
from tests.synthetic import write_release

MODEL = "atp-test"


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Local SQLite registry, synthetic releases and an isolated output folder."""
    # v1 winners depend on age, v2 winners on rank: the v1 champion loses its edge on v2.
    write_release(tmp_path / "data", "v1", driver="age")
    write_release(tmp_path / "data", "v2", driver="rank")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    for key, value in {
        "MLFLOW_TRACKING_URI": uri, "MODEL_NAME": MODEL, "MLFLOW_EXPERIMENT_NAME": "test",
        "DATA_DIR": str(tmp_path / "data"), "RUN_OUTPUT_DIR": str(tmp_path / "results"),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    return MlflowClient(uri, uri)


def run(tmp_path, *args):
    code = main(list(args))
    decision = json.loads((tmp_path / "results" / "decision.json").read_text())
    return code, decision


def champion(client):
    # The local store returns the aliased version as an int, DagsHub as a string.
    return str(client.get_registered_model(MODEL).aliases.get(CHAMPION))


def test_bootstrap_then_retain_on_the_same_release(registry, tmp_path):
    code, decision = run(tmp_path, "--dataset-version", "v1")
    assert code == 0
    assert decision["decision"] == "bootstrapped" and decision["validation"] == "passed"
    assert decision["champion_after"] == "1" and champion(registry) == "1"
    version = registry.get_model_version(MODEL, "1")
    assert version.tags[APPROVAL_TAG] == "approved"
    assert version.tags["dataset_version"] == "v1"
    assert decision["git_sha"] and decision["data_source"] == "local"
    assert "dvc_md5" not in decision

    code, decision = run(tmp_path, "--dataset-version", "v1")
    assert code == 0
    assert decision["decision"] == "retain_no_retraining" and decision["retrained"] is False
    assert champion(registry) == "1"


def test_new_release_retrains_and_promotes_a_better_candidate(registry, tmp_path):
    run(tmp_path, "--dataset-version", "v1")
    code, decision = run(tmp_path, "--dataset-version", "v2")
    assert code == 0
    assert decision["retraining_trigger"] == "new_dataset_version"
    assert decision["decision"] == "promoted"
    assert decision["champion_before"] == "1" and decision["champion_after"] == "2"
    assert decision["candidate_metrics"]["roc_auc"] > decision["champion_metrics"]["roc_auc"]
    assert champion(registry) == "2"


def test_candidate_failing_the_gate_keeps_the_champion(registry, tmp_path, monkeypatch):
    run(tmp_path, "--dataset-version", "v1")
    monkeypatch.setattr(policy, "MIN_ROC_AUC_IMPROVEMENT", 1.0)
    code, decision = run(tmp_path, "--dataset-version", "v2")
    assert code == 0
    assert decision["retrained"] is True and decision["decision"] == "retain_rejected"
    assert "improvement" in decision["reason"]
    assert champion(registry) == "1"
    assert len(registry.search_model_versions(f"name='{MODEL}'")) == 1


def test_bootstrap_below_the_floor_fails_without_a_champion(registry, tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "MIN_ROC_AUC", 0.99)
    code, decision = run(tmp_path, "--dataset-version", "v1")
    assert code == 1
    assert decision["decision"] == "bootstrap_rejected" and "floor" in decision["reason"]
    assert registry.search_registered_models() == []


def test_invalid_release_stops_before_any_registry_change(registry, tmp_path):
    run(tmp_path, "--dataset-version", "v1")
    train = tmp_path / "data" / "v2" / "train.csv"
    train.write_text(train.read_text().replace(",Hard,", ",Carpet,", 1))
    code, decision = run(tmp_path, "--dataset-version", "v2")
    assert code == 1
    assert decision["validation"] == "failed" and decision["decision"] == "failed"
    assert champion(registry) == "1"


def test_older_window_than_the_champion_is_refused(registry, tmp_path):
    run(tmp_path, "--dataset-version", "v1")
    run(tmp_path, "--dataset-version", "v2")
    code, decision = run(tmp_path, "--dataset-version", "v1")
    assert code == 1 and "older" in decision["error"]
    assert decision["validation"] == "passed"
    assert champion(registry) == "2"


def test_failed_check_after_the_alias_moved_records_the_real_champion(
        registry, tmp_path, monkeypatch):
    run(tmp_path, "--dataset-version", "v1")

    def move_then_fail(client, name, expected, target):
        client.set_registered_model_alias(name, CHAMPION, target)
        raise ValueError("The production selection could not be verified")

    monkeypatch.setattr(pipeline, "move_champion", move_then_fail)
    code, decision = run(tmp_path, "--dataset-version", "v2")
    assert code == 1 and decision["decision"] == "failed"
    # decision.json must not claim the old champion is still in production.
    assert decision["champion_before"] == "1" and decision["champion_after"] == "2"
    assert champion(registry) == "2"


def test_rollback_restores_an_approved_version(registry, tmp_path):
    run(tmp_path, "--dataset-version", "v1")
    run(tmp_path, "--dataset-version", "v2")
    code, decision = run(tmp_path, "--rollback", "1")
    assert code == 0
    assert decision["decision"] == "rolled_back"
    assert decision["champion_before"] == "2" and decision["champion_after"] == "1"
    assert champion(registry) == "1"


def test_rollback_refuses_unknown_or_unapproved_versions(registry, tmp_path):
    run(tmp_path, "--dataset-version", "v1")
    code, decision = run(tmp_path, "--rollback", "9")
    assert code == 1 and decision["decision"] == "failed"
    # A registered version without the approval tag, as a manual upload would create.
    source = registry.get_model_version(MODEL, "1").source
    registry.create_model_version(MODEL, source)
    code, decision = run(tmp_path, "--rollback", "2")
    assert code == 1 and "approved" in decision["error"]
    assert champion(registry) == "1"


def test_prediction_uses_the_champion_even_when_a_newer_version_exists(registry, tmp_path):
    run(tmp_path, "--dataset-version", "v1")
    run(tmp_path, "--dataset-version", "v2")
    run(tmp_path, "--rollback", "1")
    evaluation = tmp_path / "data" / "v2" / "evaluation.csv"
    result = predict(["--input", str(evaluation), "--limit", "3"])
    assert result["model_version"] == "1"
    assert len(result["probability_player_a_wins"]) == 3
    assert all(0 <= p <= 1 for p in result["probability_player_a_wins"])


def test_prediction_without_a_champion_fails(registry, tmp_path):
    with pytest.raises(SystemExit):
        predict(["--input", str(tmp_path / "data" / "v1" / "evaluation.csv")])


def test_rollback_requires_a_positive_version_number(registry):
    with pytest.raises(SystemExit):
        main(["--rollback", "latest"])
