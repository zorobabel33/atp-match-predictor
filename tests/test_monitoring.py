"""Correlation, failure handling, bounded metrics and drift follow the served model."""

import copy
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from app import main
from app.monitoring import Telemetry
from ml.contract import FEATURES
from ml.drift import compare_distributions, training_profile, validate_profile
from ml.registry import load_training_profile
from scripts.replay import payloads
from tests.synthetic import make_split
from tests.test_app import wait_initialized


class Model:
    def predict_proba(self, frame):
        return np.array([[0.2, 0.8]] * len(frame))


@pytest.fixture
def client(monkeypatch, capsys):
    monkeypatch.setattr(main, "load_champion", lambda: (Model(), {
        "model_name": "test", "model_version": "1", "run_id": "run-1",
        "dataset_version": "v1", "data_source": "synthetic"}))
    monkeypatch.setattr(Telemetry, "load_reference", lambda *args: None)
    with TestClient(main.create_app()) as web:
        wait_initialized(web.app)
        yield web


def records(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


def test_request_correlation_prediction_fields_and_no_private_data(client, capsys):
    response = client.post("/predict?token=private-query", json=main.example_match(),
                           headers={"X-Request-ID": "duel-123",
                                    "Authorization": "Bearer private-credential"})
    assert response.status_code == 200 and response.headers["X-Request-ID"] == "duel-123"
    logs = records(capsys)
    request = next(r for r in logs if r["event"] == "request_finished")
    prediction = next(r for r in logs if r["event"] == "prediction_made")
    assert request["request_id"] == prediction["request_id"] == "duel-123"
    assert {k: request[k] for k in ("method", "path", "status")} == {
        "method": "POST", "path": "/predict", "status": 200}
    assert request["duration_ms"] >= 0 and prediction["latency_ms"] >= 0
    assert prediction["number_of_features"] == 13 and prediction["number_of_missing_values"] == 0
    assert prediction["prediction"] == "1" and prediction["model_version"] == "1"
    assert "private" not in json.dumps(logs)
    assert not any(name in prediction for name in FEATURES)


def test_failures_have_error_logs_ids_and_metrics(client, capsys):
    response = client.post("/predict", json={})
    assert response.status_code == 422 and response.headers["X-Request-ID"]

    @client.app.get("/boom")
    def boom():
        raise RuntimeError("private-exception-detail")

    response = client.get("/boom")
    assert response.status_code == 500 and response.headers["X-Request-ID"]
    assert "private" not in response.text
    logs = records(capsys)
    assert all(r["level"] == "error" for r in logs if r["event"] == "request_finished")
    assert "private" not in json.dumps(logs)
    metrics = client.get("/metrics").text
    assert 'http_requests_total{method="POST",path="/predict",status="422"} 1.0' in metrics
    assert 'http_requests_total{method="GET",path="/boom",status="500"} 1.0' in metrics
    assert 'model_info{model_version="1"} 1.0' in metrics


def test_model_error_remains_503_and_is_not_a_prediction(client, monkeypatch, capsys):
    def broken(*args):
        raise ValueError("private-model-error")

    monkeypatch.setattr(main, "probability", broken)
    assert client.post("/predict", json=main.example_match()).status_code == 503
    logs = records(capsys)
    assert "private" not in json.dumps(logs)
    assert next(r for r in logs if r["event"] == "prediction_failed")["level"] == "error"
    for label in ("0", "1"):
        assert client.app.state.telemetry.registry.get_sample_value(
            "predictions_total", {"prediction": label, "model_version": "1"}) == 0


def test_monitoring_failure_does_not_break_a_valid_prediction(client, monkeypatch, capsys):
    def broken(*args):
        raise RuntimeError("private-monitoring-detail")

    monkeypatch.setattr(client.app.state.telemetry, "prediction", broken)
    assert client.post("/predict", json=main.example_match()).status_code == 200
    logs = records(capsys)
    assert "private" not in json.dumps(logs)
    assert any(r["event"] == "monitoring_failed" for r in logs)


def test_unmatched_paths_and_bad_request_ids_are_not_logged(client, capsys):
    response = client.get("/private-path-value", headers={"X-Request-ID": "invalid id"})
    assert response.headers["X-Request-ID"] != "invalid id"
    log = next(r for r in records(capsys) if r["event"] == "request_finished")
    assert log["path"] == "/unmatched"
    assert "private-path-value" not in client.get("/metrics").text


def test_log_level_comes_from_environment(monkeypatch, capsys):
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    telemetry = Telemetry()
    telemetry.log.info("hidden")
    telemetry.log.error("visible")
    assert [r["event"] for r in records(capsys)] == ["visible"]


def test_drift_waits_for_500_refreshes_each_100_and_recovers():
    train = make_split([2021], 500, "rank", seed=12)
    reference = training_profile(train)
    telemetry = Telemetry()
    telemetry.reference = reference
    for row in train.head(499).to_dict("records"):
        telemetry.prediction(row, 0.8, "1", 0.001)
    assert "feature_drift{" not in generate_latest(telemetry.registry).decode()
    telemetry.prediction(train.iloc[-1].to_dict(), 0.8, "1", 0.001)
    assert telemetry.drift.labels("surface")._value.get() == pytest.approx(0)
    shifted = train.copy()
    shifted["surface"] = "Grass"
    for row in shifted.head(99).to_dict("records"):
        telemetry.prediction(row, 0.8, "1", 0.001)
    assert telemetry.drift.labels("surface")._value.get() == pytest.approx(0)
    for row in shifted.iloc[99:].to_dict("records"):
        telemetry.prediction(row, 0.8, "1", 0.001)
    assert len(telemetry.window) == 500 and telemetry.drift.labels("surface")._value.get() > 0.2
    for row in train.to_dict("records"):
        telemetry.prediction(row, 0.8, "1", 0.001)
    assert telemetry.drift.labels("surface")._value.get() == pytest.approx(0)


def test_reference_is_downloaded_from_exact_approval_run_and_checks_provenance(tmp_path):
    from types import SimpleNamespace

    profile = training_profile(make_split([2021], 100, "age", 1),
                               model_run_id="original-model-run", dataset_version="v1")
    artifact = tmp_path / "training-profile.json"
    artifact.write_text(json.dumps(profile))
    calls = []

    class Client:
        def download_artifacts(self, run_id, path, destination):
            calls.append((run_id, path))
            return str(artifact)

    version = SimpleNamespace(run_id="original-model-run", tags={"approval_run_id": "approval-v1",
                                                                "dataset_version": "v1"})
    assert load_training_profile(Client(), version) == profile
    assert calls == [("approval-v1", "training-profile.json")]
    version.run_id = "different-model-run"
    with pytest.raises(ValueError, match="provenance"):
        load_training_profile(Client(), version)
    version.run_id = "original-model-run"
    version.tags.update(data_source="dvc", dvc_md5="expected-hash")
    with pytest.raises(ValueError, match="provenance"):
        load_training_profile(Client(), version)


def test_new_process_uses_the_new_models_reference_and_clears_drift(monkeypatch):
    import app.monitoring as monitoring

    frame = make_split([2021], 100, "age", 1)
    references = {"1": training_profile(frame, dataset_version="v1"),
                  "2": training_profile(frame.assign(surface="Grass"), dataset_version="v2")}

    class Client:
        def get_model_version(self, name, version):
            return version

    monkeypatch.setattr(monitoring, "configure_tracking", lambda: (Client(), "test"))
    monkeypatch.setattr(monitoring, "load_training_profile", lambda client, v: references[v])
    first, second = Telemetry(), Telemetry()
    first.load_reference({"model_name": "test", "model_version": "1"})
    second.load_reference({"model_name": "test", "model_version": "2"})
    assert first.reference is references["1"] and second.reference is references["2"]
    assert len(second.window) == 0 and second.count == 0


def test_profile_rejects_corrupted_counts_and_replay_strips_labels(tmp_path):
    frame = make_split([2021], 100, "rank", 42)
    profile = training_profile(frame)
    corrupted = copy.deepcopy(profile)
    corrupted["features"]["surface"]["counts"][0] = -1
    with pytest.raises(ValueError):
        validate_profile(corrupted)
    assert max(compare_distributions(profile, frame).values()) == pytest.approx(0)
    csv = tmp_path / "matches.csv"
    frame.to_csv(csv, index=False)
    assert set(next(payloads(csv, 1))) == set(FEATURES)
    assert set(next(payloads(csv, 1, broken=True))) == set(FEATURES) - {FEATURES[0]}


def test_demo_wait_tolerates_an_empty_scrape_during_restart(monkeypatch):
    from scripts import monitoring_check

    observations = iter([[], [{"version": "2"}]])
    monkeypatch.setattr(monitoring_check.time, "sleep", lambda seconds: None)
    assert monitoring_check.wait(lambda: next(observations)[0]["version"] == "2",
                                 "version scraped", timeout=1)


def test_demo_registry_keeps_container_paths_on_windows(tmp_path):
    from mlflow import MlflowClient

    from scripts.monitoring_demo import container_uri, store_container_uris

    windows = "file:///C:/registry/artifacts/abc/artifacts"
    assert container_uri(windows) == "file:///registry/artifacts/abc/artifacts"
    assert container_uri("file:///registry/artifacts") == "file:///registry/artifacts"
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(uri, uri)
    run = client.create_run(client.create_experiment("demo", "file:///registry/artifacts"))
    store_container_uris(tmp_path / "mlflow.db")
    assert client.get_run(run.info.run_id).info.artifact_uri == (
        f"file:///registry/artifacts/{run.info.run_id}/artifacts")
    assert client.get_experiment_by_name("demo").artifact_location == "file:///registry/artifacts"
