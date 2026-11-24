"""Inference contracts, failure states and champion selection without external services."""

import threading
import time

import mlflow
import numpy as np
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from app import main
from ml.contract import FEATURES
from ml.model import train_model
from ml.registry import APPROVAL_TAG, APPROVED, CHAMPION, record_candidate
from tests.synthetic import make_split


def wait_ready(client):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if client.get("/ready").status_code == 200:
            return
        time.sleep(0.01)
    pytest.fail("The local champion did not become ready")


def wait_initialized(application):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if application.state.load_task.done():
            return
        time.sleep(0.01)
    pytest.fail("Model initialization did not finish")


@pytest.fixture(scope="module")
def local_registry(tmp_path_factory):
    root = tmp_path_factory.mktemp("web-registry")
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    experiment = mlflow.create_experiment(
        "web-tests", artifact_location=(root / "artifacts").as_uri(),
    )
    train = make_split([2021], 160, driver="rank", seed=9)
    with mlflow.start_run(experiment_id=experiment):
        source = record_candidate(train_model(train), train, {"data_source": "synthetic"})
    client = MlflowClient(uri, uri)
    name = "atp-web-test"
    for _ in range(2):
        version = mlflow.register_model(source, name)
        client.set_model_version_tag(name, version.version, APPROVAL_TAG, APPROVED)
        client.set_model_version_tag(name, version.version, "dataset_version", "test-v1")
    return uri, name, client


@pytest.fixture
def configured_registry(local_registry, monkeypatch):
    uri, name, client = local_registry
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MODEL_NAME", name)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    client.set_model_version_tag(name, "1", APPROVAL_TAG, APPROVED)
    client.set_registered_model_alias(name, CHAMPION, "1")
    return client, name


@pytest.fixture
def web(configured_registry):
    with TestClient(main.create_app()) as client:
        wait_ready(client)
        yield client


@pytest.mark.parametrize("changes", [
    {}, {"player_a_hand": "L", "player_b_hand": "R", "best_of": 5},
    {"player_a_rank": 1, "player_a_age": 14, "player_b_age": 50,
     "player_a_ht": 150, "player_b_ht": 220, "surface": "Grass"},
])
def test_prediction_output(web, changes):
    payload = {**main.example_match(), **changes}
    response = web.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"probability_player_a_wins", "predicted_winner", "model_version"}
    score = body["probability_player_a_wins"]
    assert 0 <= score <= 1
    assert body["predicted_winner"] == ("player_a" if score >= 0.5 else "player_b")
    assert body["model_version"] == "1"


@pytest.mark.parametrize("changes,missing", [
    ({}, "surface"), ({"surface": "Carpet"}, None), ({"surface": 1}, None),
    ({"tourney_level": "D"}, None), ({"player_b_hand": "U"}, None),
    ({"best_of": 4}, None), ({"best_of": True}, None), ({"player_a_rank": 0}, None),
    ({"player_a_rank": 1.5}, None), ({"player_a_rank_points": "9000"}, None),
    ({"player_b_ht": 300}, None), ({"player_b_age": "young"}, None),
    ({"player_a_age": 13.9}, None), ({"player_a_age": 50.1}, None),
    ({"player_b_age": None}, None), ({"name": "Unrecognized field"}, None),
])
def test_invalid_input_never_reaches_model(web, monkeypatch, changes, missing):
    def forbidden(*args):
        pytest.fail("Invalid input must not reach the model")

    monkeypatch.setattr(main, "probability", forbidden)
    payload = {**main.example_match(), **changes}
    if missing:
        del payload[missing]
    assert web.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("content", ["{", "[]", "null", '{"player_a_age": NaN}',
                                          '{"player_a_age": Infinity}'])
def test_malformed_json_is_422(web, content):
    assert web.post("/predict", content=content,
                    headers={"Content-Type": "application/json"}).status_code == 422


def test_champion_is_pinned_even_when_newer_version_and_alias_change(web, configured_registry):
    client, name = configured_registry
    info = web.get("/model-info").json()
    assert info["model_version"] == "1"
    assert info["model_name"] == name and info["run_id"]
    assert info["dataset_version"] == "test-v1"
    client.set_registered_model_alias(name, CHAMPION, "2")
    assert web.get("/model-info").json()["model_version"] == "1"
    assert web.post("/predict", json=main.example_match()).json()["model_version"] == "1"


@pytest.mark.parametrize("mode", ["no_alias", "unapproved", "unconfigured", "broken_artifact"])
def test_unavailable_champion_keeps_page_available(configured_registry, monkeypatch, mode):
    client, name = configured_registry
    if mode == "no_alias":
        client.delete_registered_model_alias(name, CHAMPION)
    elif mode == "unapproved":
        client.set_model_version_tag(name, "1", APPROVAL_TAG, "pending")
    elif mode == "unconfigured":
        monkeypatch.delenv("MLFLOW_TRACKING_URI")
    else:
        def broken(version):
            raise OSError("Artifact unavailable")
        monkeypatch.setattr(main, "load_version", broken)
    application = main.create_app()
    with TestClient(application) as web:
        wait_initialized(application)
        assert web.get("/health").status_code == 200
        assert web.get("/").status_code == 200
        assert web.get("/ready").status_code == 503
        assert web.get("/model-info").status_code == 503
        assert web.post("/predict", json=main.example_match()).status_code == 503


def test_readiness_waits_for_loading_and_a_successful_prediction(configured_registry, monkeypatch):
    release = threading.Event()
    entered = threading.Event()
    original = main.probability

    def delayed(model, payload):
        entered.set()
        if not release.wait(timeout=10):
            raise TimeoutError("Startup test timed out")
        return original(model, payload)

    monkeypatch.setattr(main, "probability", delayed)
    try:
        with TestClient(main.create_app()) as web:
            assert entered.wait(timeout=10)
            assert web.get("/health").status_code == 200
            assert web.get("/ready").status_code == 503
            release.set()
            wait_ready(web)
    finally:
        release.set()


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_model_prediction_is_not_reported_as_ready(configured_registry, monkeypatch, score):
    class BrokenModel:
        def predict_proba(self, frame):
            return np.array([[1 - score, score]])

    monkeypatch.setattr(main, "load_version", lambda version: BrokenModel())
    application = main.create_app()
    with TestClient(application) as web:
        wait_initialized(application)
        assert web.get("/ready").status_code == 503
        assert web.post("/predict", json=main.example_match()).status_code == 503


def test_catalog_and_interface_are_available_without_a_model(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with TestClient(main.create_app()) as web:
        assert "Deux joueurs." in web.get("/").text
        assert web.get("/static/styles.css").status_code == 200
        assert web.get("/static/app.js").status_code == 200
        assert web.get("/contract").json()["features"] == list(FEATURES)
        catalog = web.get("/static/players.json").json()
        assert catalog["ranking_date"] == "2026-06-08"
        assert catalog["license"] == "CC BY-NC-SA 4.0"
        assert len(catalog["players"]) == 200
        for person in catalog["players"]:
            payload = main.example_match()
            payload.update({f"player_a_{name}": person[name]
                            for name in ("rank", "rank_points", "age", "ht", "hand")})
            main.MatchInput.model_validate(payload)
