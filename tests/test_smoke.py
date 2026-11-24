"""The CI smoke command must reject a broken predictor or an unexpected loaded model."""

import json
from urllib.error import HTTPError

import pytest

from scripts import smoke_test


@pytest.fixture
def service(monkeypatch, tmp_path):
    expected = {"model_name": "test", "model_version": "1", "run_id": "test-run",
                "dataset_version": "synthetic-ci"}
    path = tmp_path / "expected.json"
    path.write_text(json.dumps(expected))
    replies = {
        "/health": {"status": "ok"}, "/ready": {"status": "ready"},
        "/model-info": expected.copy(),
        "/predict": {"probability_player_a_wins": 0.75,
                     "predicted_winner": "player_a", "model_version": "1"},
    }

    def request(url, endpoint, payload=None):
        if endpoint == "/":
            return b"<!doctype html><html lang='fr'></html>"
        if payload is not None and payload["player_a_rank"] == 0:
            raise HTTPError(url, 422, "Invalid input", None, None)
        response = replies[endpoint]
        if isinstance(response, Exception):
            raise response
        return json.dumps(response).encode()

    monkeypatch.setattr(smoke_test, "request", request)
    return replies, ["--url", "http://localhost:8000", "--expected-model-info", str(path)]


@pytest.mark.parametrize("score,winner", [(0, "player_b"), (0.49, "player_b"),
                                        (0.5, "player_a"), (1, "player_a")])
def test_smoke_accepts_valid_predictions(service, score, winner):
    replies, args = service
    replies["/predict"].update(probability_player_a_wins=score, predicted_winner=winner)
    assert smoke_test.main(args) == 0


@pytest.mark.parametrize("changes", [
    {"probability_player_a_wins": -0.1}, {"probability_player_a_wins": 1.1},
    {"probability_player_a_wins": float("nan")}, {"probability_player_a_wins": None},
    {"probability_player_a_wins": "0.75"}, {"probability_player_a_wins": True},
    {"predicted_winner": "player_b"}, {"predicted_winner": "unknown"},
    {"model_version": "2"},
])
def test_smoke_rejects_invalid_predictions(service, changes):
    replies, args = service
    replies["/predict"].update(changes)
    assert smoke_test.main(args) == 1


@pytest.mark.parametrize("field", ["model_name", "model_version", "run_id", "dataset_version"])
def test_smoke_checks_expected_metadata_instead_of_trusting_loaded_version(service, field):
    replies, args = service
    replies["/model-info"][field] = "unexpected"
    assert smoke_test.main(args) == 1


@pytest.mark.parametrize("response", [
    [], {}, HTTPError("http://localhost", 500, "Broken", None, None),
])
def test_smoke_command_fails_on_a_broken_predictor(service, response):
    replies, args = service
    replies["/predict"] = response
    assert smoke_test.main(args) == 1


def test_smoke_readiness_wait_is_bounded(service, monkeypatch):
    replies, args = service
    replies["/ready"] = HTTPError("http://localhost", 503, "Not ready", None, None)
    clock = iter([0, 2])
    monkeypatch.setattr(smoke_test.time, "monotonic", lambda: next(clock))
    assert smoke_test.main([*args, "--timeout", "1"]) == 1
