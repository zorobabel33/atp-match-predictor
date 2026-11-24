"""The expected-model file must describe the registry champion that the service loads."""

import json

from scripts import expected_model
from tests import make_test_registry


def test_expected_model_matches_the_registry_champion(monkeypatch, tmp_path):
    registry = tmp_path / "registry"
    reference = make_test_registry.create_registry(registry)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(registry / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("MODEL_NAME", make_test_registry.MODEL_NAME)
    output = tmp_path / "expected.json"
    assert expected_model.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text()) == reference


def test_expected_model_fails_without_a_champion(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'empty.db').as_posix()}")
    monkeypatch.setenv("MODEL_NAME", "no-such-model")
    assert expected_model.main(["--output", str(tmp_path / "expected.json")]) == 1
    assert not (tmp_path / "expected.json").exists()
