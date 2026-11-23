import json

import pytest

from ml.ct_outputs import main, outputs, summary


@pytest.mark.parametrize(
    "decision, changed, version",
    [
        ({"decision": "bootstrapped", "champion_before": None, "champion_after": "1"}, "true", "1"),
        ({"decision": "promoted", "champion_before": "1", "champion_after": "2"}, "true", "2"),
        ({"decision": "rolled_back", "champion_before": "2", "champion_after": "1"}, "true", "1"),
        ({"decision": "retain_rejected", "champion_before": "1", "champion_after": "1"},
         "false", "1"),
        ({"decision": "retain_no_retraining", "champion_before": "1", "champion_after": "1"},
         "false", "1"),
        ({"decision": "failed", "champion_before": None, "champion_after": None}, "false", ""),
        ({"decision": "failed", "champion_before": "1", "champion_after": "1"}, "false", "1"),
        # The alias moved, then a later check failed: production must still be refreshed.
        ({"decision": "failed", "champion_before": "1", "champion_after": "2"}, "true", "2"),
        ({}, "false", ""),
    ],
)
def test_redeploy_only_when_the_champion_moved(decision, changed, version):
    assert outputs(decision) == {"champion_changed": changed, "champion_version": version}


def test_summary_lists_the_decision():
    text = summary({"decision": "promoted", "retrained": False, "champion_after": "2"})
    assert "| decision | promoted |" in text
    assert "| retrained | False |" in text
    assert "| reason | - |" in text


def test_missing_decision_file_still_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    main(tmp_path / "missing.json")
    assert (tmp_path / "output").read_text() == "champion_changed=false\nchampion_version=\n"
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"decision": "promoted", "champion_before": "1",
                                    "champion_after": "2"}))
    main(decision)
    assert "champion_changed=true\nchampion_version=2\n" in (tmp_path / "output").read_text()
    assert "| decision | promoted |" in (tmp_path / "summary").read_text()
