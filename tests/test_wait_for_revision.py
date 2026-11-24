"""Deployment checks must accept only the healthy revision of the requested image."""

import copy
import subprocess

import pytest

from scripts import wait_for_revision as checks

IMAGE = "ghcr.io/example/app:commit-sha"
HEALTHY = {"properties": {
    "active": True, "provisioningState": "Provisioned", "healthState": "Healthy",
    "runningState": "Running", "template": {"containers": [{"image": IMAGE}]},
}}
ARGS = ["--app-name", "app", "--resource-group", "group", "--revision", "app--new",
        "--image", IMAGE, "--timeout", "1"]


def test_checks_the_revision_without_latest_ready_revision_name(monkeypatch):
    # Some Container App responses omit latestReadyRevisionName even while HTTP works.
    calls = []

    def read(app, group, revision):
        calls.append((app, group, revision))
        return HEALTHY

    monkeypatch.setattr(checks, "read_revision", read)
    assert checks.main(ARGS) == 0
    assert calls == [("app", "group", "app--new")]


@pytest.mark.parametrize("change", [
    {"healthState": "Unhealthy"},
    {"active": False},
    {"template": {"containers": [{"image": "ghcr.io/example/app:old-sha"}]}},
])
def test_never_accepts_an_unhealthy_inactive_or_different_image(monkeypatch, change):
    revision = copy.deepcopy(HEALTHY)
    revision["properties"].update(change)
    monkeypatch.setattr(checks, "read_revision", lambda *args: revision)
    clock = iter([0, 2])
    monkeypatch.setattr(checks.time, "monotonic", lambda: next(clock))
    assert checks.main(ARGS) == 1


def test_waits_for_startup_before_succeeding(monkeypatch):
    starting = copy.deepcopy(HEALTHY)
    starting["properties"].update(healthState="None", provisioningState="Provisioning")
    revisions = iter([starting, HEALTHY])
    monkeypatch.setattr(checks, "read_revision", lambda *args: next(revisions))
    monkeypatch.setattr(checks.time, "sleep", lambda seconds: None)
    assert checks.main(ARGS) == 0


def test_cli_errors_fail_without_exposing_output(monkeypatch, capsys):
    def fail(*args):
        raise subprocess.CalledProcessError(1, ["az"], output="private-output",
                                            stderr="private-error")

    monkeypatch.setattr(checks, "read_revision", fail)
    assert checks.main(ARGS) == 1
    captured = capsys.readouterr()
    assert "CalledProcessError" in captured.err
    assert "private" not in captured.out + captured.err
