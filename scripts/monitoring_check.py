"""Verify the local exercise, optionally including real timed alert emails and rollback."""

import argparse
import base64
import json
import os
import subprocess
import sys
import threading
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from scripts.monitoring_demo import NAME, ROOT, switch_version
from scripts.replay import replay

COMPOSE = ["docker", "compose", "-f", "compose.yml", "-f", "monitoring/compose.demo.yml"]
API = "http://127.0.0.1:18000"
MAIL = "http://127.0.0.1:8025/api/v1/messages"


def get(url, grafana=False):
    request = Request(url)
    if grafana:
        request.add_header("Authorization", "Basic " + base64.b64encode(b"admin:admin").decode())
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def wait(check, description, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = check()
            if result:
                print(description, flush=True)
                return result
        except (OSError, ValueError, KeyError, IndexError):
            pass
        time.sleep(5)
    raise RuntimeError(f"Timed out: {description}")


def compose(*args, unavailable=False):
    env = {**os.environ, "DEMO_MODEL_NAME": "missing-monitoring-demo" if unavailable else NAME}
    subprocess.run([*COMPOSE, *args], cwd=ROOT, env=env, check=True,
                   stdout=subprocess.DEVNULL)


def normal():
    result = replay(ROOT / ".exercise-data/v1/train.csv", API + "/predict", 100, 1000, True)
    if result != {"200": 1000}:
        raise ValueError(f"Normal traffic failed: {result}")


def metric(name):
    query = urlencode({"query": name + '{job="api"}'})
    return get("http://127.0.0.1:9090/api/v1/query?" + query)["data"]["result"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alerts", action="store_true",
                        help="test firing/resolved emails (~10 min)")
    parser.add_argument("--rollback", action="store_true", help="verify local versions 1 to 2 to 1")
    args = parser.parse_args()
    output = ROOT / ".exercise-results"
    output.mkdir(exist_ok=True)
    evidence = {"environment": "isolated synthetic local exercise", "emails": []}
    wait(lambda: get(API + "/ready")["status"] == "ready", "API ready")
    if get(API + "/model-info")["model_name"] != NAME:
        raise ValueError("Refusing to exercise a non-demo model")
    wait(lambda: get("http://127.0.0.1:3000/api/health")["database"] == "ok", "Grafana ready")
    rules = get("http://127.0.0.1:3000/api/v1/provisioning/alert-rules", True)
    assert len(rules) == 3
    assert all(r["noDataState"] == "OK" for r in rules if r["uid"] != "courtside-down")
    assert len(get("http://127.0.0.1:3000/api/dashboards/uid/courtside-api", True)[
        "dashboard"]["panels"]) == 6
    wait(lambda: metric("drift_reference_ready")[0]["value"][1] == "1", "Reference loaded")
    normal()
    baseline = wait(lambda: metric("feature_drift"), "Drift metrics scraped")
    assert len(baseline) == 13 and max(float(v["value"][1]) for v in baseline) < 0.2
    evidence["baseline_drift"] = {v["metric"]["feature"]: float(v["value"][1]) for v in baseline}
    print("Six dashboard panels and 13 low-drift features verified", flush=True)

    def email(title, state, previous):
        def find():
            for item in get(MAIL)["messages"]:
                subject = item.get("Subject", item.get("subject", ""))
                identifier = item.get("ID", item.get("id"))
                if identifier not in previous and title in subject and state in subject.upper():
                    return {"id": identifier, "subject": subject}
            return None

        result = wait(find, f"Received {state} email: {title}", 360)
        evidence["emails"].append(result)

    def mail_ids():
        return {x.get("ID", x.get("id")) for x in get(MAIL)["messages"]}

    if args.alerts:
        previous = mail_ids()
        replay(ROOT / ".exercise-data/v3/evaluation.csv", API + "/predict", 100, 868)
        email("Courtside data drift", "FIRING", previous)
        normal()
        email("Courtside data drift", "RESOLVED", previous)

        previous = mail_ids()
        stop = threading.Event()

        def traffic():
            while not stop.is_set():
                replay(ROOT / ".exercise-data/v1/train.csv", API + "/predict", 20, 100)

        try:
            compose("up", "-d", "--no-deps", "api", unavailable=True)
            thread = threading.Thread(target=traffic, daemon=True)
            thread.start()
            email("Courtside high error rate", "FIRING", previous)
        finally:
            stop.set()
            if "thread" in locals():
                thread.join(timeout=15)
            compose("up", "-d", "--no-deps", "api")
        wait(lambda: get(API + "/ready")["status"] == "ready", "API restored")
        normal()
        email("Courtside high error rate", "RESOLVED", previous)

        previous = mail_ids()
        try:
            compose("stop", "api")
            email("Courtside API down", "FIRING", previous)
        finally:
            compose("start", "api")
        wait(lambda: get(API + "/ready")["status"] == "ready", "API restarted")
        email("Courtside API down", "RESOLVED", previous)

    if args.rollback:
        evidence["model_versions"] = ["1"]
        try:
            switch_version("2")
            compose("restart", "api")
            wait(lambda: get(API + "/model-info")["model_version"] == "2", "Model 2 loaded")
            wait(lambda: metric("model_info")[0]["metric"]["model_version"] == "2",
                 "Prometheus sees version 2")
            evidence["model_versions"].append("2")
            assert replay(ROOT / ".exercise-data/v2/train.csv", API + "/predict", 100, 100,
                          True) == {"200": 100}
            wait(lambda: any(v["metric"]["model_version"] == "2"
                             and float(v["value"][1]) > 0 for v in metric("predictions_total")),
                 "Version 2 prediction counters scraped")
            # Keep at least two scrape intervals so rate/increase graphs retain the short rollback.
            time.sleep(10)
        finally:
            switch_version("1")
            compose("restart", "api")
        wait(lambda: get(API + "/model-info")["model_version"] == "1", "Model 1 restored")
        wait(lambda: metric("model_info")[0]["metric"]["model_version"] == "1",
             "Prometheus sees version 1")
        evidence["model_versions"].append("1")
        normal()
    (output / "check.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("Monitoring exercise verified; evidence in .exercise-results/check.json", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        print(f"Monitoring check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
