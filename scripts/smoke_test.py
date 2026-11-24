"""Verify a running inference service through HTTP, including its expected model and prediction."""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# A fixed representative request makes this check independent of the service's /contract response.
SAMPLE = {
    "surface": "Hard", "tourney_level": "G", "best_of": 5,
    "player_a_rank": 1, "player_a_rank_points": 9500, "player_a_age": 23.5,
    "player_a_ht": 188, "player_a_hand": "R",
    "player_b_rank": 45, "player_b_rank_points": 1200, "player_b_age": 29.1,
    "player_b_ht": 183, "player_b_hand": "L",
}


class SmokeError(ValueError):
    """The service returned an unexpected result."""


def request(base_url, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    query = Request(base_url.rstrip("/") + path, data=data,
                    headers={"Content-Type": "application/json"} if data else {})
    with urlopen(query, timeout=5) as response:
        return response.read()


def check_service(base_url, expected, timeout=90):
    deadline = time.monotonic() + timeout
    while True:
        try:
            ready = json.loads(request(base_url, "/ready"))
            if isinstance(ready, dict) and ready.get("status") == "ready":
                break
        except (URLError, TimeoutError, OSError, ValueError):
            pass
        if time.monotonic() >= deadline:
            raise SmokeError("The service did not become ready before the deadline")
        time.sleep(1)

    if json.loads(request(base_url, "/health")).get("status") != "ok":
        raise SmokeError("Health check failed")
    if b"<html" not in request(base_url, "/").lower():
        raise SmokeError("The web page was not served")
    info = json.loads(request(base_url, "/model-info"))
    if not isinstance(info, dict):
        raise SmokeError("Model information is not a JSON object")
    for field in ("model_name", "model_version", "run_id", "dataset_version"):
        if info.get(field) != expected[field]:
            raise SmokeError(f"Unexpected {field} in /model-info")

    result = json.loads(request(base_url, "/predict", SAMPLE))
    if not isinstance(result, dict):
        raise SmokeError("The prediction is not a JSON object")
    score = result.get("probability_player_a_wins")
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise SmokeError("The prediction probability is missing or invalid")
    if result.get("predicted_winner") != ("player_a" if score >= 0.5 else "player_b"):
        raise SmokeError("The predicted winner is inconsistent with its probability")
    if result.get("model_version") != expected["model_version"]:
        raise SmokeError("The prediction used an unexpected model version")

    try:
        request(base_url, "/predict", {**SAMPLE, "player_a_rank": 0})
    except HTTPError as exc:
        if exc.code != 422:
            raise SmokeError("Invalid input did not return 422") from None
    else:
        raise SmokeError("Invalid input was accepted")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--expected-model-info", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    try:
        expected = json.loads(args.expected_model_info.read_text())
        result = check_service(args.url, expected, args.timeout)
    except (SmokeError, URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
        # Do not echo response bodies or URLs, which may contain runtime credentials.
        detail = str(exc) if isinstance(exc, SmokeError) else type(exc).__name__
        print(f"Smoke test failed: {detail}", file=sys.stderr)
        return 1
    print(f"Smoke test passed: model version {result['model_version']}, prediction verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
