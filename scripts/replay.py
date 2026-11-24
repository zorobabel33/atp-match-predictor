"""Replay only the 13 tennis features; labels and match identifiers never enter /predict."""

import argparse
import json
import math
import time
import uuid
from collections import Counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from ml.contract import FEATURES, INTEGER, RANGES


def payloads(path, limit, shuffle=False, broken=False):
    frame = pd.read_csv(path)
    if not set(FEATURES) <= set(frame.columns):
        raise ValueError("CSV is missing tennis features")
    if shuffle:
        frame = frame.sample(frac=1, random_state=42)
    for row in frame.head(limit).to_dict(orient="records"):
        payload = {name: int(row[name]) if name in INTEGER else float(row[name])
                   if name in RANGES else str(row[name]) for name in FEATURES}
        if broken:
            del payload[FEATURES[0]]
        yield payload


def replay(path, url, rate, limit, shuffle=False, broken=False):
    statuses = Counter()
    for payload in payloads(path, limit, shuffle, broken):
        started = time.monotonic()
        request = Request(url, data=json.dumps(payload, allow_nan=False).encode(),
                          headers={"Content-Type": "application/json",
                                   "X-Request-ID": uuid.uuid4().hex})
        try:
            with urlopen(request, timeout=10) as response:
                statuses[str(response.status)] += 1
        except HTTPError as exc:
            statuses[str(exc.code)] += 1
        except (URLError, OSError, TimeoutError):
            statuses["unreachable"] += 1
        time.sleep(max(0, 1 / rate - (time.monotonic() - started)))
    return dict(statuses)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18000/predict")
    parser.add_argument("--rate", type=float, default=20)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--broken", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.rate) or args.rate <= 0 or args.limit <= 0:
        parser.error("rate and limit must be positive and finite")
    print("Responses by status:", replay(args.file, args.url, args.rate, args.limit,
                                         args.shuffle, args.broken))


if __name__ == "__main__":
    main()
