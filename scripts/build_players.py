"""Create a small, attributed player snapshot from the pinned tennis_atp archive."""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

from ml.contract import ALLOWED_VALUES, RANGES
from ml.prepare_data import SOURCE_COMMIT, SOURCE_REPO

FILES = ("atp_players.csv", "atp_rankings_current.csv")


def build_catalog(players, rankings, limit=200):
    latest = max(row["ranking_date"] for row in rankings)
    reference = datetime.strptime(latest, "%Y%m%d").date()
    people = {row["player_id"]: row for row in players}
    result = []
    for ranking in sorted((r for r in rankings if r["ranking_date"] == latest),
                          key=lambda r: int(r["rank"])):
        person = people.get(ranking["player"])
        if person is None:
            continue
        try:
            birthday = datetime.strptime(person["dob"], "%Y%m%d").date()
            profile = {
                "id": person["player_id"],
                "name": f"{person['name_first']} {person['name_last']}".strip(),
                "country": person["ioc"],
                "rank": int(ranking["rank"]), "rank_points": int(ranking["points"]),
                "age": round((reference - birthday).days / 365.2425, 1),
                "ht": int(person["height"]), "hand": person["hand"],
            }
        except (ValueError, KeyError):
            continue
        if profile["hand"] not in ALLOWED_VALUES["player_a_hand"]:
            continue
        if any(profile[key] < low or (high is not None and profile[key] > high)
               for key in ("rank", "rank_points", "age", "ht")
               for low, high in [RANGES[f"player_a_{key}"]]):
            continue
        result.append(profile)
        if len(result) >= limit:
            break
    if len(result) < 2:
        raise ValueError("At least two complete, contract-compliant ranked players are required")
    return {"ranking_date": reference.isoformat(), "players": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("raw/players"))
    parser.add_argument("--output", type=Path, default=Path("app/static/players.json"))
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    if args.limit < 2:
        parser.error("--limit must be at least 2")
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    rows, hashes = [], {}
    for filename in FILES:
        path = args.raw_dir / filename
        if not path.exists():
            url = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}/atp/{filename}"
            with urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8-sig", newline="") as source:
            rows.append(list(csv.DictReader(source)))
    catalog = build_catalog(*rows, limit=args.limit)
    catalog.update({"source": SOURCE_REPO, "source_commit": SOURCE_COMMIT,
                    "license": "CC BY-NC-SA 4.0", "source_sha256": hashes})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Wrote {len(catalog['players'])} players, ranked {catalog['ranking_date']}")


if __name__ == "__main__":
    main()
