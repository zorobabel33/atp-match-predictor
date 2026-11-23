"""Build a prepared dataset release from the pinned tennis_atp archive."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from ml.contract import ALLOWED_VALUES, COLUMNS, FEATURES, INTEGER, LABEL, PLAYER_FEATURES, RANGES

# The original JeffSackmann/tennis_atp repository is no longer public. This archive mirrors it
# under the same CC BY-NC-SA 4.0 license, pinned to one commit for reproducibility.
SOURCE_REPO = "Aneeshers/tennis-sackmann-archive"
SOURCE_COMMIT = "83733587353df8a41f2fd4f516147d5aa83f5a8d"
SOURCE_URL = "https://raw.githubusercontent.com/{repo}/{commit}/atp/atp_matches_{season}.csv"

SEED = 42
# Training seasons, then the later evaluation season that no model of that release has seen.
RELEASES = {
    "v1": (tuple(range(2016, 2022)), 2022),
    "v2": (tuple(range(2016, 2023)), 2023),
}


def load_raw(raw_dir, seasons):
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in seasons:
        path = raw_dir / f"atp_matches_{season}.csv"
        if not path.exists():
            url = SOURCE_URL.format(repo=SOURCE_REPO, commit=SOURCE_COMMIT, season=season)
            with urlopen(url, timeout=60) as response, path.open("wb") as output:
                shutil.copyfileobj(response, output)
        frames.append(pd.read_csv(path).assign(season=season))
    return pd.concat(frames, ignore_index=True)


def player_a_is_winner(match_id):
    """Seeded coin flip per match: the same match gets the same players in every release."""
    digest = hashlib.sha256(f"{SEED}-{match_id}".encode()).digest()
    return digest[0] % 2 == 0


def build_matches(raw):
    """Keep pre-match information only and hide the winner behind a random player order."""
    removed = {}
    walkover = raw["score"].astype(str).str.contains("W/O", regex=False)
    removed["walkover"] = int(walkover.sum())
    raw = raw[~walkover]

    match_id = raw["tourney_id"].astype(str) + "-" + raw["match_num"].astype(str)
    a_wins = match_id.map(player_a_is_winner)
    matches = pd.DataFrame({
        "match_id": match_id,
        "tourney_date": raw["tourney_date"],
        "surface": raw["surface"],
        "tourney_level": raw["tourney_level"],
        "best_of": raw["best_of"],
    })
    for name in PLAYER_FEATURES:
        matches[f"player_a_{name}"] = raw[f"winner_{name}"].where(a_wins, raw[f"loser_{name}"])
        matches[f"player_b_{name}"] = raw[f"loser_{name}"].where(a_wins, raw[f"winner_{name}"])
    matches[LABEL] = a_wins.astype("int64")
    matches["season"] = raw["season"]

    missing = matches[list(FEATURES)].isna().any(axis=1)
    removed["missing_value"] = int(missing.sum())
    matches = matches[~missing]

    valid = pd.Series(True, index=matches.index)
    for column, allowed in ALLOWED_VALUES.items():
        valid &= matches[column].isin(allowed)
    removed["out_of_scope_category"] = int((~valid).sum())
    matches = matches[valid]

    valid = pd.Series(True, index=matches.index)
    for column, (low, high) in RANGES.items():
        valid &= matches[column] >= low
        if high is not None:
            valid &= matches[column] <= high
    removed["out_of_range"] = int((~valid).sum())
    matches = matches[valid].astype({column: "int64" for column in INTEGER})

    if matches["match_id"].duplicated().any():
        raise ValueError("match_id must be unique")
    matches = matches.sort_values(["tourney_date", "match_id"], ignore_index=True)
    return matches.loc[:, [*COLUMNS, "season"]], removed


def split_release(matches, version):
    train_seasons, evaluation_season = RELEASES[version]
    train = matches[matches["season"].isin(train_seasons)]
    evaluation = matches[matches["season"] == evaluation_season]
    if train.empty or evaluation.empty:
        raise ValueError(f"{version}: empty training or evaluation split")
    if train["tourney_date"].max() >= evaluation["tourney_date"].min():
        raise ValueError(f"{version}: training matches must precede the evaluation season")
    return train.loc[:, list(COLUMNS)], evaluation.loc[:, list(COLUMNS)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, choices=sorted(RELEASES))
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    train_seasons, evaluation_season = RELEASES[args.version]
    raw = load_raw(args.raw_dir, (*train_seasons, evaluation_season))
    matches, removed = build_matches(raw)
    train, evaluation = split_release(matches, args.version)

    target = args.output_dir / args.version
    target.mkdir(parents=True, exist_ok=True)
    train.to_csv(target / "train.csv", index=False)
    evaluation.to_csv(target / "evaluation.csv", index=False)
    summary = {
        "version": args.version,
        "source": f"{SOURCE_REPO}@{SOURCE_COMMIT}",
        "raw_rows": len(raw),
        "removed": removed,
        "train_rows": len(train),
        "train_seasons": f"{train_seasons[0]}-{train_seasons[-1]}",
        "train_last_date": int(train["tourney_date"].max()),
        "evaluation_rows": len(evaluation),
        "evaluation_season": evaluation_season,
        "evaluation_first_date": int(evaluation["tourney_date"].min()),
        "train_a_wins_rate": round(float(train[LABEL].mean()), 4),
        "evaluation_a_wins_rate": round(float(evaluation[LABEL].mean()), 4),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
