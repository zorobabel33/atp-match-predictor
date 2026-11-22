"""Small contract-compliant releases for tests, so CI never needs Azure or DagsHub."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml.contract import COLUMNS


def make_split(seasons, rows_per_season, driver, seed):
    """Matches whose winner depends on one feature: "rank" or "age"."""
    rng = np.random.default_rng(seed)
    n = len(seasons) * rows_per_season
    season = np.repeat(seasons, rows_per_season)
    index = np.tile(np.arange(rows_per_season), len(seasons))
    level = rng.choice(["G", "M", "A", "F"], n)
    frame = pd.DataFrame({
        "match_id": [f"{s}-{i}" for s, i in zip(season, index, strict=True)],
        "tourney_date": season * 10000 + (1 + index % 11) * 100 + 1,
        "surface": rng.choice(["Hard", "Clay", "Grass"], n),
        "tourney_level": level,
        "best_of": np.where(level == "G", 5, 3),
    })
    for side in "ab":
        frame[f"player_{side}_rank"] = rng.integers(1, 300, n)
        frame[f"player_{side}_rank_points"] = rng.integers(1, 10000, n)
        frame[f"player_{side}_age"] = rng.uniform(18, 35, n).round(1)
        frame[f"player_{side}_ht"] = rng.integers(170, 205, n)
        frame[f"player_{side}_hand"] = rng.choice(["R", "L"], n)
    if driver == "rank":
        signal = (frame["player_b_rank"] - frame["player_a_rank"]) / 40
    else:
        signal = (frame["player_b_age"] - frame["player_a_age"]) / 2
    frame["a_wins"] = (rng.random(n) < 1 / (1 + np.exp(-signal))).astype(int)
    return frame.loc[:, list(COLUMNS)]


def write_release(root, version, driver, seed=0, rows_per_season=120):
    """v1: train 2016-2021, evaluate 2022. v2: train 2016-2022, evaluate 2023."""
    last_train = {"v1": 2021, "v2": 2022}[version]
    folder = root / version
    folder.mkdir(parents=True, exist_ok=True)
    train = make_split(list(range(2016, last_train + 1)), rows_per_season, driver, seed)
    evaluation = make_split([last_train + 1], rows_per_season, driver, seed + 1)
    train.to_csv(folder / "train.csv", index=False)
    evaluation.to_csv(folder / "evaluation.csv", index=False)
    return {"train": folder / "train.csv", "evaluation": folder / "evaluation.csv"}


if __name__ == "__main__":
    # Controlled demonstration data: v1 winners depend on age, v2 winners on rank (concept drift),
    # so the v1 champion loses its edge on v2 and a retrained candidate is promoted.
    target = Path(sys.argv[1])
    write_release(target, "v1", driver="age")
    write_release(target, "v2", driver="rank")
    print(f"Synthetic v1 and v2 written to {target}")
