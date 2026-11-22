import pandas as pd
import pytest

from ml.contract import COLUMNS, LABEL
from ml.prepare_data import build_matches, split_release


def raw_match(match_num, season=2021, **overrides):
    row = {
        "tourney_id": f"{season}-0001", "match_num": match_num,
        "tourney_date": season * 10000 + 301, "season": season,
        "surface": "Hard", "tourney_level": "A", "best_of": 3, "score": "6-4 6-4",
        "winner_rank": 5.0, "winner_rank_points": 5000.0, "winner_age": 25.1,
        "winner_ht": 188.0, "winner_hand": "R",
        "loser_rank": 80.0, "loser_rank_points": 700.0, "loser_age": 30.4,
        "loser_ht": 180.0, "loser_hand": "L",
        # Match statistics that must never reach the prepared data.
        "minutes": 95, "w_ace": 10, "l_ace": 2,
    }
    row.update(overrides)
    return row


def raw_frame(rows):
    return pd.DataFrame(rows)


def test_output_has_contract_columns_and_no_winner_loser_or_match_statistics():
    matches, _ = build_matches(raw_frame([raw_match(n) for n in range(1, 11)]))
    train = matches.drop(columns="season")
    assert tuple(train.columns) == COLUMNS
    assert not [c for c in matches.columns if c.startswith(("winner_", "loser_", "w_", "l_"))]
    assert "minutes" not in matches.columns and "score" not in matches.columns


def test_label_matches_the_player_order():
    matches, _ = build_matches(raw_frame([raw_match(n) for n in range(1, 41)]))
    # The winner is ranked 5 in every fixture row.
    assert (matches.loc[matches[LABEL] == 1, "player_a_rank"] == 5).all()
    assert (matches.loc[matches[LABEL] == 0, "player_b_rank"] == 5).all()
    assert set(matches[LABEL]) == {0, 1}


def test_player_order_is_deterministic_and_independent_of_other_rows():
    rows = [raw_match(n) for n in range(1, 21)]
    full, _ = build_matches(raw_frame(rows))
    subset, _ = build_matches(raw_frame(rows[5:12]))
    merged = subset.merge(full, on="match_id", suffixes=("_subset", "_full"))
    assert len(merged) == 7
    assert (merged[f"{LABEL}_subset"] == merged[f"{LABEL}_full"]).all()


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"score": "W/O"}, "walkover"),
        ({"winner_rank": None}, "missing_value"),
        ({"loser_ht": None}, "missing_value"),
        ({"tourney_level": "D"}, "out_of_scope_category"),
        ({"surface": "Carpet"}, "out_of_scope_category"),
        ({"winner_hand": "U"}, "out_of_scope_category"),
        ({"best_of": 1}, "out_of_scope_category"),
        ({"loser_ht": 3.0}, "out_of_range"),
        ({"winner_age": 70.0}, "out_of_range"),
        ({"winner_rank_points": 0.0}, "out_of_range"),
    ],
)
def test_invalid_rows_are_removed_and_counted(overrides, reason):
    matches, removed = build_matches(raw_frame([raw_match(1), raw_match(2, **overrides)]))
    assert list(matches["match_id"]) == ["2021-0001-1"]
    assert removed[reason] == 1


def test_split_keeps_evaluation_season_after_training():
    rows = [raw_match(n, season=s) for s in range(2016, 2024) for n in range(1, 4)]
    matches, _ = build_matches(raw_frame(rows))
    train, evaluation = split_release(matches, "v1")
    assert set(train["tourney_date"] // 10000) == set(range(2016, 2022))
    assert set(evaluation["tourney_date"] // 10000) == {2022}
    train, evaluation = split_release(matches, "v2")
    assert train["tourney_date"].max() < evaluation["tourney_date"].min()
    assert set(evaluation["tourney_date"] // 10000) == {2023}


def test_split_rejects_overlapping_dates():
    rows = [raw_match(n, season=s) for s in range(2016, 2023) for n in range(1, 3)]
    # A 2021-season tournament dated after the start of the 2022 season.
    rows.append(raw_match(9, season=2021, tourney_date=20220301))
    matches, _ = build_matches(raw_frame(rows))
    with pytest.raises(ValueError, match="precede"):
        split_release(matches, "v1")
