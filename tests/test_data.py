import pandas as pd
import pytest

from ml.data import DataError, dvc_hash, read_split, validate_release
from tests.synthetic import write_release


@pytest.fixture
def release(tmp_path):
    return write_release(tmp_path, "v1", driver="rank")


def load(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def test_valid_release_passes_and_gets_typed(release):
    train, evaluation = validate_release(release)
    assert len(train) == 720 and len(evaluation) == 120
    assert train["player_a_rank"].dtype == "int64"
    assert train["player_a_age"].dtype == "float64"
    assert set(train["a_wins"]) == {0, 1}


def test_dvc_pointers_of_both_versions_are_readable():
    assert dvc_hash("v1").endswith(".dir")
    assert dvc_hash("v2").endswith(".dir")


@pytest.mark.parametrize(
    "column, value, message",
    [
        ("player_a_rank", "", "missing value"),
        ("surface", "Carpet", "outside the contract in surface"),
        ("tourney_level", "D", "outside the contract in tourney_level"),
        ("best_of", "4", "outside the contract in best_of"),
        ("player_b_hand", "U", "outside the contract in player_b_hand"),
        ("player_a_rank", "abc", "non-numeric value in player_a_rank"),
        ("player_a_rank", "12.5", "non-integer value in player_a_rank"),
        ("player_b_ht", "3", "out of range in player_b_ht"),
        ("player_a_age", "70.0", "out of range in player_a_age"),
        ("player_a_rank_points", "0", "out of range in player_a_rank_points"),
        ("tourney_date", "20221345", "YYYYMMDD"),
        ("a_wins", "2", "a_wins must be 0 or 1"),
    ],
)
def test_invalid_value_is_rejected(release, column, value, message):
    frame = load(release["train"])
    frame.loc[0, column] = value
    frame.to_csv(release["train"], index=False)
    with pytest.raises(DataError, match=message):
        read_split(release["train"])


@pytest.mark.parametrize("change", ["leaky_column", "missing_column", "reordered"])
def test_schema_must_match_the_contract_exactly(release, change):
    frame = load(release["train"])
    if change == "leaky_column":
        frame["winner_name"] = "x"
    elif change == "missing_column":
        frame = frame.drop(columns="player_b_ht")
    else:
        frame = frame[list(reversed(frame.columns))]
    frame.to_csv(release["train"], index=False)
    with pytest.raises(DataError, match="wrong CSV schema"):
        read_split(release["train"])


def test_label_needs_both_classes(release):
    frame = load(release["train"])
    frame["a_wins"] = "1"
    frame.to_csv(release["train"], index=False)
    with pytest.raises(DataError, match="both classes"):
        read_split(release["train"])


def test_duplicate_match_is_rejected(release):
    frame = load(release["train"])
    frame.loc[1, "match_id"] = frame.loc[0, "match_id"]
    frame.to_csv(release["train"], index=False)
    with pytest.raises(DataError, match="unique"):
        read_split(release["train"])


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(DataError, match="unreadable"):
        read_split(tmp_path / "missing.csv")


def test_evaluation_must_follow_training(release):
    frame = load(release["evaluation"])
    frame["tourney_date"] = "20210601"
    frame.to_csv(release["evaluation"], index=False)
    with pytest.raises(DataError, match="precede"):
        validate_release(release)


def test_a_match_cannot_be_in_both_splits(release):
    frame = load(release["evaluation"])
    frame.loc[0, "match_id"] = load(release["train"]).loc[0, "match_id"]
    frame.to_csv(release["evaluation"], index=False)
    with pytest.raises(DataError, match="both the training and the evaluation"):
        validate_release(release)
