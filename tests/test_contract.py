from ml.contract import (
    CATEGORICAL,
    COLUMNS,
    FEATURES,
    LABEL,
    LEAKY_COLUMNS,
    LEAKY_PREFIXES,
    NUMERIC,
)


def test_features_exclude_winner_loser_and_match_statistics():
    for column in FEATURES:
        assert not column.startswith(LEAKY_PREFIXES), column
        assert column not in LEAKY_COLUMNS, column


def test_label_and_identifiers_are_not_features():
    assert LABEL not in FEATURES
    assert "match_id" not in FEATURES
    assert "tourney_date" not in FEATURES


def test_columns_are_unique_and_split_into_numeric_and_categorical():
    assert len(set(COLUMNS)) == len(COLUMNS)
    assert set(CATEGORICAL) | set(NUMERIC) == set(FEATURES)
    assert not set(CATEGORICAL) & set(NUMERIC)


def test_both_players_have_the_same_features():
    a_side = [column.removeprefix("player_a_") for column in FEATURES if "player_a_" in column]
    b_side = [column.removeprefix("player_b_") for column in FEATURES if "player_b_" in column]
    assert a_side == b_side
