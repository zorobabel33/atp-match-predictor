"""Data contract shared by data preparation, training, validation and the web app."""

# Only information known before the match starts.
MATCH_FEATURES = ("surface", "tourney_level", "best_of")
PLAYER_FEATURES = ("rank", "rank_points", "age", "ht", "hand")
PLAYER_A_FEATURES = tuple(f"player_a_{name}" for name in PLAYER_FEATURES)
PLAYER_B_FEATURES = tuple(f"player_b_{name}" for name in PLAYER_FEATURES)
FEATURES = (*MATCH_FEATURES, *PLAYER_A_FEATURES, *PLAYER_B_FEATURES)
CATEGORICAL = ("surface", "tourney_level", "player_a_hand", "player_b_hand")
NUMERIC = tuple(column for column in FEATURES if column not in CATEGORICAL)

# Davis Cup (level D) is a team event with many incomplete rows, so it is out of scope.
ALLOWED_VALUES = {
    "surface": ("Hard", "Clay", "Grass"),
    "tourney_level": ("G", "M", "A", "F"),
    "best_of": (3, 5),
    "player_a_hand": ("R", "L"),
    "player_b_hand": ("R", "L"),
}
# Inclusive bounds, None means no upper bound.
_PLAYER_RANGES = {"rank": (1, None), "rank_points": (1, None), "age": (14, 50), "ht": (150, 220)}
RANGES = {
    f"player_{side}_{name}": bounds for side in "ab" for name, bounds in _PLAYER_RANGES.items()
}
INTEGER = ("best_of", *(c for c in RANGES if not c.endswith("_age")))

LABEL = "a_wins"
# match_id identifies a row, tourney_date orders train before evaluation.
COLUMNS = ("match_id", "tourney_date", *FEATURES, LABEL)

# Raw tennis_atp columns that reveal the winner or are only known after the match.
LEAKY_PREFIXES = ("winner_", "loser_", "w_", "l_")
LEAKY_COLUMNS = ("score", "minutes")
