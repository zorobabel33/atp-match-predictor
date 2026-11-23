"""Data contract shared by data preparation, training, validation and the web app."""

# Only information known before the match starts.
MATCH_FEATURES = ("surface", "tourney_level", "best_of")
PLAYER_FEATURES = ("rank", "rank_points", "age", "ht", "hand")
PLAYER_A_FEATURES = tuple(f"player_a_{name}" for name in PLAYER_FEATURES)
PLAYER_B_FEATURES = tuple(f"player_b_{name}" for name in PLAYER_FEATURES)
FEATURES = (*MATCH_FEATURES, *PLAYER_A_FEATURES, *PLAYER_B_FEATURES)
CATEGORICAL = ("surface", "tourney_level", "player_a_hand", "player_b_hand")
NUMERIC = tuple(column for column in FEATURES if column not in CATEGORICAL)

LABEL = "a_wins"
# match_id identifies a row, tourney_date orders train before evaluation.
COLUMNS = ("match_id", "tourney_date", *FEATURES, LABEL)

# Raw tennis_atp columns that reveal the winner or are only known after the match.
LEAKY_PREFIXES = ("winner_", "loser_", "w_", "l_")
LEAKY_COLUMNS = ("score", "minutes")
