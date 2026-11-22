"""Score CSV rows with the champion selected at execution time."""

import argparse
import json

import pandas as pd

from ml.model import features
from ml.registry import configure_tracking, get_champion, load_version


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive row count")
    client, name = configure_tracking()
    champion = get_champion(client, name)
    if champion is None:
        parser.error("no approved champion is deployed")
    frame = pd.read_csv(args.input, nrows=args.limit)
    if frame.empty:
        parser.error("the prediction CSV has no observations")
    probabilities = load_version(champion).predict_proba(features(frame))[:, 1]
    result = {"model_name": name, "model_version": str(champion.version),
              "probability_player_a_wins": probabilities.tolist()}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
