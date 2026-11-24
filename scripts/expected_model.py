"""Write the model metadata that a running service must report, read from the MLflow registry."""

import argparse
import json
import sys
from pathlib import Path

from ml.registry import configure_tracking, get_champion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        client, name = configure_tracking()
        version = get_champion(client, name)
    except Exception as exc:  # noqa: BLE001 - report the type only, never URLs or credentials
        print(f"Could not read the champion: {type(exc).__name__}", file=sys.stderr)
        return 1
    if version is None:
        print(f"No approved champion for {name}", file=sys.stderr)
        return 1
    expected = {
        "model_name": name,
        "model_version": str(version.version),
        "run_id": version.run_id,
        "dataset_version": version.tags.get("dataset_version"),
    }
    args.output.write_text(json.dumps(expected, indent=2) + "\n")
    print(f"Expected {name} version {expected['model_version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
