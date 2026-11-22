"""Turn decision.json into GitHub Actions step outputs and a run summary."""

import json
import os
import sys
from pathlib import Path

SUMMARY_FIELDS = (
    "operation", "dataset_version", "validation", "retrained", "retraining_trigger", "decision",
    "reason", "champion_before", "champion_after", "run_id", "error",
)


def outputs(decision):
    changed = (decision.get("decision") != "failed"
               and decision.get("champion_before") != decision.get("champion_after"))
    return {
        "champion_changed": str(changed).lower(),
        "champion_version": decision.get("champion_after") or "",
    }


def summary(decision):
    rows = [f"| {field} | {'-' if decision.get(field) is None else decision[field]} |"
            for field in SUMMARY_FIELDS]
    return "\n".join(["### CT decision", "", "| Field | Value |", "|---|---|", *rows, ""])


def main(path):
    decision_file = Path(path)
    decision = json.loads(decision_file.read_text()) if decision_file.exists() else {}
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
        for key, value in outputs(decision).items():
            handle.write(f"{key}={value}\n")
    if decision:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write(summary(decision))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
