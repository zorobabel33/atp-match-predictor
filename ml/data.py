"""Fetch a DVC dataset release, record its identity and validate the data contract."""

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml.contract import ALLOWED_VALUES, COLUMNS, INTEGER, LABEL, RANGES

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
VERSIONS = ("v1", "v2")
SPLITS = ("train", "evaluation")
NUMERIC_COLUMNS = ("tourney_date", *dict.fromkeys((*INTEGER, *RANGES)), LABEL)
INTEGER_COLUMNS = ("tourney_date", *INTEGER, LABEL)


class DataError(ValueError):
    """A release failed acquisition or validation."""


def dvc_pointer(version):
    return DATA_ROOT / f"{version}.dvc"


def dvc_hash(version):
    try:
        return yaml.safe_load(dvc_pointer(version).read_text())["outs"][0]["md5"]
    except (OSError, KeyError, IndexError, TypeError, yaml.YAMLError):
        raise DataError(f"Missing or invalid DVC pointer for {version}") from None


def fetch(version):
    """Pull the release from the DVC remote. DATA_DIR replaces the remote in tests."""
    if version not in VERSIONS:
        raise DataError(f"Unknown dataset version {version}")
    local = os.getenv("DATA_DIR")
    if local:
        folder = Path(local) / version
    else:
        env = {**os.environ, "DVC_NO_ANALYTICS": "true"}
        command = [sys.executable, "-m", "dvc", "pull", "-q", str(dvc_pointer(version))]
        try:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        except (OSError, subprocess.CalledProcessError):
            raise DataError(f"dvc pull failed for {version}") from None
        folder = DATA_ROOT / version
    return {split: folder / f"{split}.csv" for split in SPLITS}


def _validate_values(frame, name):
    if frame.empty or frame.isna().any().any():
        raise DataError(f"{name}: empty dataset or missing value")

    for column, allowed in ALLOWED_VALUES.items():
        if not frame[column].isin([str(value) for value in allowed]).all():
            raise DataError(f"{name}: value outside the contract in {column}")

    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values).all():
            raise DataError(f"{name}: non-numeric value in {column}")
        integer = column in INTEGER_COLUMNS
        if integer and (values % 1 != 0).any():
            raise DataError(f"{name}: non-integer value in {column}")
        if column in RANGES:
            low, high = RANGES[column]
            if (values < low).any() or (high is not None and (values > high).any()):
                raise DataError(f"{name}: value out of range in {column}")
        frame[column] = values.astype("int64" if integer else "float64")

    dates = pd.to_datetime(frame["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    if dates.isna().any():
        raise DataError(f"{name}: tourney_date must be a YYYYMMDD date")
    if not frame[LABEL].isin([0, 1]).all() or frame[LABEL].nunique() != 2:
        raise DataError(f"{name}: {LABEL} must be 0 or 1 and contain both classes")
    if frame["match_id"].duplicated().any():
        raise DataError(f"{name}: match_id must be unique")
    return frame


def read_split(path):
    """Require the published header, then normalize types before scikit-learn sees the data."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = csv.reader(handle, strict=True)
            if tuple(next(rows, [])) != COLUMNS:
                raise DataError(f"{path.name}: wrong CSV schema")
            if any(len(row) != len(COLUMNS) for row in rows):
                raise DataError(f"{path.name}: inconsistent number of CSV fields")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    except (OSError, UnicodeError, csv.Error, pd.errors.ParserError):
        raise DataError(f"{path.name}: unreadable CSV") from None
    return _validate_values(frame, path.name)


def validate_release(paths):
    train = read_split(paths["train"])
    evaluation = read_split(paths["evaluation"])
    if train["tourney_date"].max() >= evaluation["tourney_date"].min():
        raise DataError("Training matches must precede the evaluation window")
    if set(train["match_id"]) & set(evaluation["match_id"]):
        raise DataError("A match appears in both the training and the evaluation split")
    return train, evaluation
