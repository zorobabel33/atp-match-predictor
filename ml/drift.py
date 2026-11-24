"""Training histograms and Population Stability Index shared by CT and online monitoring."""

import numpy as np

from ml.contract import ALLOWED_VALUES, CATEGORICAL, FEATURES


def training_profile(frame, **metadata):
    if frame.empty or frame.loc[:, list(FEATURES)].isna().any().any():
        raise ValueError("A reference needs complete training features")
    columns = {}
    for name in FEATURES:
        if name in CATEGORICAL:
            categories = list(ALLOWED_VALUES[name])
            counts = [(frame[name] == value).sum() for value in categories]
            columns[name] = {"kind": "categorical", "categories": categories,
                             "counts": [int(n) for n in counts] + [0]}
        else:
            cuts = np.unique(np.quantile(frame[name], np.arange(0.1, 1, 0.1))).tolist()
            counts, _ = np.histogram(frame[name], bins=[-np.inf, *cuts, np.inf])
            columns[name] = {"kind": "numeric", "cuts": cuts, "counts": counts.tolist()}
    return {"schema_version": 1, "rows": len(frame), "features": columns, **metadata}


def validate_profile(profile):
    if profile.get("schema_version") != 1 or set(profile.get("features", {})) != set(FEATURES):
        raise ValueError("Invalid reference schema")
    if not isinstance(profile.get("rows"), int) or profile["rows"] <= 0:
        raise ValueError("Invalid reference sample size")
    for name, column in profile["features"].items():
        categorical = name in CATEGORICAL
        if column.get("kind") != ("categorical" if categorical else "numeric"):
            raise ValueError("Invalid reference feature type")
        edges = column.get("categories" if categorical else "cuts", [])
        if categorical:
            if edges != list(ALLOWED_VALUES[name]):
                raise ValueError("Invalid reference categories")
        elif any(not np.isfinite(x) for x in edges) or edges != sorted(set(edges)):
            raise ValueError("Invalid reference histogram cuts")
        counts = column.get("counts", [])
        if (len(counts) != len(edges) + 1 or any(type(n) is not int or n < 0 for n in counts)
                or sum(counts) != profile["rows"]):
            raise ValueError("Invalid reference histogram counts")
    return profile


def compare_distributions(reference, frame):
    """PSI uses fixed training bins and 1e-4 smoothing; no labels or outcomes are required."""
    validate_profile(reference)
    if frame.empty or frame.loc[:, list(FEATURES)].isna().any().any():
        raise ValueError("Drift needs complete observations")
    result = {}
    for name, column in reference["features"].items():
        if column["kind"] == "categorical":
            categories = column["categories"]
            counts = [(frame[name] == value).sum() for value in categories]
            counts.append((~frame[name].isin(categories)).sum())
        else:
            counts, _ = np.histogram(frame[name], bins=[-np.inf, *column["cuts"], np.inf])
        expected = np.maximum(np.array(column["counts"], dtype=float) / reference["rows"], 1e-4)
        observed = np.maximum(np.array(counts, dtype=float) / len(frame), 1e-4)
        expected /= expected.sum()
        observed /= observed.sum()
        result[name] = float(np.sum((observed - expected) * np.log(observed / expected)))
    return result
