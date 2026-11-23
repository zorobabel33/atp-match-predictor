"""Compare model families on training seasons only, before any held-out evaluation.

Fits on v1 seasons 2016-2020 and validates on 2021. The 2022 and 2023 held-out seasons are not read.
"""

import os
import sys

import mlflow
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.contract import CATEGORICAL, NUMERIC
from ml.data import dvc_hash, fetch, read_split
from ml.model import build_model, evaluate, features, labels, rank_reference_auc
from ml.pipeline import code_version
from ml.registry import configure_tracking

EXPERIMENT = "atp-model-selection"
VALIDATION_START = 20210101


def gradient_boosting():
    preprocess = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORICAL)),
        ("passthrough", list(NUMERIC)),
    )
    classifier = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=42,
    )
    return make_pipeline(preprocess, classifier)


def logistic(C):
    return lambda: build_model(C=C, max_iter=1000)


CANDIDATES = {
    "logreg-C0.1": (logistic(0.1), {"model": "LogisticRegression", "C": 0.1}),
    "logreg-C1": (logistic(1.0), {"model": "LogisticRegression", "C": 1.0}),
    "logreg-C10": (logistic(10.0), {"model": "LogisticRegression", "C": 10.0}),
    "hist-gradient-boosting": (gradient_boosting, {
        "model": "HistGradientBoostingClassifier", "max_iter": 200, "learning_rate": 0.05,
        "max_leaf_nodes": 15,
    }),
}


def main():
    configure_tracking()
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", EXPERIMENT))
    train = read_split(fetch("v1")["train"])
    fit = train[train["tourney_date"] < VALIDATION_START]
    validation = train[train["tourney_date"] >= VALIDATION_START]
    sha, dirty = code_version()
    common = {"dataset_version": "v1", "dvc_md5": dvc_hash("v1"), "fit_seasons": "2016-2020",
              "validation_season": "2021"}
    tags = {"mlflow.source.git.commit": sha, "git_dirty": str(dirty).lower()}

    with mlflow.start_run(run_name="rank-reference"):
        mlflow.set_tags(tags)
        mlflow.log_params({**common, "model": "better ranked player wins"})
        mlflow.log_metric("validation_roc_auc", rank_reference_auc(validation))

    for run_name, (factory, params) in CANDIDATES.items():
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(tags)
            mlflow.log_params({**common, **params})
            model = factory().fit(features(fit), labels(fit))
            scores = evaluate(model, validation)
            mlflow.log_metrics({f"validation_{name}": value for name, value in scores.items()})
            print(run_name, {name: round(value, 4) for name, value in scores.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
