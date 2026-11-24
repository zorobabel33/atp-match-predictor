"""Classifier, metrics and the naive ranking reference."""

from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.contract import CATEGORICAL, FEATURES, LABEL, NUMERIC
from ml.drift import compare_distributions as compare_distributions
from ml.drift import training_profile as training_profile

# Chosen on training seasons only (fit 2016-2020, validate 2021). ROC-AUC barely depends on C.
HYPERPARAMETERS = {"C": 1.0, "max_iter": 1000}


def features(frame):
    return frame.loc[:, list(FEATURES)]


def labels(frame):
    return frame[LABEL]


def build_model(C, max_iter):
    # Encoders and scaler are fitted inside the pipeline, on training rows only.
    preprocess = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORICAL)),
        (StandardScaler(), list(NUMERIC)),
    )
    return make_pipeline(preprocess, LogisticRegression(C=C, max_iter=max_iter))


def train_model(train, params=HYPERPARAMETERS):
    return build_model(**params).fit(features(train), labels(train))


def evaluate(model, frame):
    expected = labels(frame)
    scores = model.predict_proba(features(frame))[:, 1]
    return {
        "roc_auc": float(roc_auc_score(expected, scores)),
        "log_loss": float(log_loss(expected, scores, labels=[0, 1])),
        "accuracy": float(accuracy_score(expected, scores >= 0.5)),
    }


def rank_reference_auc(frame):
    """ROC-AUC of the rule "the better ranked player wins", the bar a model has to clear."""
    return float(roc_auc_score(labels(frame), frame["player_b_rank"] - frame["player_a_rank"]))
