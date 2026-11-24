"""Evaluate a dataset release or roll back the approved production model."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
from mlflow.exceptions import MlflowException

from ml import policy
from ml.data import ROOT, VERSIONS, DataError, dvc_hash, dvc_pointer, fetch, validate_release
from ml.model import (
    HYPERPARAMETERS,
    compare_distributions,
    evaluate,
    rank_reference_auc,
    train_model,
    training_profile,
)
from ml.registry import (
    APPROVAL_TAG,
    APPROVED,
    DEFAULT_EXPERIMENT,
    configure_tracking,
    get_champion,
    load_training_profile,
    load_version,
    move_champion,
    record_candidate,
    register_approved,
)


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def code_version():
    sha = os.getenv("GITHUB_SHA") or _git("rev-parse", "HEAD") or "unknown"
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return sha, dirty


def _log_scores(prefix, scores):
    mlflow.log_metrics({f"{prefix}_{name}": value for name, value in scores.items()})


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_release(client, name, release, workspace, result):
    # DATA_DIR replaces DVC for tests and controlled demonstrations: never log a DVC hash then.
    result["data_source"] = "local" if os.getenv("DATA_DIR") else "dvc"
    mlflow.set_tag("data_source", result["data_source"])
    if result["data_source"] == "dvc":
        result["dvc_md5"] = dvc_hash(release)
        mlflow.log_param("dvc_md5", result["dvc_md5"])
        shutil.copyfile(dvc_pointer(release), workspace / f"{release}.dvc")
    train, evaluation = validate_release(fetch(release))
    _write_json(workspace / "training-profile.json", training_profile(
        train, model_run_id=result["run_id"], dataset_version=release,
        dvc_md5=result.get("dvc_md5"), data_source=result["data_source"],
    ))
    result["validation"] = "passed"
    mlflow.set_tag("validation", "passed")
    mlflow.log_metrics({"train_rows": len(train), "evaluation_rows": len(evaluation)})
    mlflow.log_params({
        **HYPERPARAMETERS,
        "model": "LogisticRegression",
        "min_roc_auc": policy.MIN_ROC_AUC,
        "min_roc_auc_improvement": policy.MIN_ROC_AUC_IMPROVEMENT,
        "max_roc_auc_drop": policy.MAX_ROC_AUC_DROP,
        "train_last_date": int(train["tourney_date"].max()),
        "evaluation_first_date": int(evaluation["tourney_date"].min()),
    })
    result["reference_rank_roc_auc"] = rank_reference_auc(evaluation)
    mlflow.log_metric("reference_rank_roc_auc", result["reference_rank_roc_auc"])

    champion = get_champion(client, name)
    champion_version = str(champion.version) if champion else None
    result["champion_before"] = result["champion_after"] = champion_version
    if champion is None and release != "v1":
        raise ValueError("Run v1 to approve the initial champion before evaluating a later window")

    champion_auc = None
    if champion:
        try:
            drift = compare_distributions(load_training_profile(client, champion), evaluation)
            _write_json(workspace / "drift.json", {
                "reference_version": champion_version, "threshold": policy.DRIFT_THRESHOLD,
                "features": drift, "drifted": [n for n, v in drift.items()
                                               if v >= policy.DRIFT_THRESHOLD],
            })
            mlflow.log_metrics({f"drift_{name}": value for name, value in drift.items()})
        except (MlflowException, OSError, ValueError, KeyError):
            # Legacy champions can be backfilled without retraining or moving their alias.
            result["drift_reference"] = "unavailable"
        if int(champion.tags["training_last_date"]) >= evaluation["tourney_date"].min():
            # The data itself is valid: this is a release order error, not a validation failure.
            raise ValueError("This evaluation window is older than the champion's training data")
        scores = evaluate(load_version(champion), evaluation)
        _log_scores("champion", scores)
        result["champion_metrics"] = scores
        champion_auc = scores["roc_auc"]
        approval_auc = float(champion.tags["approval_roc_auc"])
        mlflow.log_metrics({"champion_approval_roc_auc": approval_auc,
                            "roc_auc_drop": approval_auc - champion_auc})
        trigger = policy.retraining_trigger(
            champion_auc, approval_auc, champion.tags["dataset_version"], release,
        )
        if trigger is None:
            result["decision"] = "retain_no_retraining"
            result["reason"] = "same dataset version, champion above the floor and stable"
            return
        result["retraining_trigger"] = trigger
    else:
        result["retraining_trigger"] = "initial_release"

    result["retrained"] = True
    estimator = train_model(train)
    scores = evaluate(estimator, evaluation)
    _log_scores("candidate", scores)
    result["candidate_metrics"] = scores
    model_uri = record_candidate(estimator, train, {
        "dataset_version": release, "data_source": result["data_source"],
        "dvc_md5": result.get("dvc_md5") or "none", "run_id": result["run_id"],
        "roc_auc": scores["roc_auc"],
    })
    result["candidate_model_uri"] = model_uri
    reason = policy.rejection_reason(scores["roc_auc"], champion_auc)
    if reason:
        result["decision"] = "retain_rejected" if champion else "bootstrap_rejected"
        result["reason"] = reason
        return

    target = register_approved(client, name, model_uri, {
        "dataset_version": release,
        "data_source": result["data_source"],
        "dvc_md5": result.get("dvc_md5") or "none",
        "training_last_date": int(train["tourney_date"].max()),
        "approval_roc_auc": scores["roc_auc"],
        "approval_run_id": result["run_id"],
        "git_sha": result["git_sha"],
    })
    result["candidate_version"] = target
    _move_champion(client, name, champion_version, target, result)
    result["decision"] = "promoted" if champion else "bootstrapped"


def roll_back(client, name, requested_version, result):
    champion = get_champion(client, name)
    if champion is None:
        raise ValueError("Approve an initial champion before requesting a rollback")
    prior = str(champion.version)
    result["champion_before"] = result["champion_after"] = prior
    result["rollback_version"] = requested_version
    version = client.get_model_version(name, requested_version)
    if version.tags.get(APPROVAL_TAG) != APPROVED:
        raise ValueError("Rollback requires a previously approved model version")
    if prior != requested_version:
        # Verify the stored artifact can still be loaded before exposing it to inference.
        load_version(version)
        _move_champion(client, name, prior, requested_version, result)
    result["champion_after"] = requested_version
    result["decision"] = "rolled_back" if prior != requested_version else "rollback_unchanged"


def _move_champion(client, name, expected, target, result):
    """Move the alias. If the move fails, record where the alias really points."""
    try:
        move_champion(client, name, expected, target)
    except (ValueError, MlflowException):
        # The verification can fail after the alias already moved.
        try:
            current = get_champion(client, name)
            result["champion_after"] = str(current.version) if current else None
        except (ValueError, MlflowException):
            pass
        raise
    result["champion_after"] = target


def execute(args):
    client, name = configure_tracking()
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT))
    release = args.dataset_version
    operation = "pipeline" if release else "rollback"
    output = Path(os.getenv("RUN_OUTPUT_DIR", "results"))
    output.mkdir(parents=True, exist_ok=True)
    sha, dirty = code_version()
    run_name = f"{operation}-{release or args.rollback}"
    with TemporaryDirectory(prefix="atp-results-") as tmp, \
            mlflow.start_run(run_name=run_name) as run:
        workspace = Path(tmp)
        result = {
            "operation": operation, "dataset_version": release, "run_id": run.info.run_id,
            "model_name": name, "git_sha": sha, "git_dirty": dirty,
            "validation": "pending" if release else "not_applicable",
            "retrained": False, "champion_before": None, "champion_after": None,
        }
        mlflow.set_tags({"operation": operation, "dataset_version": release or "not_applicable",
                         "model_name": name, "mlflow.source.git.commit": sha,
                         "git_dirty": str(dirty).lower()})
        if release:
            mlflow.log_param("dataset_version", release)
        failed = False
        try:
            if release:
                run_release(client, name, release, workspace, result)
            else:
                roll_back(client, name, args.rollback, result)
        except (ValueError, MlflowException, OSError, KeyError, RuntimeError) as exc:
            if isinstance(exc, DataError):
                result["validation"] = "failed"
            result["decision"] = "failed"
            # MLflow transport errors may contain endpoint URLs. Keep details at the source.
            if isinstance(exc, MlflowException):
                result["error"] = f"MLflow request failed ({exc.error_code})"
            else:
                result["error"] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            failed = True
        finally:
            mlflow.set_tags({
                "decision": result.get("decision", "failed"),
                "validation": result["validation"],
                "retrained": str(result["retrained"]).lower(),
                **{field: result.get(field) or "none"
                   for field in ("champion_before", "champion_after", "candidate_version",
                                 "retraining_trigger")},
            })
            if result.get("reason"):
                mlflow.set_tag("reason", result["reason"])
            _write_json(workspace / "decision.json", result)
            # decision.json and the DVC pointer go to MLflow and to the CT run artifact.
            for path in workspace.iterdir():
                shutil.copyfile(path, output / path.name)
                mlflow.log_artifact(str(path))
            if failed or result.get("decision") == "bootstrap_rejected":
                mlflow.end_run(status="FAILED")
            print(json.dumps(result, indent=2, sort_keys=True))
    return int(failed or result.get("decision") == "bootstrap_rejected")


def _version_number(value):
    if not value.isascii() or not value.isdecimal() or int(value) < 1:
        raise argparse.ArgumentTypeError("model versions are positive integers")
    return str(int(value))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dataset-version", choices=VERSIONS)
    action.add_argument("--rollback", type=_version_number, metavar="VERSION")
    try:
        return execute(parser.parse_args(argv))
    except (ValueError, MlflowException, OSError) as exc:
        message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        print(f"Pipeline error: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
