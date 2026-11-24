"""Quality gates on ROC-AUC (higher is better), fixed before evaluating v1 or v2.

Calibrated on training seasons only (fit 2016-2020, validate 2021), see ml/README_experiments.md.
"""

# "The better ranked player wins" reaches 0.673 on the validation season: a model must beat it.
MIN_ROC_AUC = 0.68
# About twice the bootstrap standard deviation (0.0005) of a paired ROC-AUC difference.
MIN_ROC_AUC_IMPROVEMENT = 0.001
MAX_ROC_AUC_DROP = 0.02
# Monitoring signal only: investigate drift; promotion still uses the existing quality gates.
DRIFT_THRESHOLD = 0.20


def retraining_trigger(champion_auc, approval_auc, champion_dataset, dataset_version):
    """Return why the champion must be retrained, or None to keep it without training."""
    if dataset_version != champion_dataset:
        return "new_dataset_version"
    if champion_auc < MIN_ROC_AUC:
        return "below_quality_floor"
    if approval_auc - champion_auc >= MAX_ROC_AUC_DROP:
        return "performance_drop"
    return None


def rejection_reason(candidate_auc, champion_auc=None):
    """Return why a candidate fails its gates, or None when it can be promoted."""
    if candidate_auc < MIN_ROC_AUC:
        return f"candidate roc_auc {candidate_auc:.4f} is below the floor {MIN_ROC_AUC}"
    if champion_auc is not None and candidate_auc - champion_auc < MIN_ROC_AUC_IMPROVEMENT:
        return (f"improvement {candidate_auc - champion_auc:+.4f} over the champion is below "
                f"{MIN_ROC_AUC_IMPROVEMENT}")
    return None


def passes_promotion(candidate_auc, champion_auc=None):
    return rejection_reason(candidate_auc, champion_auc) is None
