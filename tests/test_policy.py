import pytest

from ml.policy import (
    MAX_ROC_AUC_DROP,
    MIN_ROC_AUC,
    MIN_ROC_AUC_IMPROVEMENT,
    passes_promotion,
    rejection_reason,
    retraining_trigger,
)


def test_gates_are_ordered_for_a_higher_is_better_metric():
    assert 0.5 < MIN_ROC_AUC < 1
    assert 0 < MIN_ROC_AUC_IMPROVEMENT < MAX_ROC_AUC_DROP


@pytest.mark.parametrize(
    "candidate, champion, expected",
    [
        (0.70, None, True),                       # bootstrap above the floor
        (MIN_ROC_AUC, None, True),                # exactly on the floor
        (0.67, None, False),                      # bootstrap below the floor
        (0.712, 0.710, True),                     # enough improvement
        (0.7105, 0.710, False),                   # improvement below the gate
        (0.705, 0.710, False),                    # worse than the champion
        (0.71, 0.71, False),                      # equal is not an improvement
        (0.675, 0.600, False),                    # big improvement but below the floor
    ],
)
def test_promotion_rule(candidate, champion, expected):
    assert passes_promotion(candidate, champion) is expected
    assert (rejection_reason(candidate, champion) is None) is expected


def test_rejection_reason_names_the_failed_gate():
    assert "below the floor" in rejection_reason(0.60)
    assert "improvement" in rejection_reason(0.7105, 0.710)


@pytest.mark.parametrize(
    "champion_auc, approval_auc, champion_dataset, release, expected",
    [
        (0.71, 0.71, "v1", "v1", None),                        # same data, stable: keep
        (0.70, 0.71, "v1", "v1", None),                        # small drop: keep
        (0.71, 0.71, "v1", "v2", "new_dataset_version"),
        (0.67, 0.69, "v2", "v2", "below_quality_floor"),
        (0.69, 0.715, "v2", "v2", "performance_drop"),
    ],
)
def test_retraining_rule(champion_auc, approval_auc, champion_dataset, release, expected):
    assert retraining_trigger(champion_auc, approval_auc, champion_dataset, release) == expected
