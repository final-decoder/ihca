import numpy as np
import pytest

from ihca.metrics import binary_metrics, bootstrap_intervals, prevalence_predictive_values, youden_threshold


def test_auc_and_threshold_against_rank_counts():
    y, scores = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    threshold = youden_threshold(y, scores)
    assert threshold == 0.8
    result = binary_metrics(y, scores, threshold)
    assert result["auroc"] == 0.75
    assert result["sensitivity"] == 0.5
    assert result["specificity"] == 1
    assert result["f1"] == pytest.approx(2 / 3)


def test_oof_operating_points_use_per_record_decisions():
    result = binary_metrics([0, 1, 0, 1], [.8, .9, .3, .4], predictions=[0, 1, 0, 1])
    assert result["sensitivity"] == result["specificity"] == 1


def test_pair_bootstrap_keeps_both_classes():
    y, scores, groups = np.tile([0, 1], 4), np.tile([0.1, 0.9], 4), np.repeat(np.arange(4), 2)
    intervals = bootstrap_intervals(y, scores, scores >= 0.5, groups, 30, 42)
    assert intervals["valid_resamples"] == 30
    assert intervals["ci_95"]["auroc"] == [1.0, 1.0]


def test_prevalence_predictive_values_by_contingency_counts():
    ppv, npv = prevalence_predictive_values(.8, .9, [.1])
    assert ppv[0] == pytest.approx(8 / 17)
    assert npv[0] == pytest.approx(81 / 83)


def test_threshold_selection_rejects_single_class():
    with pytest.raises(ValueError, match="both classes"):
        youden_threshold([0, 0], [.1, .4])
