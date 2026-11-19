import numpy as np
from sklearn.metrics import roc_auc_score

from ihca.metrics import RankedOutcomes, binary_metrics, bootstrap_intervals


def test_weighted_group_draws_match_explicit_record_repetition():
    labels = np.array([0, 1, 1, 0, 1, 0, 1])
    scores = np.array([.8, .8, .4, .4, .9, .1, .1])
    groups = np.array(["x", "x", "a", "z", "a", "z", "b"])
    decisions = np.array([0, 1, 1, 1, 0, 0, 1], bool)
    draws, seed = 79, 22
    got = bootstrap_intervals(labels, scores, decisions, groups, draws, seed)
    ref = []
    rng = np.random.default_rng(seed)
    names = np.unique(groups)
    for _ in range(draws):
        picked = rng.integers(len(names), size=len(names))
        ix = np.concatenate([np.flatnonzero(groups == names[i]) for i in picked])
        if np.unique(labels[ix]).size == 2:
            ref.append(binary_metrics(labels[ix], scores[ix], predictions=decisions[ix]))
    assert 0 < got["valid_resamples"] == len(ref) < draws
    for metric, ends in got["ci_95"].items():
        np.testing.assert_allclose(ends, np.quantile([r[metric] for r in ref], [.025, .975]))


def test_tied_ranking_handles_skipped_leading_thresholds():
    y = np.array([0, 1, 0, 1, 0])
    score = np.array([.99, .8, .8, .3, .1])
    counts = np.array([0, 2, 1, 0, 3])
    roc, pr = RankedOutcomes(y, score).areas(counts)
    yy, ss = np.repeat(y, counts), np.repeat(score, counts)
    np.testing.assert_allclose([roc, pr],
                               [roc_auc_score(yy, ss), binary_metrics(yy, ss)["auprc"]])
