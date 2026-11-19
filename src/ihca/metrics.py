"""Discrimination metrics with matched-group bootstrap intervals."""

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def validate_scores(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != scores.shape or not len(y):
        raise ValueError("Labels and scores must be equal-length nonempty vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Finite scores and binary labels are required")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Model scores must lie in [0, 1]")
    return y.astype(int), scores


def youden_threshold(y, scores):
    y, scores = validate_scores(y, scores)
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection requires both classes in inner validation")
    fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
    keep = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    thresholds, statistic = thresholds[keep], (tpr - fpr)[keep]
    # roc_curve orders thresholds from high to low; the first maximum is deterministic.
    return float(thresholds[np.flatnonzero(np.isclose(statistic, statistic.max(), rtol=0, atol=1e-12))[0]])


def binary_metrics(y, scores, threshold=0.5, predictions=None):
    y, scores = validate_scores(y, scores)
    predicted = scores >= threshold if predictions is None else np.asarray(predictions, dtype=bool)
    if predicted.shape != y.shape:
        raise ValueError("Predictions must match the label vector")
    tp = int(((y == 1) & predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    both = len(np.unique(y)) == 2
    precision, recall, _ = precision_recall_curve(y, scores) if both else (None, None, None)
    return {
        "n": len(y), "cases": int(y.sum()), "case_fraction": float(y.mean()),
        "auroc": float(roc_auc_score(y, scores)) if both else None,
        "auprc": float(auc(recall, precision)) if both else None,
        "average_precision": float(average_precision_score(y, scores)) if both else None,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def group_draws(groups, repeats, seed):
    """Yield record multiplicities for draws of whole connected patient/pair groups."""
    _, membership = np.unique(groups, return_inverse=True)
    count = int(membership.max()) + 1
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        counts = np.bincount(rng.integers(count, size=count), minlength=count)
        yield counts[membership]


class RankedOutcomes:
    """Sort thresholds once and evaluate weighted curves at distinct observed scores."""

    def __init__(self, y, scores):
        y, scores = validate_scores(y, scores)
        self.order = np.argsort(-scores, kind="stable")
        ranked = scores[self.order]
        self.starts = np.r_[0, 1 + np.flatnonzero(ranked[1:] != ranked[:-1])]
        self.positive = y[self.order]

    def areas(self, weights):
        ranked = weights[self.order]
        total = np.add.reduceat(ranked, self.starts)
        observed = total > 0
        pos = np.add.reduceat(ranked * self.positive, self.starts)[observed]
        total = total[observed]
        neg = total - pos
        if not pos.sum() or not neg.sum():
            return None
        tp, fp = pos.cumsum(dtype=float), neg.cumsum(dtype=float)
        precision = np.r_[1., tp / (tp + fp)]
        roc = np.sum(neg * (tp - 0.5 * pos)) / (tp[-1] * fp[-1])
        pr = np.sum(pos * (precision[:-1] + precision[1:])) / (2 * tp[-1])
        return float(roc), float(pr)


def bootstrap_intervals(y, scores, predictions, groups, repeats=2000, seed=42):
    y, scores = validate_scores(y, scores)
    groups = np.asarray(groups)
    if groups.shape != y.shape:
        raise ValueError("Bootstrap groups must match the label vector")
    predictions = np.asarray(predictions, dtype=bool)
    if predictions.shape != y.shape:
        raise ValueError("Predictions must match the label vector")
    ranking = RankedOutcomes(y, scores)
    true_positive = (y == 1) & predictions
    true_negative = (y == 0) & ~predictions
    keys = ("auroc", "auprc", "sensitivity", "specificity", "f1")
    values = {k: [] for k in keys}
    for weights in group_draws(groups, repeats, seed):
        areas = ranking.areas(weights)
        if areas is None:
            continue
        cases = weights @ y
        controls = weights.sum() - cases
        tp, tn = weights[true_positive].sum(), weights[true_negative].sum()
        results = (*areas, tp / cases, tn / controls, 2 * tp / (cases + tp + controls - tn))
        for key, value in zip(keys, results):
            values[key].append(value)
    return {"requested_resamples": repeats, "valid_resamples": len(values["auroc"]),
            "ci_95": {k: np.quantile(v, [0.025, 0.975]).tolist() if v else None for k, v in values.items()}}


def prevalence_predictive_values(sensitivity, specificity, prevalence):
    p = np.asarray(prevalence, dtype=float)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("Prevalence must lie strictly between 0 and 1")
    if not 0 <= sensitivity <= 1 or not 0 <= specificity <= 1:
        raise ValueError("Sensitivity and specificity must lie in [0, 1]")
    ppv_den = sensitivity * p + (1 - specificity) * (1 - p)
    npv_den = specificity * (1 - p) + (1 - sensitivity) * p
    ppv = np.divide(sensitivity * p, ppv_den, out=np.full_like(p, np.nan), where=ppv_den != 0)
    npv = np.divide(specificity * (1 - p), npv_den, out=np.full_like(p, np.nan), where=npv_den != 0)
    return ppv, npv
