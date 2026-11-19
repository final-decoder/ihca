"""Group-preserving discrimination summaries and paired comparisons."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ihca.metrics import RankedOutcomes, binary_metrics, bootstrap_intervals, group_draws


def performance(frame, repeats, seed):
    scored = frame.dropna(subset=["score"])
    result = {
        "total": len(frame),
        "scored": len(scored),
        "unavailable": len(frame) - len(scored),
        "groups_scored": int(scored.group_id.nunique()),
        "cases": int(scored.label.sum()),
        "controls": int((scored.label == 0).sum()),
    }
    if scored.label.nunique() < 2:
        return {**result, "status": "both outcome classes required"}
    y = scored.label.to_numpy(dtype=int)
    scores = scored.score.to_numpy(dtype=float)
    decisions = scored.prediction.to_numpy(dtype=bool)
    metrics = binary_metrics(y, scores, predictions=decisions)
    if result["groups_scored"] < 2:
        result.update(metrics, status="point estimates only: one independent group", valid_resamples=0)
        for key in ("auroc", "auprc", "sensitivity", "specificity", "f1"):
            result[key + "_low"] = result[key + "_high"] = np.nan
        return result
    ci = bootstrap_intervals(y, scores, decisions, scored.group_id.to_numpy(), repeats, seed)
    result.update(metrics, status="estimated", valid_resamples=ci["valid_resamples"])
    for key, bounds in ci["ci_95"].items():
        result[key + "_low"], result[key + "_high"] = bounds if bounds else (np.nan, np.nan)
    return result


def paired_differences(frame, alternatives, repeats, seed):
    if repeats < 2:
        raise ValueError("At least two paired bootstrap draws are required")
    y = frame.label.to_numpy(dtype=int)
    groups = frame.group_id.to_numpy()
    if np.unique(groups).size < 2:
        raise ValueError("At least two independent groups are required for paired uncertainty estimates")
    full = frame.score.to_numpy()
    observed = {m: roc_auc_score(y, s) - roc_auc_score(y, full) for m, s in alternatives.items()}
    draws = {m: [] for m in alternatives}
    reference_ranking = RankedOutcomes(y, full)
    rankings = {name: RankedOutcomes(y, scores) for name, scores in alternatives.items()}
    for weights in group_draws(groups, repeats, seed):
        reference = reference_ranking.areas(weights)
        if reference is None:
            continue
        for name, ranking in rankings.items():
            draws[name].append(ranking.areas(weights)[0] - reference[0])
    rows = []
    for model, delta in observed.items():
        values = np.asarray(draws[model])
        if not len(values):
            raise ValueError("No valid paired bootstrap samples")
        low, high = np.quantile(values, [0.025, 0.975])
        # Two-sided bootstrap sign-tail test with a finite-sample correction.
        tail = min(np.sum(values <= 0), np.sum(values >= 0))
        p = min(1.0, 2 * (1 + tail) / (1 + len(values)))
        rows.append(
            {
                "model": model,
                "delta_auroc": delta,
                "delta_low": low,
                "delta_high": high,
                "p_two_sided": p,
                "valid_pairs_of_estimates": len(values),
            }
        )
    return pd.DataFrame(rows)
