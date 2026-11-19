"""Deterministic partitions that preserve pairs and repeated patients."""

import numpy as np


def outer_folds(groups, folds=5, seed=42):
    groups = np.asarray(groups)
    unique, counts = np.unique(groups, return_counts=True)
    if len(unique) < folds:
        raise ValueError("Not enough independent matched groups for the requested folds")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    order = order[np.argsort(-counts[order], kind="stable")]
    assignments, sizes = {}, np.zeros(folds, dtype=int)
    for j in order:
        fold = int(np.argmin(sizes))
        assignments[unique[j]] = fold
        sizes[fold] += counts[j]
    result = np.asarray([assignments[g] for g in groups])
    return [np.flatnonzero(result == fold) for fold in range(folds)]


def inner_split(available, groups, fraction=0.2, seed=42):
    available = np.asarray(available)
    unique = np.unique(groups[available])
    if len(unique) < 2:
        raise ValueError("At least two groups are required for fitting and inner validation")
    rng = np.random.default_rng(seed)
    n = min(len(unique) - 1, max(1, round(len(unique) * fraction)))
    selected = rng.permutation(unique)[:n]
    mask = np.isin(groups[available], selected)
    return available[~mask], available[mask]


def training_batches(length, batch_size, rng):
    """Merge a terminal singleton so BatchNorm can train without dropping records."""
    if length < 2:
        raise ValueError("Training requires at least two records")
    order = rng.permutation(length)
    batches = [order[i:i + batch_size].tolist() for i in range(0, length, batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return batches
