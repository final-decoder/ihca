"""Cohort descriptions, matched phenotypes, modality contrasts and incidence scenarios."""

from datetime import datetime
import json

import numpy as np
import pandas as pd

from ihca.metrics import binary_metrics, prevalence_predictive_values
from ihca.schema import CLINICAL_BINARY, CLINICAL_FEATURES, ECG_BINARY, ECG_FEATURES
from . import plots
from .scores import marker_flags, rule_scores
from .statistics import paired_differences, performance
from .study import fold_partitions, read_json, read_predictions


def interval_hours(frame):
    hours = []
    for ecg, event in zip(frame.ecg_time, frame.index_time):
        duration = (datetime.fromisoformat(event) - datetime.fromisoformat(ecg)).total_seconds() / 3600
        if duration <= 0:
            raise ValueError("Index time must follow ECG acquisition")
        hours.append(duration)
    return pd.Series(hours, index=frame.index)


def ablation(study, out):
    full = study.oof
    scores = {}
    rows = []
    for name in ("clinical", "structured", "waveform", "full"):
        path = study.model(name)
        if json.loads(read_json(path / "config.json"))["model"]["mode"] != name:
            raise ValueError(f"Expected a {name} model directory")
        parts = list(fold_partitions(path, study.dev))
        if len(parts) != len(study.folds) or any(
            any(not np.array_equal(np.sort(a), np.sort(b)) for a, b in zip(p[1:], q[1:]))
            for p, q in zip(parts, study.folds)
        ):
            raise ValueError("Ablations must share all fitting, validation and held-out partitions")
        frame = read_predictions(path / "oof_predictions.csv", study.dev)
        if not np.array_equal(frame.fold, full.fold):
            raise ValueError("Ablation folds must match by record")
        rows.append({"model": name, **performance(frame, study.repeats, study.seed)})
        if name != "full":
            scores[name] = frame.score.to_numpy()
    contrasts = paired_differences(full, scores, study.cfg["paired_bootstrap"], study.seed)
    pd.DataFrame(rows).merge(contrasts, on="model", how="left").to_csv(
        out / "modality_comparison.csv", index=False
    )
    by_fold = []
    for fold, group in full.groupby("fold"):
        by_fold.append(
            {
                "fold": fold,
                "pairs": group.pair_id.nunique(),
                "threshold": group.threshold.iloc[0],
                **performance(group, study.repeats, study.seed),
            }
        )
    fold_table = pd.DataFrame(by_fold)
    fold_table.to_csv(out / "fold_performance.csv", index=False)
    fold_table[["auroc", "auprc", "threshold", "sensitivity", "specificity", "f1"]].agg(
        ["mean", "std"]
    ).to_csv(out / "fold_summary.csv")


def group_summary(frame, values, binary=False):
    groups = [
        pd.to_numeric(values[frame.label == label]).dropna().to_numpy(dtype=float) for label in (1, 0)
    ]
    result = {}
    spreads, means = [], []
    for prefix, x in zip(("case", "control"), groups):
        mean = float(x.mean()) if len(x) else np.nan
        sd = float(x.std(ddof=1)) if len(x) > 1 else np.nan
        means.append(mean)
        spreads.append(mean * (1 - mean) if binary else sd**2)
        result[prefix + "_n"] = len(x)
        if binary:
            result.update({prefix + "_positive": int(x.sum()), prefix + "_percent": 100 * mean})
        else:
            quantiles = np.quantile(x, [0.25, 0.5, 0.75]) if len(x) else [np.nan] * 3
            result.update(
                {
                    prefix + "_mean": mean,
                    prefix + "_sd": sd,
                    prefix + "_q25": quantiles[0],
                    prefix + "_median": quantiles[1],
                    prefix + "_q75": quantiles[2],
                }
            )
    denom = np.sqrt(np.mean(spreads))
    result["abs_smd"] = (
        abs(means[0] - means[1]) / denom if denom > 0 else (0.0 if means[0] == means[1] else np.nan)
    )
    return result


def describe(study, out):
    rules = study.cfg["clinical_rules"]
    for cohort, frame in (("development", study.dm), ("external", study.em)):
        if frame.index_department.astype(str).str.strip().eq("").any():
            raise ValueError("Index departments must be supplied for summaries")
        flags = marker_flags(frame, rules)
        scores = rule_scores(frame, rules)
        baseline = [
            {"feature": name, **group_summary(frame, frame[name], name in CLINICAL_BINARY)}
            for name in CLINICAL_FEATURES
        ]
        for department in sorted(frame.index_department.unique()):
            baseline.append(
                {
                    "feature": "department: " + department,
                    **group_summary(frame, frame.index_department == department, True),
                }
            )
        ecg = [
            {"feature": name, **group_summary(frame, frame[name], name in ECG_BINARY)}
            for name in ECG_FEATURES
        ]
        for name in flags:
            ecg.append({"feature": name, **group_summary(frame, flags[name], True)})
        severity = []
        for name, cutoffs in (("news", [5, 7]), ("apache", [10, 15, 20])):
            severity.append({"feature": name, **group_summary(frame, scores[name])})
            for cutoff in cutoffs:
                binary = (scores[name] >= cutoff).astype(float).where(scores[name].notna())
                severity.append({"feature": f"{name} >= {cutoff}", **group_summary(frame, binary, True)})
        for suffix, rows in (("baseline", baseline), ("ecg", ecg), ("severity", severity)):
            pd.DataFrame(rows).to_csv(out / f"{cohort}_{suffix}.csv", index=False)
        case_times = interval_hours(frame[frame.label == 1])
        case_times.rename("hours").to_csv(out / f"{cohort}_ecg_to_arrest.csv")
        summary = {
            "cases": len(case_times),
            "mean": case_times.mean(),
            "sd": case_times.std(),
            "median": case_times.median(),
            "q25": case_times.quantile(0.25),
            "q75": case_times.quantile(0.75),
            "fraction_le_24h": (case_times <= 24).mean(),
            "fraction_le_48h": (case_times <= 48).mean(),
        }
        pd.DataFrame([summary]).to_csv(out / f"{cohort}_ecg_to_arrest_summary.csv", index=False)
        plots.intervals(case_times, out / f"{cohort}_ecg_to_arrest.png")


def subgroups(study, out):
    frame, pred = study.dm, study.oof
    cases = frame.loc[frame.label == 1].copy()
    masks = {
        "Male": frame.sex == 1,
        "Female": frame.sex == 0,
        "Age <=70": frame.age <= 70,
        "Age >70": frame.age > 70,
    }
    for field, levels in (
        ("rhythm", ("shockable", "nonshockable")),
        ("etiology", ("cardiovascular", "noncardiovascular")),
    ):
        unknown = set(cases[field]) - set(levels) - {""}
        if unknown:
            raise ValueError(f"Unknown {field} values: {unknown}")
        for level in levels:
            masks[level] = frame.pair_id.isin(cases.loc[cases[field] == level, "pair_id"])
    hours = interval_hours(cases)
    for name, select in (("ECG to arrest <=24 h", hours <= 24), ("ECG to arrest >24 h", hours > 24)):
        masks[name] = frame.pair_id.isin(cases.loc[select, "pair_id"])
    rows = [
        {"subgroup": name, **performance(pred.loc[mask], study.repeats, study.seed)}
        for name, mask in masks.items()
    ]
    table = pd.DataFrame(rows)
    table.to_csv(out / "phenotype_performance.csv", index=False)
    plots.subgroups(table, out / "phenotype_auroc.png")


def prevalence(study, out):
    rates = np.asarray(study.cfg["incidence_per_1000"], dtype=float)
    if rates.ndim != 1 or not len(rates) or not np.isfinite(rates).all():
        raise ValueError("Finite incidence scenarios are required")
    rows = []
    for cohort, frame in (("development", study.oof), ("external", study.external)):
        m = binary_metrics(frame.label, frame.score, predictions=frame.prediction)
        se, sp = m["sensitivity"], m["specificity"]
        ppv, npv = prevalence_predictive_values(se, sp, rates / 1000)
        for rate, pos, neg in zip(rates, ppv, npv):
            rows.append(
                {
                    "cohort": cohort,
                    "incidence_per_1000": rate,
                    "sensitivity": se,
                    "specificity": sp,
                    "ppv": pos,
                    "npv": neg,
                    "positive_per_1000": rate * se + (1000 - rate) * (1 - sp),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out / "incidence_scenarios.csv", index=False)
    plots.prevalence(table, out / "predictive_values.png")
