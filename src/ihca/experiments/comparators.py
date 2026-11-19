"""Rule and fitted comparators using the full model's grouped development splits."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from ihca.engine import write_json
from ihca.metrics import youden_threshold
from ihca.schema import CLINICAL_FEATURES
from .scores import rule_scores
from .statistics import performance


CUTOFF_KEYS = {"news": "news_positive", "apache": "apache_positive", "ecg_rule": "ecg_positive"}


def prediction_table(data, score, threshold, scale=1, folds=None):
    raw = np.asarray(score, dtype=float)
    threshold = np.broadcast_to(threshold, raw.shape)
    frame = pd.DataFrame(
        {k: data[k] for k in ("record_id", "patient_id", "pair_id", "group_id", "label")}
    )
    frame["native_score"] = raw
    frame["native_threshold"] = threshold
    frame["score"] = raw / scale
    frame["threshold"] = threshold / scale
    frame["prediction"] = np.where(np.isfinite(raw), raw >= threshold, np.nan)
    if folds is not None:
        frame["fold"] = folds
    return frame.set_index("record_id")


def rule_predictions(data, metadata, rules, folds=None):
    values = rule_scores(metadata, rules)
    tables = {}
    scales = {"news": 20, "apache": 71, "ecg_rule": sum(map(float, rules["ecg_weights"].values()))}
    for model, scale in scales.items():
        cutoff = float(rules[CUTOFF_KEYS[model]])
        if not 0 <= cutoff <= scale:
            raise ValueError(f"Invalid cutoff for {model}")
        tables[model] = prediction_table(data, values[model], cutoff, scale, folds)
    return tables


def fit_comparators(study, output):
    output.mkdir()
    data = study.dev
    settings = study.cfg
    manifest = {
        "features": list(CLINICAL_FEATURES),
        "rules": settings["clinical_rules"],
        "development_patients": sorted(set(data["patient_id"])),
        "folds": [],
    }
    fold_ids = study.oof.fold.to_numpy()
    tables = rule_predictions(data, study.dm, manifest["rules"], fold_ids)
    fitted_names = ("random_forest", "xgboost")
    predictions = {name: np.full(len(fold_ids), np.nan) for name in fitted_names}
    thresholds = {name: np.full(len(fold_ids), np.nan) for name in fitted_names}
    for fold, fit, inner, held in study.folds:
        common = {"random_state": study.seed + fold, "n_jobs": settings["threads"]}
        models = {
            "random_forest": RandomForestClassifier(**settings["random_forest"], **common),
            "xgboost": XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", **settings["xgboost"], **common
            ),
        }
        entries = {}
        for name, model in models.items():
            model.fit(data["clinical"][fit], data["label"][fit].astype(int))
            inner_score = model.predict_proba(data["clinical"][inner])[:, 1]
            threshold = youden_threshold(data["label"][inner], inner_score)
            predictions[name][held] = model.predict_proba(data["clinical"][held])[:, 1]
            thresholds[name][held] = threshold
            if name == "xgboost":
                file = f"{name}_{fold}.json"
                model.save_model(output / file)
            else:
                file = f"{name}_{fold}.joblib"
                joblib.dump(model, output / file)
            entries[name] = {"file": file, "threshold": threshold}
        manifest["folds"].append({"fold": fold, "models": entries})
    for name in fitted_names:
        if not np.isfinite(predictions[name]).all() or not np.isfinite(thresholds[name]).all():
            raise ValueError("Incomplete comparator fold coverage")
        tables[name] = prediction_table(data, predictions[name], thresholds[name], folds=fold_ids)
    write_json(output / "models.json", manifest)
    return tables


def frozen_comparators(directory, data, meta, threads=2):
    directory = Path(directory).resolve()
    info = json.loads((directory / "models.json").read_text(encoding="utf-8"))
    if list(CLINICAL_FEATURES) != info["features"] or set(info["development_patients"]) & set(
        data["patient_id"]
    ):
        raise ValueError("External patients must be distinct and predictor schema must match")
    tables = rule_predictions(data, meta, info["rules"])
    for name in ("random_forest", "xgboost"):
        scores, thresholds = [], []
        for fold in info["folds"]:
            entry = fold["models"][name]
            file = (directory / entry["file"]).resolve()
            if file.parent != directory:
                raise ValueError("Invalid model file location")
            if name == "xgboost":
                model = XGBClassifier(n_jobs=threads)
                model.load_model(file)
            else:
                model = joblib.load(file)
                model.n_jobs = threads
            scores.append(model.predict_proba(data["clinical"])[:, 1])
            thresholds.append(entry["threshold"])
        tables[name] = prediction_table(data, np.mean(scores, axis=0), np.median(thresholds))
    return tables


def compare(study, out):
    from .plots import curves

    internal = fit_comparators(study, out / "models")
    external = frozen_comparators(out / "models", study.ext, study.em, study.cfg["threads"])
    internal["full"] = study.oof.assign(
        native_score=study.oof.score, native_threshold=study.oof.threshold
    )
    external["full"] = study.external.assign(
        native_score=study.external.score, native_threshold=study.external.threshold
    )
    for cohort, tables in (("development", internal), ("external", external)):
        records, summaries = [], []
        for name, table in tables.items():
            records.append(table.assign(model=name))
            cutoffs = table.native_threshold.unique()
            source = (
                "prespecified"
                if name in CUTOFF_KEYS
                else (
                    "inner_validation_by_fold" if cohort == "development" else "median_inner_threshold"
                )
            )
            summaries.append(
                {
                    "model": name,
                    "operating_threshold_native": float(cutoffs[0]) if len(cutoffs) == 1 else np.nan,
                    "threshold_source": source,
                    **performance(table, study.repeats, study.seed),
                }
            )
        pd.concat(records).to_csv(out / f"{cohort}_predictions.csv")
        pd.DataFrame(summaries).to_csv(out / f"{cohort}_performance.csv", index=False)
        curves(tables, out / f"{cohort}_roc_pr.png")
