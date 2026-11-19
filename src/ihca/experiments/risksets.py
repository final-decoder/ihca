"""Frozen out-of-fold scoring of nested incidence-density risk sets."""

from collections import defaultdict

import numpy as np
import pandas as pd

from ihca.data import connected_groups, load_data
from ihca.engine import get_device, load_checkpoint, predict_model
from . import plots
from .statistics import performance
from .study import aligned_metadata


def timestamps(frame, key):
    if frame[key].astype(str).str.strip().eq("").any():
        raise ValueError(f"Incomplete {key}")
    return pd.to_datetime(frame[key], utc=True, format="ISO8601", errors="raise")


def department_windows(pool):
    """Restrict candidate snapshots to the known validity of their index ward."""
    fields = {"department_start_time", "department_end_time"}
    present = fields.intersection(pool.columns)
    if not present:
        return pool
    if present != fields:
        raise ValueError("Provide both department_start_time and department_end_time")
    pool = pool.copy()
    pool["ward_start"] = timestamps(pool, "department_start_time")
    pool["ward_end"] = timestamps(pool, "department_end_time")
    if (pool.ward_start < pool.admission).any() or (pool.ward_end <= pool.ward_start).any():
        raise ValueError("Invalid department interval in the candidate pool")
    return pool


def select_candidates(pool, case, primary, elapsed, forbidden, required, rng):
    assigned = pool.admission + elapsed
    allowed = (
        (pool.sex == case.sex)
        & (pool.index_department == case.index_department)
        & (pool.ecg >= pool.admission)
        & (pool.ecg < assigned)
        & (pool.risk_end > assigned)
        & ~pool.patient_id.isin(forbidden | {case.patient_id, primary.patient_id})
    )
    if "ward_start" in pool:
        allowed &= (assigned >= pool.ward_start) & (assigned < pool.ward_end)
    eligible = pool.loc[allowed, ["patient_id", "ecg", "age"]].copy()
    eligible["distance"] = (eligible.age - case.age).abs()
    eligible["tie"] = rng.random(len(eligible))
    eligible = eligible.sort_values(["ecg", "tie"], ascending=[False, True]).drop_duplicates(
        "patient_id"
    )
    eligible = eligible.sort_values(["distance", "tie"])
    if len(eligible) < required:
        raise ValueError(
            f"Case {case.name} needs {required} additional distinct controls; found {len(eligible)}"
        )
    return eligible.index[:required].tolist()


def ratios(study, out):
    ratios = sorted(set(study.cfg["controls_per_case"]))
    if not ratios or ratios[0] != 1 or any(type(k) is not int or k < 1 for k in ratios):
        raise ValueError("Control ratios must be positive integers including one")
    data = load_data(study.path("candidate_data"))
    if str(data["cohort"]) != "prediction" or np.isfinite(data["label"]).any():
        raise ValueError("Supply candidate ECG records through the unlabeled prediction interface")
    pool = aligned_metadata(data, study.path("candidate_metadata"))
    pool["admission"] = timestamps(pool, "admission_time")
    pool["risk_end"] = timestamps(pool, "at_risk_until")
    pool["ecg"] = timestamps(pool, "ecg_time")
    pool = department_windows(pool)
    if (pool.risk_end <= pool.admission).any() or pool.index_department.str.strip().eq("").any():
        raise ValueError("Invalid risk interval or missing index department")
    dev, oof = study.dm, study.oof
    if set(pool.index) & set(dev.index):
        raise ValueError("Candidate record IDs must be distinct from development record IDs")
    elapsed = timestamps(dev, "index_time") - timestamps(dev, "admission_time")
    if (elapsed <= pd.Timedelta(0)).any():
        raise ValueError("Index time must be later than admission")
    forbidden = {
        k: set(study.dev["patient_id"][np.r_[train, inner]]) for k, train, inner, _ in study.folds
    }
    rng = np.random.default_rng(study.seed)
    needs = defaultdict(set)
    rows = []
    for pair, block in dev.groupby("pair_id", sort=True):
        case = block.loc[block.label == 1].iloc[0]
        ctl = block.loc[block.label == 0].iloc[0]
        if (
            case.patient_id == ctl.patient_id
            or case.sex != ctl.sex
            or case.index_department != ctl.index_department
        ):
            raise ValueError(f"Invalid primary matched pair: {pair}")
        fold = int(oof.at[case.name, "fold"])
        for row, rank in ((case, 0), (ctl, 1)):
            rows.append(
                (
                    pair,
                    str(row.name),
                    row.patient_id,
                    int(row.label),
                    rank,
                    fold,
                    oof.at[row.name, "score"],
                    oof.at[row.name, "threshold"],
                )
            )
        chosen = select_candidates(
            pool, case, ctl, elapsed[case.name], forbidden[fold], ratios[-1] - 1, rng
        )
        for rank, rid in enumerate(chosen, 2):
            needs[fold].add(rid)
            rows.append(
                (
                    pair,
                    rid,
                    pool.at[rid, "patient_id"],
                    0,
                    rank,
                    fold,
                    np.nan,
                    oof.at[case.name, "threshold"],
                )
            )
    table = pd.DataFrame(
        rows,
        columns=["pair_id", "record_id", "patient_id", "label", "rank", "fold", "score", "threshold"],
    )
    device = get_device("cpu")
    for fold, ids in sorted(needs.items()):
        ordered = sorted(ids)
        model, scaler, state = load_checkpoint(study.model("full") / f"fold_{fold}.pt", device)
        if set(pool.loc[ordered, "patient_id"]) & set(state["fit_patients"]):
            raise ValueError("A candidate was used to fit its assigned fold model")
        score = predict_model(
            model, data, pool.index.get_indexer(ordered), scaler, study.cfg["batch_size"], device
        )
        mask = (table.fold == fold) & table.score.isna()
        table.loc[mask, "score"] = table.loc[mask, "record_id"].map(dict(zip(ordered, score)))
    if not np.isfinite(table.score).all():
        raise ValueError("Unscored risk-set member")
    table["prediction"] = table.score >= table.threshold
    table.to_csv(out / "nested_risk_sets.csv", index=False)
    summaries = []
    for k in ratios:
        selected = table.loc[table["rank"] <= k].copy()
        selected["group_id"] = connected_groups(selected.pair_id, selected.patient_id)
        summaries.append({"controls_per_case": k, **performance(selected, study.repeats, study.seed)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "sampling_ratios.csv", index=False)
    plots.ratios(summary, out / "sampling_ratios.png")
