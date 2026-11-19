"""Construct primary incidence-density matches from eligible case and control ECG exports."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .risksets import department_windows, select_candidates, timestamps


def read_manifest(path):
    rows = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("record_id")
    if not rows.index.is_unique or "" in rows.index or rows.patient_id.str.strip().eq("").any():
        raise ValueError("Unique record IDs and nonempty patient IDs are required")
    for field in ("age", "sex", "quality_score", "sampling_rate_hz"):
        rows[field] = pd.to_numeric(rows[field])
    if not rows.sex.isin([0, 1]).all() or rows.index_department.str.strip().eq("").any():
        raise ValueError("Each record requires a binary sex and index department")
    rows["waveform_path"] = [
        str((Path(path).resolve().parent / p).resolve()) for p in rows.waveform_path
    ]
    rows["ecg"] = timestamps(rows, "ecg_time")
    rows["admission"] = timestamps(rows, "admission_time")
    valid = (rows.age >= 18) & (rows.sampling_rate_hz == 500) & rows.quality_score.between(80, 100)
    valid &= (timestamps(rows, "clinical_time") <= rows.ecg) & (rows.ecg >= rows.admission)
    return rows.loc[valid].copy()


def match(cases_path, pool_path, destination, seed=42):
    destination = Path(destination).resolve()
    cases, pool = read_manifest(cases_path), read_manifest(pool_path)
    if "case_event_id" not in cases or cases.case_event_id.str.strip().eq("").any():
        raise ValueError("Supply an adjudicated case_event_id for each case ECG")
    pool["risk_end"] = timestamps(pool, "at_risk_until")
    pool = department_windows(pool)
    cases["event"] = timestamps(cases, "index_time")
    first_event = cases.groupby(["patient_id", "admission"])["event"].transform("min")
    cases = cases.loc[cases.event == first_event]
    if (cases.groupby(["patient_id", "admission"]).case_event_id.nunique() > 1).any():
        raise ValueError("Multiple event IDs identify the first arrest in one admission")
    cases = cases.loc[cases.ecg < cases.event].sort_values(["ecg"], kind="stable")
    if not len(cases):
        raise ValueError("No eligible index case recordings")
    rng = np.random.default_rng(seed)
    output = []
    for event_id, records in cases.groupby("case_event_id", sort=True):
        if (
            records.patient_id.nunique() != 1
            or records.event.nunique() != 1
            or records.admission.nunique() != 1
        ):
            raise ValueError(f"Inconsistent case event: {event_id}")
        case = records.iloc[-1]
        elapsed = case.event - case.admission
        chosen = select_candidates(pool, case, case, elapsed, set(), 1, rng)[0]
        control = pool.loc[chosen]
        for label, row in ((1, case), (0, control)):
            data = row.drop(labels=["ecg", "admission", "event", "risk_end", "ward_start", "ward_end"],
                            errors="ignore").to_dict()
            for key in ("ecg_time", "clinical_time", "score_time", "admission_time", "at_risk_until"):
                if data.get(key):
                    data[key] = pd.to_datetime(data[key], utc=True).isoformat()
            try:
                wave_path = os.path.relpath(row.waveform_path, destination.parent)
            except ValueError:
                wave_path = str(Path(row.waveform_path).resolve())
            data.update(
                record_id=f"{event_id}_{label}",
                source_record_id=row.name,
                pair_id=event_id,
                label=label,
                index_time=(row.admission + elapsed).isoformat(),
                waveform_path=wave_path,
            )
            output.append(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(destination, mode="x", index=False)
    return len(output)
