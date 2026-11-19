"""Pre-index NEWS, APACHE II and threshold-based ECG markers in native units."""

from datetime import datetime

import numpy as np
import pandas as pd


def value(row, key, low=None, high=None, integer=False):
    raw = row.get(key, "")
    if pd.isna(raw) or str(raw).strip() == "":
        return np.nan
    number = float(raw)
    if (
        not np.isfinite(number)
        or (low is not None and number < low)
        or (high is not None and number > high)
        or (integer and not number.is_integer())
    ):
        raise ValueError(f"Invalid {key}: {raw}")
    return number


def points(number, cuts, weights, side="right"):
    if np.isnan(number):
        return np.nan
    return weights[np.searchsorted(cuts, number, side=side)]


def news(row):
    avpu = str(row.get("avpu", "")).strip().upper()
    if avpu and avpu not in ("A", "V", "P", "U"):
        raise ValueError("AVPU must be A, V, P or U")
    consciousness = 3 * int(avpu != "A") if avpu else np.nan

    def reading(name, precision=1, upper=None):
        number = value(row, name, 0, upper)
        return np.floor(number * precision + 0.5) / precision

    return sum(
        [
            points(reading("respiratory_rate"), [8, 11, 20, 24], [3, 1, 0, 2, 3], "left"),
            points(reading("spo2_percent", upper=100), [91, 93, 95], [3, 2, 1, 0], "left"),
            2 * value(row, "supplemental_oxygen", 0, 1, True),
            points(reading("temperature_c", 10), [35, 36, 38, 39], [3, 1, 0, 1, 2], "left"),
            points(reading("systolic_bp"), [90, 100, 110, 219], [3, 2, 1, 0, 3], "left"),
            points(reading("heart_rate"), [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3], "left"),
            consciousness,
        ]
    )


# Ascending lower bounds and points in the Knaus acute physiology scoring table.
APACHE_BANDS = {
    "apache_core_temp_c": ([30, 32, 34, 36, 38.5, 39, 41], [4, 3, 2, 1, 0, 1, 3, 4]),
    "apache_map_mmhg": ([50, 70, 110, 130, 160], [4, 2, 0, 2, 3, 4]),
    "apache_heart_rate": ([40, 55, 70, 110, 140, 180], [4, 3, 2, 0, 2, 3, 4]),
    "apache_respiratory_rate": ([6, 10, 12, 25, 35, 50], [4, 2, 1, 0, 1, 3, 4]),
    "sodium_mmol_l": ([111, 120, 130, 150, 155, 160, 180], [4, 3, 2, 0, 1, 2, 3, 4]),
    "potassium_mmol_l": ([2.5, 3, 3.5, 5.5, 6, 7], [4, 2, 1, 0, 1, 3, 4]),
    "hematocrit_percent": ([20, 30, 46, 50, 60], [4, 2, 0, 1, 2, 4]),
    "wbc_10e9_l": ([1, 3, 15, 20, 40], [4, 2, 0, 1, 2, 4]),
}


def apache_ii(row):
    total = sum(points(value(row, name, 0), *band) for name, band in APACHE_BANDS.items())
    fio2 = value(row, "fio2_fraction", 0.21, 1)
    if np.isnan(fio2):
        return np.nan
    if fio2 >= 0.5:
        oxygen = points(value(row, "aa_gradient_mmhg"), [200, 350, 500], [0, 2, 3, 4])
    else:
        oxygen = points(value(row, "pao2_mmhg", 0), [55, 61, np.nextafter(70.0, np.inf)], [4, 3, 1, 0])
    ph = value(row, "arterial_ph", 0, 14)
    if np.isfinite(ph):
        acid = points(ph, [7.15, 7.25, 7.33, 7.50, 7.60, 7.70], [4, 3, 2, 0, 1, 3, 4])
    else:
        acid = points(
            value(row, "bicarbonate_mmol_l", 0), [15, 18, 22, 32, 41, 52], [4, 3, 2, 0, 1, 3, 4]
        )
    creat = points(value(row, "creatinine_mg_dl", 0), [0.6, 1.5, 2, 3.5], [2, 0, 2, 3, 4])
    renal = value(row, "acute_renal_failure", 0, 1, True)
    age = points(value(row, "age", 18), [45, 55, 65, 75], [0, 2, 3, 5, 6])
    chronic = value(row, "severe_chronic_or_immunocompromised", 0, 1, True)
    if chronic == 1:
        category = str(row.get("admission_type", "")).strip().lower()
        if category not in ("elective", "emergency", "medical", ""):
            raise ValueError("Unknown APACHE admission type")
        chronic = {"elective": 2, "emergency": 5, "medical": 5, "": np.nan}[category]
    return (
        total
        + oxygen
        + acid
        + creat * (1 + renal)
        + (15 - value(row, "gcs", 3, 15, True))
        + age
        + chronic
    )


def marker_flags(frame, rules):
    for key in ("qtc_male_ms", "qtc_female_ms", "tpeak_tend_ms"):
        if not np.isfinite(rules[key]) or rules[key] <= 0:
            raise ValueError(f"Positive cutoff required for {key}")
    qtc_cut = np.where(frame.sex == 1, rules["qtc_male_ms"], rules["qtc_female_ms"])
    return pd.DataFrame(
        {
            "qrs_t_above_90": frame.qrs_t_angle_deg > 90,
            "prolonged_qtc": frame.qtc_bazett_ms > qtc_cut,
            "prolonged_tpeak_tend": frame.tpeak_tend_ms > rules["tpeak_tend_ms"],
        },
        index=frame.index,
    ).astype(int)


def rule_scores(frame, rules):
    scores = []
    for record, row in frame.iterrows():
        try:
            cutoff = datetime.fromisoformat(row["ecg_time"])
            latest = str(row.get("score_time", "")).strip()
            if latest and datetime.fromisoformat(latest) > cutoff:
                raise ValueError("Score input is dated after ECG acquisition")
            scores.append((news(row), apache_ii(row)) if latest else (np.nan, np.nan))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid scoring data for {record}: {exc}") from exc
    result = pd.DataFrame(scores, columns=["news", "apache"], index=frame.index)
    flags = marker_flags(frame, rules)
    weights = pd.Series(rules["ecg_weights"], dtype=float)
    if (
        set(weights.index) != set(flags.columns)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError("Provide finite nonnegative weights for all three conventional ECG markers")
    result["ecg_rule"] = flags.dot(weights)
    return result
