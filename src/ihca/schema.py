"""Ordered input definitions used by preparation, checkpoints and inference."""

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
ECG_FEATURES = (
    "ecg_heart_rate", "pr_interval_ms", "qrs_duration_ms", "qt_interval_ms",
    "qtc_bazett_ms", "qrs_t_angle_deg", "lvh_present", "tpeak_tend_ms",
    "ecg_atrial_fibrillation", "ventricular_arrhythmia", "st_elevation",
    "st_depression", "t_wave_inversion",
)
CLINICAL_FEATURES = (
    "age", "sex", "bmi", "hypertension", "diabetes", "coronary_artery_disease",
    "heart_failure", "chronic_kidney_disease", "atrial_fibrillation", "copd",
    "prior_myocardial_infarction", "prior_stroke", "family_history_scd",
    "antiarrhythmic_medication", "current_smoker", "systolic_bp", "diastolic_bp",
    "heart_rate", "respiratory_rate", "temperature_c", "spo2_percent",
)
ECG_BINARY = {"lvh_present", "ecg_atrial_fibrillation", "ventricular_arrhythmia",
              "st_elevation", "st_depression", "t_wave_inversion"}
CLINICAL_BINARY = set(CLINICAL_FEATURES[3:15]) | {"sex"}


def feature_schema():
    return {
        "version": 1, "leads": list(LEADS), "sampling_rate_hz": 500, "samples": 5000,
        "ecg_features": list(ECG_FEATURES), "clinical_features": list(CLINICAL_FEATURES),
        "ecg_binary": sorted(ECG_BINARY), "clinical_binary": sorted(CLINICAL_BINARY),
    }
