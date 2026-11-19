# Data specification

Prepare CSV manifests from adjudicated matched cohorts using the study eligibility criteria, incidence-density risk sets, nearest-neighbor age matching, and exact sex and index-department matching. Export the most recent eligible ECG preceding each assigned index time. Export clinical variables available at or before that ECG, together with the automated ECG measurements. The acquisition system supplies calibration and quality scores. Department is used for cohort matching and is excluded from the model's input features.

Each row describes one case or one matched control. Reused controls receive distinct record IDs and retain the same patient ID. Each matched pair has exactly one row labeled 1 and one row labeled 0. Patient IDs should use a consistent pseudonymous namespace across sites and visits; do not assign different IDs to the same patient across cohorts.

## Manifest metadata

| Column | Definition |
| --- | --- |
| `record_id` | Unique record identifier |
| `patient_id` | Stable participant identifier |
| `pair_id` | Matched pair identifier; required for labeled cohorts |
| `label` | Case 1, matched control 0; omitted for prediction |
| `waveform_path` | XML or NPY path, absolute or relative to the CSV |
| `sampling_rate_hz` | 500 |
| `quality_score` | Acquisition quality score on a 0–100 scale; at least 80 |
| `ecg_time` | ISO 8601 ECG acquisition time |
| `clinical_time` | Latest availability time among the exported clinical inputs |
| `index_time` | Case arrest time or assigned control index time; omitted for prediction |

Use a consistent time zone, with offsets either present on all timestamps or absent on all. Clinical availability must be at or before ECG acquisition; ECG acquisition must precede the matched index time. Historical diagnoses and medication variables must describe the status available at that time.

## Waveforms

NPY files contain a finite `float32` or `float64` array of shape `(12, 5000)` in millivolts, ordered as `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`. Each record covers ten seconds at 500 Hz. Missing samples, flat leads and incomplete channels are rejected. Preparation preserves the calibrated waveform. Loading applies a separate mean and population standard deviation to each lead within each record.

XML input uses a configurable mapping. The default document has an `ecg` root with a `sampling_rate_hz="500"` attribute, twelve `lead` elements, and a `measurements` element. Each lead has `name`, `gain` and `baseline` attributes and a `samples` child. Calibration is `(sample - baseline) * gain`, where gain converts to millivolts. `samples` accepts whitespace/comma-separated numeric values with `encoding="text"`, or little-endian signed 16-bit samples with `encoding="base64_int16_le"`. Lead names determine the ordering; XML element order does not.

For a different acquisition schema, supply `--xml-map path/to/mapping.json`. The mapping accepts these keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `sample_rate` | `@sampling_rate_hz` | Sampling rate selector relative to root |
| `leads` | `./lead` | XPath selecting lead elements |
| `lead_name` | `@name` | Lead identity selector within each lead |
| `samples` | `./samples` | Sample text selector within each lead |
| `gain` | `@gain` | Calibration gain selector within each lead |
| `baseline` | `@baseline` | Calibration baseline selector within each lead |
| `encoding` | Read from `samples/@encoding` | Optional fixed encoding |
| `lead_names` | Identity map | Map acquisition lead names to canonical names |
| `measurements` | `./measurements/<feature>` | Map ECG feature names to selectors relative to root |

Selectors use ElementTree XPath notation and support an `@attribute` suffix. Namespaced tags use `{namespace-uri}tag`. A nonempty CSV measurement takes precedence over the corresponding XML field. Missing measurements are rejected; the preparation pipeline does not derive automated interpretations from the waveform.

## Structured ECG fields

| Field | Unit or encoding |
| --- | --- |
| `ecg_heart_rate` | beats/min |
| `pr_interval_ms` | ms |
| `qrs_duration_ms` | ms |
| `qt_interval_ms` | ms |
| `qtc_bazett_ms` | ms, acquisition-system Bazett-corrected QT |
| `qrs_t_angle_deg` | Frontal QRS–T angle, 0–180 degrees |
| `lvh_present` | Automated LVH interpretation, absent 0 / present 1 |
| `tpeak_tend_ms` | ms |
| `ecg_atrial_fibrillation` | absent 0 / present 1 |
| `ventricular_arrhythmia` | absent 0 / present 1 |
| `st_elevation` | absent 0 / present 1 |
| `st_depression` | absent 0 / present 1 |
| `t_wave_inversion` | absent 0 / present 1 |

## Clinical fields

All 21 fields are required in this order:

| Field | Unit or encoding |
| --- | --- |
| `age` | Years, at least 18 |
| `sex` | Female 0 / male 1 |
| `bmi` | kg/m² |
| `hypertension` | absent 0 / present 1 |
| `diabetes` | absent 0 / present 1 |
| `coronary_artery_disease` | absent 0 / present 1 |
| `heart_failure` | absent 0 / present 1 |
| `chronic_kidney_disease` | absent 0 / present 1 |
| `atrial_fibrillation` | absent 0 / present 1 |
| `copd` | absent 0 / present 1 |
| `prior_myocardial_infarction` | absent 0 / present 1 |
| `prior_stroke` | absent 0 / present 1 |
| `family_history_scd` | absent 0 / present 1 |
| `antiarrhythmic_medication` | no 0 / yes 1 |
| `current_smoker` | no 0 / yes 1 |
| `systolic_bp` | mmHg |
| `diastolic_bp` | mmHg |
| `heart_rate` | beats/min |
| `respiratory_rate` | breaths/min |
| `temperature_c` | °C |
| `spo2_percent` | % |

Binary values retain their encoding. Continuous ECG and clinical fields use min-max parameters fitted on each fold's fitting partition. Constant training features use a denominator of one. Held-out values can lie outside `[0, 1]`; they are neither refitted nor clipped.

Prepared NPZ files contain numeric waveform, ECG, clinical and label arrays; string identifiers; connected pair/patient group IDs; the cohort designation; and the ordered input schema. They can be loaded without pickle support. Development, external and prediction datasets use the same input definitions.
