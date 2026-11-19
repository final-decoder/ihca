# Study experiments

Run commands from the project root with the `experiments` extra installed. Each run publishes to a new directory only after all selected analyses finish. Input NPZs are made by `ihca prepare`. The development and external metadata CSVs can be the same manifests supplied to preparation. Their unique record IDs and patient IDs must match the corresponding NPZ. A repeated patient keeps the same ID. Cohort metadata, phenotype labels and comparator-only variables are kept separate from the network predictors.

## Inputs and primary matching

For separate encounter, transfer and recording tables, use [transfer-aware cohort assembly](cohort-assembly.md). The following entry point accepts exports in which the eligible department at the assigned time is already resolved.

The primary analysis uses one adjudicated IHCA case and one at-risk matched control per pair. Follow [the acquisition and feature contract](data.md). To construct the primary pairs from eligible hospital exports, the following command selects the last eligible case ECG, matches sex and index department exactly, then minimizes absolute age difference. Reuse of controls across risk sets is allowed. The output records include the original source record IDs.

```console
ihca-experiments match --cases data/eligible_cases.csv --candidates data/eligible_controls.csv --output data/development.csv
```

Use a separate site-specific candidate pool to build the external cohort. Case exports include `case_event_id`, adjudicated `index_time`, `admission_time` and the model manifest fields. Each candidate row identifies an eligible ECG, the index department, `admission_time` and `at_risk_until` (the earliest of first arrest, hospital exit or end of observation). For a case arrested at time T after its admission, the candidate must still be at risk T after its own admission. The last eligible ECG before that assigned time is used. Provide adjudicated adult admissions and complete, acquisition-checked ECGs with a common time convention. For transfers use the interval columns below.

## Cohort metadata

For transfers in a candidate export, include `department_start_time` and `department_end_time` on every row. The index must fall within that half-open interval as well as the hospitalization risk period. A valid ECG obtained before the transfer can be used. Omit both columns only when the supplied index department applies throughout the risk period. This applies to `match` and the extended ratio analysis.

For multiple arrests during an admission, the matcher keeps the first adjudicated event and its most recent eligible ECG. Each eligible admission and patient needs a consistent identifier or admission timestamp in the source export.

| Column | Contents |
|---|---|
| `record_id`, `patient_id`, `pair_id` | Record, participant and primary matched pair identifiers |
| `index_department`, `admission_time` | Department at the index time and hospital admission timestamp |
| `ecg_time`, `index_time` | ECG acquisition and arrest or assigned control index time |
| `rhythm` | On case rows, `shockable` or `nonshockable` |
| `etiology` | On case rows, `cardiovascular` or `noncardiovascular` |
| `score_time` | Latest availability time for NEWS and APACHE II component values, at or before the ECG |
| `supplemental_oxygen`, `avpu` | Oxygen administered (0/1) and A/V/P/U consciousness level for the original NEWS |
| `apache_core_temp_c`, `apache_map_mmhg`, `apache_heart_rate`, `apache_respiratory_rate` | Core temperature, mean arterial pressure, pulse per minute and breaths per minute for APACHE II |
| `fio2_fraction`, `pao2_mmhg`, `aa_gradient_mmhg` | Inspired oxygen fraction (0.21–1), arterial oxygen pressure for FiO₂ below 0.5, alveolar–arterial gradient at or above 0.5 |
| `arterial_ph`, `bicarbonate_mmol_l` | Arterial pH or, when arterial pH is unavailable, serum bicarbonate |
| `sodium_mmol_l`, `potassium_mmol_l`, `creatinine_mg_dl` | Serum sodium, potassium and creatinine |
| `acute_renal_failure`, `hematocrit_percent`, `wbc_10e9_l`, `gcs` | Acute renal failure (0/1), hematocrit in percent, white cells in 10⁹/L and numerical Glasgow Coma Scale (3–15) |
| `severe_chronic_or_immunocompromised`, `admission_type` | Pre-admission chronic severe organ insufficiency or immunocompromise (0/1). With a positive flag use `medical`, `emergency` surgery or `elective` surgery |

Export the APACHE II component values with the greatest scoring abnormality in the prespecified pre-ECG observation window, using the correct oxygenation measurement and paired inspired oxygen fraction. Use presedation GCS when appropriate. NEWS uses contemporaneous clinical vitals and original AVPU scoring. Provide NEWS counts, saturation and blood pressure at chart resolution (whole units) and temperature to one decimal. Fields required by only one comparator may be blank. That score is unavailable for the record, the other methods retain it, and each performance report gives its scored denominator. An absent `score_time` makes both clinical scores unavailable. Binary variables use 0/1. Case-only phenotype metadata can be blank for unclassified cases and should be blank on controls.

## Analyses and outputs

The full study configuration contains input paths, four model directories and statistical settings. Supply only the resources required by selected analyses. Tables use full-precision numerical values. All model operating thresholds come from development procedures or a prespecified fixed rule.

| Analysis | Computation | Files inside the analysis directory |
|---|---|---|
| `compare` | NEWS, APACHE II, rule-based ECG marker count, random forest, XGBoost and the full model. Trees fit on the same fitting records as each full-model fold, using the 21 clinical variables. An inner validation set selects the tree threshold. External tree probabilities are averaged and fold thresholds use the median. | Development/external performance, record-level predictions, ROC and precision–recall plots, serialized fold models |
| `ablation` | All four saved modality configurations, identified by record and checked for identical splits. Alternative minus full AUROC uses the same matched-group draws for both scores, percentile 95% intervals, and a two-sided sign-tail P value with a finite-draw correction. Default 1,000 draws. | `modality_comparison.csv`, `fold_performance.csv`, `fold_summary.csv` |
| `shap` | Held-out full-model score, fixed waveform representation, 13 ECG and 21 clinical features. Forward and reverse permutation paths estimate interventional Shapley values against reference rows drawn exclusively from that fold's fitting records. | Individual signed contributions, mean absolute rankings, ranking figure |
| `subgroups` | Sex and age strata use individual records. Rhythm, etiology and ECG-to-arrest strata retain the case and its matched control. Age partition at 70 years, time partition at 24 hours. | Subgroup denominators, measures, matched-group intervals and AUROC plot |
| `ratios` | Retain the original case and control, then add nearest-age eligible controls from distinct patients in each risk set. The same ordered controls are reused at increasing ratios. Score each extra control with the held-out case's frozen fold, excluding that fold's fitting and inner-validation patients. | Nested risk-set membership with scores, thresholds and ranks, scenario metrics and AUROC/AUPRC plot |
| `prevalence` | Convert development and external sensitivity and specificity to PPV, NPV and expected positive classifications per 1,000 at each configured incidence. Development decisions use each record's own threshold. | Incidence-scenario table and predictive-value plot |
| `describe` | Development and external clinical, department, ECG and severity-score distributions, absolute standardized mean differences and ECG-to-arrest intervals. | Cohort summaries, individual case intervals, distribution statistics, histograms and cumulative fractions |

For raw score distributions use `native_score`. Each performance table includes the source of the operating threshold and its value in native units when a single cutoff applies. For fold-specific decisions, each record carries its own threshold. The unified prediction tables divide NEWS by 20, APACHE II by 71 and the ECG score by the sum of its prespecified marker weights (3 by default). These monotone rescalings preserve ranking and the configured raw operating points. By default, the ECG comparator adds one point per flag for frontal QRS–T angle above 90°, QTc above the configured sex-specific cutoff and Tpeak–Tend above the configured cutoff. Prespecify the rule thresholds, marker weights and fixed positive cutoffs before evaluation. Continuous SMD uses the unweighted pooled within-group variance. Binary SMD uses the group proportions. An undefined SMD or predictive value is left blank. The point estimates are record-weighted. Confidence intervals resample the connected groups of matching and repeated patients. A group containing both outcomes contributes a point estimate even when it is the only independent group available.

For the extended matching analysis, prepare the complete candidate ECG pool as an unlabeled prediction NPZ and supply the associated candidate metadata in the study configuration. Candidate record IDs must be distinct from primary development record IDs, and its patient identifiers must use the same namespace. The eligible risk end and department must apply to each candidate snapshot. Insufficient eligible unique controls for a requested ratio produce a diagnostic identifying the case. No case is silently dropped to achieve a ratio.

To score an additional independent labeled cohort with saved comparators, load only your study's own model files:

```console
ihca-experiments score-comparators --models outputs/experiments/compare/models --data data/external.npz --metadata data/external.csv --output outputs/external_comparators
```

The local web interface reads the full model checkpoints and their development fitting references. Upload a 500 Hz calibrated XML recording, enter the required clinical measurements and availability times, and review the mean fold score, median threshold and structured contributions. Structured ECG values are read from XML or supplied explicitly. The report can be exported as a self-contained HTML file or printed. Model scores express discrimination within the cohort design.
