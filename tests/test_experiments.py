import numpy as np
import pytest

# The core distribution can be installed without the experiment extra.
pytest.importorskip("xgboost")
pd = pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("matplotlib")
from ihca.experiments.attribution import permutation_shap
from ihca.experiments.matching import match
from ihca.experiments.risksets import select_candidates
from ihca.experiments.scores import apache_ii, news, rule_scores
from ihca.experiments.statistics import paired_differences, performance
from ihca.experiments.study import new_result_directory


def test_news_boundaries_oxygen_and_avpu():
    row = dict(
        respiratory_rate=12,
        spo2_percent=96,
        supplemental_oxygen=0,
        temperature_c=36.1,
        systolic_bp=111,
        heart_rate=51,
        avpu="A",
    )
    assert news(row) == 0
    row.update(
        respiratory_rate=8,
        spo2_percent=91,
        supplemental_oxygen=1,
        temperature_c=35,
        systolic_bp=90,
        heart_rate=40,
        avpu="V",
    )
    assert news(row) == 20
    row["avpu"] = ""
    assert np.isnan(news(row))


def test_apache_oxygenation_renal_and_chronic_points():
    row = dict(
        apache_core_temp_c=37,
        apache_map_mmhg=85,
        apache_heart_rate=85,
        apache_respiratory_rate=19,
        sodium_mmol_l=140,
        potassium_mmol_l=4.1,
        hematocrit_percent=39,
        wbc_10e9_l=8,
        fio2_fraction=0.21,
        pao2_mmhg=80,
        arterial_ph=7.4,
        creatinine_mg_dl=1,
        acute_renal_failure=0,
        age=40,
        gcs=15,
        severe_chronic_or_immunocompromised=0,
    )
    assert apache_ii(row) == 0
    row.update(
        fio2_fraction=0.5,
        aa_gradient_mmhg=350,
        creatinine_mg_dl=2,
        acute_renal_failure=1,
        gcs=10,
        age=70,
        severe_chronic_or_immunocompromised=1,
        admission_type="elective",
    )
    assert apache_ii(row) == 3 + 6 + 5 + 5 + 2
    row.update(arterial_ph="", bicarbonate_mmol_l=17)
    assert apache_ii(row) == 24
    row["aa_gradient_mmhg"] = ""
    assert np.isnan(apache_ii(row))


def test_permutation_shap_exact_linear_model():
    coeff = np.array([0.2, -0.3, 0.1])
    x = np.array([0.9, 0.1, 0.6])
    baseline = np.array([[0.3, 0.2, 0.4]])
    base, phi, score = permutation_shap(
        x, baseline, lambda v: 0.4 + v @ coeff, 4, np.random.default_rng(7)
    )
    np.testing.assert_allclose(phi, (x - baseline[0]) * coeff, atol=1e-10)
    assert abs(base + sum(phi) - score) < 1e-10


def test_ecg_weights_and_pre_index_availability():
    rules = dict(
        qtc_male_ms=450,
        qtc_female_ms=460,
        tpeak_tend_ms=100,
        ecg_weights={"qrs_t_above_90": 2, "prolonged_qtc": 3, "prolonged_tpeak_tend": 5},
    )
    rows = pd.DataFrame(
        dict(
            sex=[1, 0],
            qrs_t_angle_deg=[90, 91],
            qtc_bazett_ms=[455, 455],
            tpeak_tend_ms=[101, 100],
            ecg_time=["2025-06-01T10:00:00"] * 2,
            score_time=["", ""],
        )
    )
    scores = rule_scores(rows, rules)
    np.testing.assert_array_equal(scores.ecg_rule, [8, 2])
    assert scores[["news", "apache"]].isna().all().all()
    rows.loc[0, "score_time"] = "2025-06-01T10:01:00"
    with pytest.raises(ValueError, match="after ECG"):
        rule_scores(rows, rules)


def test_uncertainty_requires_independent_groups():
    frame = pd.DataFrame(
        dict(label=[0, 1], score=[0.1, 0.9], prediction=[False, True], group_id=["g", "g"])
    )
    result = performance(frame, 40, 8)
    assert result["auroc"] == 1 and result["valid_resamples"] == 0 and np.isnan(result["auroc_low"])
    with pytest.raises(ValueError, match="independent groups"):
        paired_differences(frame, {"same": frame.score.to_numpy()}, 40, 8)


def test_paired_comparison_shares_resampled_members():
    labels = np.tile([0, 1], 5)
    full = np.where(labels, 0.9, 0.1)
    frame = pd.DataFrame(dict(label=labels, score=full, group_id=np.repeat(np.arange(5), 2)))
    rows = paired_differences(frame, {"identical": full, "opposite": 1 - full}, 99, 10).set_index(
        "model"
    )
    assert rows.at["identical", "delta_auroc"] == 0 and rows.at["identical", "p_two_sided"] == 1
    assert rows.at["opposite", "delta_low"] == rows.at["opposite", "delta_high"] == -1


def test_riskset_uses_elapsed_time_latest_ecg_and_fitting_exclusions():
    utc = lambda s: pd.Timestamp(s, tz="UTC")
    pool = pd.DataFrame(
        dict(
            patient_id=["a", "a", "future", "ended", "fit"],
            age=[50] * 5,
            sex=[1] * 5,
            index_department=["A"] * 5,
            admission=[utc("2020-04-01")] * 5,
            ecg=[
                utc("2020-04-02"),
                utc("2020-04-03"),
                utc("2020-04-06"),
                utc("2020-04-02"),
                utc("2020-04-02"),
            ],
            risk_end=[utc("2020-04-10")] * 3 + [utc("2020-04-04"), utc("2020-04-10")],
        ),
        index=["old", "latest", "later", "closed", "trained"],
    )
    case = pd.Series(dict(patient_id="case", age=50, sex=1, index_department="A"))
    ctl = pd.Series(dict(patient_id="primary"))
    args = (pool, case, ctl, pd.Timedelta(days=4), {"fit"})
    assert select_candidates(*args, 1, np.random.default_rng(1)) == ["latest"]
    with pytest.raises(ValueError, match="distinct controls"):
        select_candidates(*args, 2, np.random.default_rng(1))


def test_failed_report_does_not_publish_partial_files(tmp_path):
    out = tmp_path / "report"
    with pytest.raises(ValueError):
        with new_result_directory(out) as stage:
            (stage / "partial.csv").touch()
            raise ValueError("failed")
    assert not out.exists()


def test_match_uses_first_arrest_and_last_eligible_ecg(tmp_path):
    common = dict(
        age=62,
        sex=1,
        index_department="ward",
        quality_score=95,
        sampling_rate_hz=500,
        admission_time="2025-01-01T00:00:00",
        clinical_time="2025-01-01T01:00:00",
        waveform_path="acquisition.npy",
    )
    rows = []
    for record, event, ecg, arrest in [
        ("earlier", "initial", 2, 5),
        ("last-before", "initial", 4, 5),
        ("recurrence", "later", 6, 8),
    ]:
        rows.append(
            dict(
                common,
                record_id=record,
                patient_id="p1",
                case_event_id=event,
                ecg_time=f"2025-01-01T{ecg:02d}:00:00",
                index_time=f"2025-01-01T{arrest:02d}:00:00",
            )
        )
    case_path, pool_path, result = (tmp_path / n for n in ["cases.csv", "pool.csv", "matched.csv"])
    pd.DataFrame(rows).to_csv(case_path, index=False)
    pd.DataFrame(
        [
            dict(
                common,
                record_id="control",
                patient_id="p2",
                ecg_time="2025-01-01T03:00:00",
                at_risk_until="2025-01-02T00:00:00",
            )
        ]
    ).to_csv(pool_path, index=False)
    assert match(case_path, pool_path, result) == 2
    selected = pd.read_csv(result)
    assert list(selected.source_record_id) == ["last-before", "control"]
    assert set(selected.pair_id) == {"initial"}
