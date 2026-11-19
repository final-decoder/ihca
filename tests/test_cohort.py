import csv

import numpy as np
import pandas as pd
import pytest

from ihca.data import load_data, prepare_data
from ihca.experiments.cohort import assemble
from ihca.experiments.risksets import department_windows, select_candidates
from ihca.schema import CLINICAL_FEATURES, ECG_FEATURES


def table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def hospital(root):
    def time(year, hour):
        return f"{year}-03-04T{hour:02d}:00:00+00:00"

    stays, wards, ecgs = [], [], []
    features = {k: 0 for k in (*ECG_FEATURES, *CLINICAL_FEATURES[2:])}
    np.save(root / "wave.npy", np.tile(np.arange(5000, dtype="f"), (12, 1)))
    np.save(root / "flat.npy", np.zeros((12, 5000), dtype="f"))

    def stay(key, year, age, patient, sex=1, arrest=None, case=0, discharge=20):
        stays.append(dict(encounter_id=key, patient_id=patient, age=age, sex=sex,
                          admission_time=time(year, 1), discharge_time=time(year, discharge),
                          first_ihca_time=time(year, arrest) if arrest else "", case_eligible=case))

    def ward(key, year, name, start, end):
        wards.append(dict(encounter_id=key, department=name, start_time=time(year, start),
                          end_time=time(year, end)))

    def recording(key, year, name, hour, path="wave.npy", **kw):
        row = dict(encounter_id=key, record_id=name, waveform_path=path, sampling_rate_hz=500,
                   quality_score=91, ecg_time=time(year, hour), clinical_time=time(year, 1), **features)
        row.update(kw)
        ecgs.append(row)

    stay("C", 2022, 68, "case", arrest=10, case=1)
    ward("C", 2022, "Ward B", 1, 20)
    recording("C", 2022, "case-earlier", 2)
    recording("C", 2022, "case-latest", 9)
    recording("C", 2022, "case-at-index", 10)
    for key, age, pat, arrest, end in (("K1", 69, "p1", None, 20), ("K2", 67, "p2", None, 20),
                                       ("self", 68, "case", None, 20), ("ended", 68, "p3", None, 10),
                                       ("event", 68, "p4", 9, 20), ("gap", 68, "p5", None, 20)):
        stay(key, 2019, age, pat, arrest=arrest, discharge=end)
        ward(key, 2019, "Ward A", 1, 10 if key != "ended" else 5)
        if key != "ended":
            ward(key, 2019, "Ward B", 11 if key == "gap" else 10, end)
        recording(key, 2019, key + "-valid", 4)
    recording("K1", 2019, "bad-wave", 6, "flat.npy")
    recording("K1", 2019, "bad-feature", 7, bmi="")
    recording("K1", 2019, "bad-time", 8, clinical_time=time(2019, 9))
    recording("K1", 2019, "at-index", 10)
    return stays, wards, ecgs


def run(root, tables, output):
    return assemble(*(table(root / name, rows) for name, rows in
                      zip(("enc.csv", "wards.csv", "ecgs.csv"), tables)), output)


def test_boundaries_latest_complete_and_stable_ties(tmp_path):
    tables = hospital(tmp_path)
    output = tmp_path / "matched" / "primary.csv"
    result = run(tmp_path, tables, output)
    assert result == {"encounters": 7, "screened_out_ecgs": 4, "cases_without_ecg": 0, "pairs": 1}
    with output.open(newline="", encoding="utf-8") as stream:
        case, control = csv.DictReader(stream)
    assert (case["source_record_id"], control["source_record_id"]) == ("case-latest", "K1-valid")
    assert case["index_department"] == control["index_department"] == "Ward B"
    assert case["index_time"].startswith("2022") and control["index_time"].startswith("2019")
    assert (output.parent / control["waveform_path"]).resolve() == tmp_path / "wave.npy"
    dataset = tmp_path / "ready.npz"
    prepare_data(output, dataset, "development")
    assert list(load_data(dataset)["label"]) == [1, 0]
    again = output.with_name("reordered.csv")
    run(tmp_path, [rows[::-1] for rows in tables], again)
    assert again.read_bytes() == output.read_bytes()


@pytest.mark.parametrize("problem", ["overlap", "unknown", "case gap", "no control", "missing waveform"])
def test_inconsistent_exports_do_not_publish(tmp_path, problem):
    stays, wards, ecgs = hospital(tmp_path)
    if problem == "overlap":
        wards.append(wards[0].copy())
    elif problem == "unknown":
        ecgs[0]["encounter_id"] = "unknown"
    elif problem == "case gap":
        wards[0]["end_time"] = "2022-03-04T09:00:00+00:00"
    elif problem == "no control":
        for stay in stays[1:]:
            stay["sex"] = 0
    else:
        ecgs[0]["waveform_path"] = "absent.npy"
    destination = tmp_path / "out.csv"
    with pytest.raises((ValueError, FileNotFoundError)):
        run(tmp_path, (stays, wards, ecgs), destination)
    assert not destination.exists()


def test_extended_control_pool_uses_ward_at_assigned_index():
    admission = pd.Timestamp("2023-05-01T00:00Z")
    pool = pd.DataFrame({"patient_id": ["new", "left", "future"], "sex": 1, "age": [71, 70, 70],
                         "index_department": "cardiology", "admission": admission,
                         "ecg": admission + pd.Timedelta(hours=1),
                         "risk_end": admission + pd.Timedelta(days=2),
                         "department_start_time": ["2023-05-01T06:00Z", "2023-05-01T00:00Z",
                                                   "2023-05-01T08:00Z"],
                         "department_end_time": ["2023-05-02T00:00Z", "2023-05-01T06:00Z",
                                                 "2023-05-02T00:00Z"]}, index=["a", "b", "c"])
    case = pd.Series({"sex": 1, "age": 70, "index_department": "cardiology", "patient_id": "x"})
    ids = select_candidates(department_windows(pool), case, case, pd.Timedelta(hours=6),
                            set(), 1, np.random.default_rng(1))
    assert ids == ["a"]
