import csv
import json

import numpy as np
import pytest

from ihca.data import (FeatureScaler, connected_groups, load_data, normalize_waveforms,
                       numeric_features, prepare_data, read_xml_waveform, validate_waveform)
from ihca.schema import CLINICAL_FEATURES, ECG_BINARY, ECG_FEATURES, LEADS
from ihca.splits import inner_split, outer_folds, training_batches


def test_within_record_normalization_and_flat_rejection():
    x = np.random.default_rng(4).normal(size=(2, 12, 5000)).astype(np.float32)
    x[1] = 7 * x[0] + 31
    y = normalize_waveforms(x)
    np.testing.assert_allclose(y.mean(-1), 0, atol=1e-6)
    np.testing.assert_allclose(y.std(-1), 1, atol=1e-6)
    np.testing.assert_allclose(y[0], y[1], atol=3e-6)
    with pytest.raises(ValueError, match="flat"):
        validate_waveform(np.zeros((12, 5000)))


def test_waveform_completeness():
    with pytest.raises(ValueError, match="5000"):
        validate_waveform(np.ones((12, 4999)))
    x = np.ones((12, 5000)); x[1, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_waveform(x)


def test_xml_reorders_calibrates_and_extracts(tmp_path):
    text = '<ecg sampling_rate_hz="500">'
    for i, name in reversed(list(enumerate(LEADS))):
        text += f'<lead name="{name}" gain="0.01" baseline="10"><samples encoding="text">{10+i} {20+i}</samples></lead>'
    text += '<measurements><qt_interval_ms>410</qt_interval_ms></measurements></ecg>'
    path = tmp_path / "record.xml"; path.write_text(text)
    waveform, measures = read_xml_waveform(path)
    np.testing.assert_allclose(waveform[:, 0], np.arange(12) * 0.01)
    np.testing.assert_allclose(waveform[:, 1], (np.arange(12) + 10) * 0.01)
    assert measures["qt_interval_ms"] == "410"


def test_xml_rejects_incomplete_and_wrong_rate(tmp_path):
    path = tmp_path / "record.xml"
    path.write_text('<ecg sampling_rate_hz="250"/>')
    with pytest.raises(ValueError, match="500 Hz"):
        read_xml_waveform(path)
    path.write_text('<ecg sampling_rate_hz="500"/>')
    with pytest.raises(ValueError, match="twelve"):
        read_xml_waveform(path)


def test_required_measurements_are_not_imputed():
    with pytest.raises(ValueError, match="Missing"):
        numeric_features({}, ECG_FEATURES, ECG_BINARY)
    with pytest.raises(ValueError, match="Binary"):
        numeric_features({"st_elevation": 2}, ["st_elevation"], ECG_BINARY)


def test_scaling_uses_only_fitting_records_and_retains_binary():
    clinical = np.ones((3, 21), dtype=np.float32)
    clinical[:, 0] = [20, 40, 100]
    clinical[:, 1] = [0, 0, 1]
    data = {"clinical": clinical, "ecg": np.zeros((3, 13), dtype=np.float32)}
    scaler = FeatureScaler.fit(data, [0, 1])
    result = scaler.transform(data, "clinical", [2])
    assert result[0, 0] == 4
    assert result[0, 1] == 1


def test_pairs_and_reused_patients_stay_together():
    pairs = np.repeat([f"p{i}" for i in range(10)], 2)
    patients = np.asarray([f"s{i}" for i in range(20)])
    patients[3] = patients[1]
    groups = connected_groups(pairs, patients)
    assert len(set(groups[:4])) == 1
    folds = outer_folds(groups, 5, 42)
    assert sorted(np.concatenate(folds).tolist()) == list(range(20))
    for heldout in folds:
        available = np.setdiff1d(np.arange(20), heldout)
        fit, inner = inner_split(available, groups, 0.2, 42)
        for a, b in ((fit, inner), (fit, heldout), (inner, heldout)):
            assert not set(patients[a]) & set(patients[b])
            assert not set(pairs[a]) & set(pairs[b])


@pytest.mark.parametrize("n,batch", [(9, 4), (3, 2), (10, 64), (20, 4)])
def test_batchnorm_batches_preserve_all_records(n, batch):
    batches = training_batches(n, batch, np.random.default_rng(42))
    assert sorted(sum(batches, [])) == list(range(n))
    assert all(len(x) >= 2 for x in batches)


def manifest_rows(tmp_path):
    wave = np.random.default_rng(2).normal(size=(12, 5000)).astype(np.float32)
    np.save(tmp_path / "wave.npy", wave)
    row = dict(record_id="r0", patient_id="s0", pair_id="p0", label="0", waveform_path="wave.npy",
               sampling_rate_hz="500", quality_score="90", ecg_time="2026-01-01T09:00:00",
               clinical_time="2026-01-01T08:00:00", index_time="2026-01-01T10:00:00")
    row.update({name: "0" for name in CLINICAL_FEATURES + ECG_FEATURES})
    row["age"] = "65"
    other = dict(row, record_id="r1", patient_id="s1", label="1")
    return [row, other]


def save_manifest(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)


def test_prepare_load_and_missing_time_boundary(tmp_path):
    rows = manifest_rows(tmp_path)
    manifest, output = tmp_path / "records.csv", tmp_path / "records.npz"
    save_manifest(manifest, rows)
    assert prepare_data(manifest, output, "development")["pairs"] == 1
    data = load_data(output)
    assert data["waveform"].shape == (2, 12, 5000)
    rows[0]["clinical_time"] = "2026-01-01T11:00:00"
    save_manifest(manifest, rows)
    with pytest.raises(ValueError, match="Clinical inputs"):
        prepare_data(manifest, tmp_path / "future.npz", "development")
    assert not (tmp_path / "future.npz").exists()


def test_prediction_manifest_needs_no_outcomes(tmp_path):
    rows = manifest_rows(tmp_path)
    for row in rows:
        for key in ("label", "pair_id", "index_time"):
            row.pop(key)
    manifest = tmp_path / "unlabeled.csv"; save_manifest(manifest, rows)
    output = tmp_path / "unlabeled.npz"
    prepare_data(manifest, output, "prediction")
    assert np.isnan(load_data(output)["label"]).all()


def test_single_class_matched_pair_is_rejected(tmp_path):
    rows = manifest_rows(tmp_path); rows[1]["label"] = "0"
    manifest = tmp_path / "invalid.csv"; save_manifest(manifest, rows)
    with pytest.raises(ValueError, match="one case and one control"):
        prepare_data(manifest, tmp_path / "invalid.npz", "development")
