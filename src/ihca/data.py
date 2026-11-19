"""ECG ingestion, complete-case validation and training-partition scaling."""

import base64
import csv
from datetime import datetime
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import (CLINICAL_BINARY, CLINICAL_FEATURES, ECG_BINARY, ECG_FEATURES,
                     LEADS, feature_schema)


def xml_value(element, selector):
    if selector.startswith("@"):
        return element.get(selector[1:])
    if "/@" in selector:
        path, attribute = selector.rsplit("/@", 1)
        target = element.find(path)
        return target.get(attribute) if target is not None else None
    target = element.find(selector)
    return target.text if target is not None else None


def read_xml_waveform(path, mapping=None):
    mapping = mapping or {}
    payload = Path(path).read_bytes()
    if len(payload) > 32 * 1024 * 1024 or b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise ValueError("XML must be at most 32 MiB and cannot contain a DTD or entity declarations")
    root = ET.fromstring(payload)
    rate = float(xml_value(root, mapping.get("sample_rate", "@sampling_rate_hz")) or "nan")
    if rate != 500:
        raise ValueError("ECG sampling rate must be 500 Hz; resampling is not performed")
    channels = {}
    name_map = mapping.get("lead_names", {})
    for lead in root.findall(mapping.get("leads", "./lead")):
        name = xml_value(lead, mapping.get("lead_name", "@name"))
        name = name_map.get(name, name)
        if name not in LEADS or name in channels:
            raise ValueError(f"Unknown or duplicate lead: {name}")
        text = xml_value(lead, mapping.get("samples", "./samples"))
        if not text:
            raise ValueError(f"Empty samples for lead {name}")
        encoding = mapping.get("encoding") or xml_value(lead, "./samples/@encoding") or "text"
        if encoding == "text":
            values = np.asarray(text.replace(",", " ").split(), dtype=np.float64)
        elif encoding == "base64_int16_le":
            values = np.frombuffer(base64.b64decode(text, validate=True), dtype="<i2").astype(float)
        else:
            raise ValueError(f"Unsupported waveform encoding: {encoding}")
        gain = float(xml_value(lead, mapping.get("gain", "@gain")) or "nan")
        baseline = float(xml_value(lead, mapping.get("baseline", "@baseline")) or "nan")
        if not np.isfinite([gain, baseline]).all() or gain <= 0:
            raise ValueError("Each XML lead requires a finite positive mV gain and finite baseline")
        channels[name] = (values - baseline) * gain
    if set(channels) != set(LEADS):
        raise ValueError("All twelve ECG leads are required; leads are not reconstructed")
    waveform = np.stack([channels[name] for name in LEADS])
    measurements = {}
    for name in ECG_FEATURES:
        selector = mapping.get("measurements", {}).get(name, f"./measurements/{name}")
        value = xml_value(root, selector)
        if value is not None and str(value).strip():
            measurements[name] = value
    return waveform, measurements


def validate_waveform(waveform):
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.shape != (12, 5000):
        raise ValueError(f"Expected [12, 5000] waveform, received {waveform.shape}")
    if not np.isfinite(waveform).all():
        raise ValueError("ECG contains non-finite values; waveform imputation is not performed")
    if np.any(waveform.std(axis=-1, dtype=np.float64) < 1e-8):
        raise ValueError("ECG contains a flat lead")
    return waveform


def normalize_waveforms(waveform):
    mean = waveform.mean(axis=-1, keepdims=True, dtype=np.float64)
    std = waveform.std(axis=-1, keepdims=True, dtype=np.float64)
    if np.any(std < 1e-8) or not np.isfinite(waveform).all():
        raise ValueError("Waveforms must be finite and nonconstant in each lead")
    return ((waveform - mean) / std).astype(np.float32)


def numeric_features(row, names, binary, fallback=None):
    fallback = fallback or {}
    result = []
    for name in names:
        value = row.get(name)
        if value is None or not str(value).strip():
            value = fallback.get(name)
        if value is None or not str(value).strip():
            raise ValueError(f"Missing required feature: {name}")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"Non-finite feature: {name}")
        if name in binary and value not in (0, 1):
            raise ValueError(f"Binary feature {name} must be 0 or 1")
        result.append(value)
    return result


def connected_groups(pairs, patients):
    """Keep matched pairs and all observations of a patient in one component."""
    parent = list(range(len(pairs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, (pair, patient) in enumerate(zip(pairs, patients)):
        for key in (("pair", pair), ("patient", patient)):
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    roots = [find(i) for i in range(len(pairs))]
    ordered = {r: j for j, r in enumerate(dict.fromkeys(roots))}
    return np.asarray([f"g{ordered[r]:06d}" for r in roots])


def validate_pairs(pairs, labels):
    for pair in np.unique(pairs):
        target = labels[pairs == pair]
        if len(target) != 2 or sorted(target.tolist()) != [0, 1]:
            raise ValueError(f"Matched pair {pair} must contain one case and one control")


def record_inputs(row, base, mapping=None):
    """Read and validate the three model inputs at ECG acquisition time."""
    if float(row.get("sampling_rate_hz", "nan")) != 500:
        raise ValueError("sampling_rate_hz must be 500")
    quality = float(row.get("quality_score", "nan"))
    if not 80 <= quality <= 100:
        raise ValueError("quality_score must be between 80 and 100")
    if datetime.fromisoformat(row["clinical_time"]) > datetime.fromisoformat(row["ecg_time"]):
        raise ValueError("Clinical inputs must be available at or before ECG acquisition")
    path = base / row["waveform_path"]
    measurements = {}
    if path.suffix.lower() == ".xml":
        waveform, measurements = read_xml_waveform(path, mapping)
    elif path.suffix.lower() == ".npy":
        waveform = np.load(path, allow_pickle=False)
    else:
        raise ValueError("waveform_path must point to an XML or NPY file")
    waveform = validate_waveform(waveform)
    ecg = numeric_features(row, ECG_FEATURES, ECG_BINARY, measurements)
    clinical = numeric_features(row, CLINICAL_FEATURES, CLINICAL_BINARY)
    if clinical[0] < 18:
        raise ValueError("Adult participants must be at least 18 years old")
    if not 0 <= ecg[5] <= 180:
        raise ValueError("Frontal QRS-T angle must lie in [0, 180] degrees")
    return waveform, ecg, clinical


def prepare_data(manifest, output, cohort, xml_map=None):
    manifest, output = Path(manifest), Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    mapping = json.loads(Path(xml_map).read_text(encoding="utf-8")) if xml_map else None
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The manifest contains no records")
    waves, ecg, clinical, labels, ids, patients, pairs = [], [], [], [], [], [], []
    for row in rows:
        record_id = row.get("record_id", "").strip()
        try:
            if not record_id or not row.get("patient_id", "").strip():
                raise ValueError("record_id and patient_id are required")
            ecg_time = datetime.fromisoformat(row["ecg_time"])
            if cohort != "prediction" and ecg_time >= datetime.fromisoformat(row["index_time"]):
                raise ValueError("ECG acquisition must precede the assigned index time")
            waveform, ecg_values, clinical_values = record_inputs(row, manifest.parent, mapping)
            if cohort == "prediction":
                label, pair = float("nan"), row.get("pair_id") or record_id
            else:
                label = float(row["label"])
                pair = row.get("pair_id", "").strip()
                if label not in (0, 1) or not pair:
                    raise ValueError("Labeled cohorts require binary label and pair_id")
            waves.append(waveform)
            ecg.append(ecg_values)
            clinical.append(clinical_values)
            labels.append(label)
            ids.append(record_id)
            patients.append(row["patient_id"].strip())
            pairs.append(pair)
        except (ValueError, KeyError, TypeError, OSError, ET.ParseError) as error:
            raise ValueError(f"Record {record_id or '<missing>'}: {error}") from error
    if len(set(ids)) != len(ids):
        raise ValueError("record_id values must be unique")
    pairs, patients, labels = np.asarray(pairs), np.asarray(patients), np.asarray(labels, dtype=np.float32)
    if cohort != "prediction":
        validate_pairs(pairs, labels)
    arrays = dict(
        waveform=np.stack(waves), ecg=np.asarray(ecg, dtype=np.float32),
        clinical=np.asarray(clinical, dtype=np.float32), label=labels,
        record_id=np.asarray(ids), patient_id=patients, pair_id=pairs,
        group_id=connected_groups(pairs, patients), cohort=np.asarray(cohort),
        schema=np.asarray(json.dumps(feature_schema(), sort_keys=True)),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return {"records": len(rows), "pairs": len(np.unique(pairs)),
            "groups": len(np.unique(arrays["group_id"])), "cohort": cohort}


def load_data(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    required = {"waveform", "ecg", "clinical", "label", "record_id", "patient_id",
                "pair_id", "group_id", "cohort", "schema"}
    if set(data) != required or json.loads(str(data["schema"])) != feature_schema():
        raise ValueError("Dataset schema does not match this implementation")
    n = len(data["record_id"])
    shapes = {"waveform": (n, 12, 5000), "ecg": (n, len(ECG_FEATURES)),
              "clinical": (n, len(CLINICAL_FEATURES)), "label": (n,),
              "patient_id": (n,), "pair_id": (n,), "group_id": (n,)}
    if n == 0 or any(data[k].shape != shape for k, shape in shapes.items()):
        raise ValueError("Dataset arrays have inconsistent shapes")
    if len(set(data["record_id"].tolist())) != n:
        raise ValueError("Dataset contains duplicate record IDs")
    for key in ("waveform", "ecg", "clinical"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Non-finite values in {key}")
    if str(data["cohort"]) not in {"development", "external", "prediction"}:
        raise ValueError("Unknown dataset cohort")
    if str(data["cohort"]) != "prediction":
        validate_pairs(data["pair_id"], data["label"])
    expected_groups = connected_groups(data["pair_id"], data["patient_id"])
    if not np.array_equal(data["group_id"], expected_groups):
        raise ValueError("Dataset groups must preserve both patients and matched pairs")
    # Bound temporary float64 work arrays for large control pools.
    normalized = np.empty(data["waveform"].shape, dtype=np.float32)
    for start in range(0, n, 32):
        normalized[start:start + 32] = normalize_waveforms(data["waveform"][start:start + 32])
    data["waveform"] = normalized
    return data


class FeatureScaler:
    def __init__(self, state):
        self.state = state

    @classmethod
    def fit(cls, data, indices):
        state = {}
        if not len(indices):
            raise ValueError("A nonempty training partition is required")
        for key, names, binary in (("ecg", ECG_FEATURES, ECG_BINARY),
                                   ("clinical", CLINICAL_FEATURES, CLINICAL_BINARY)):
            values = data[key][indices]
            minimum, width = values.min(axis=0), np.ptp(values, axis=0)
            for i, name in enumerate(names):
                if name in binary:
                    minimum[i], width[i] = 0, 1
            width[width == 0] = 1
            state[key] = {"minimum": minimum.tolist(), "width": width.tolist()}
        return cls(state)

    def transform(self, data, key, indices):
        state = self.state[key]
        return ((data[key][indices] - np.asarray(state["minimum"])) /
                np.asarray(state["width"])).astype(np.float32)


class ECGDataset(Dataset):
    def __init__(self, data, indices, scaler):
        self.data, self.indices = data, np.asarray(indices)
        self.ecg = scaler.transform(data, "ecg", self.indices)
        self.clinical = scaler.transform(data, "clinical", self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return (torch.from_numpy(self.data["waveform"][j]), torch.from_numpy(self.ecg[i]),
                torch.from_numpy(self.clinical[i]), torch.tensor(self.data["label"][j]))
