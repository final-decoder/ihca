"""Build elapsed-time risk sets from hospital stays, ward histories and ECG snapshots."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from xml.etree.ElementTree import ParseError

from ihca.data import record_inputs
from ihca.schema import CLINICAL_FEATURES, ECG_FEATURES


def read_table(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        missing = sorted(set(required) - set(fields))
        if missing or len(fields) != len(set(fields)):
            raise ValueError(f"{path}: missing columns {missing} or duplicate column names")
        for number, row in enumerate(reader, start=2):
            if None in row or None in row.values():
                raise ValueError(f"{path}, line {number}: field count differs from header")
            yield {key: value.strip() for key, value in row.items()}


def timestamp(value):
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.utcoffset() is None:
        instant = instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc)


@dataclass
class Stay:
    identifier: str
    patient: str
    age: float
    sex: int
    admission: datetime
    discharge: datetime
    arrest: datetime | None
    eligible: bool
    rhythm: str
    etiology: str
    spells: list = field(default_factory=list)
    starts: list = field(default_factory=list)
    recordings: list = field(default_factory=list)
    times: list = field(default_factory=list)

    def department(self, index):
        i = bisect_right(self.starts, index) - 1
        if i >= 0:
            _, end, ward = self.spells[i]
            if index < end or index == end == self.discharge:
                return ward
        return None

    def latest(self, index):
        i = bisect_left(self.times, index)
        return self.recordings[i - 1][2] if i else None

    @property
    def risk_end(self):
        return self.arrest or self.discharge


def load_stays(path):
    stays = {}
    fields = ("encounter_id", "patient_id", "age", "sex", "admission_time",
              "discharge_time", "first_ihca_time", "case_eligible")
    for row in read_table(path, fields):
        key, patient = row["encounter_id"], row["patient_id"]
        if not key or not patient or key in stays:
            raise ValueError(f"Empty patient or missing/duplicate encounter ID: {key}")
        admission, discharge = (timestamp(row[k]) for k in ("admission_time", "discharge_time"))
        arrest = timestamp(row["first_ihca_time"]) if row["first_ihca_time"] else None
        if discharge <= admission or (arrest and not admission < arrest <= discharge):
            raise ValueError(f"{key}: invalid admission, discharge or first arrest time")
        age, sex = float(row["age"]), float(row["sex"])
        if not math.isfinite(age) or age < 0 or sex not in (0, 1):
            raise ValueError(f"{key}: invalid age or sex")
        if row["case_eligible"] not in ("0", "1") or (row["case_eligible"] == "1" and not arrest):
            raise ValueError(f"{key}: case_eligible must be 0/1 and a case requires a first arrest")
        stays[key] = Stay(key, patient, age, int(sex), admission, discharge, arrest,
                          row["case_eligible"] == "1", row.get("rhythm", ""), row.get("etiology", ""))
    return stays


def encounter(stays, row):
    key = row["encounter_id"]
    if key not in stays:
        raise ValueError(f"Unknown encounter ID: {key}")
    return stays[key]


def load_departments(path, stays):
    for row in read_table(path, ("encounter_id", "department", "start_time", "end_time")):
        stay = encounter(stays, row)
        start, end = timestamp(row["start_time"]), timestamp(row["end_time"])
        ward = row["department"]
        if not ward or not stay.admission <= start < end <= stay.discharge:
            raise ValueError(f"{stay.identifier}: invalid department or ward interval")
        stay.spells.append((start, end, ward))
    for stay in stays.values():
        stay.spells.sort()
        if any(left[1] > right[0] for left, right in zip(stay.spells, stay.spells[1:])):
            raise ValueError(f"{stay.identifier}: overlapping department intervals")
        stay.starts = [start for start, _, _ in stay.spells]


def load_recordings(path, stays, mapping):
    base = Path(path).resolve().parent
    fields = ("encounter_id", "record_id", "waveform_path", "sampling_rate_hz",
              "quality_score", "ecg_time", "clinical_time", *CLINICAL_FEATURES[2:])
    seen = set()
    screened_out = 0
    for row in read_table(path, fields):
        stay = encounter(stays, row)
        record = row["record_id"]
        if not record or record in seen:
            raise ValueError(f"Missing or duplicate ECG record ID: {record}")
        seen.add(record)
        for key, value in (("age", stay.age), ("sex", stay.sex)):
            if row.get(key) and float(row[key]) != value:
                raise ValueError(f"{record}: {key} differs between encounter and ECG")
            row[key] = str(value)
        if row.get("patient_id") and row["patient_id"] != stay.patient:
            raise ValueError(f"{record}: patient differs between encounter and ECG")
        instant = timestamp(row["ecg_time"])
        clinical = timestamp(row["clinical_time"])
        if not stay.admission <= instant < stay.risk_end or stay.age < 18:
            screened_out += 1
            continue
        row.update(ecg_time=instant.isoformat(), clinical_time=clinical.isoformat())
        if row.get("score_time"):
            row["score_time"] = timestamp(row["score_time"]).isoformat()
        try:
            _, ecg, _ = record_inputs(row, base, mapping)
        except (ValueError, EOFError, ParseError):
            screened_out += 1
            continue
        row.update(zip(ECG_FEATURES, ecg))
        row["waveform_path"] = str((base / row["waveform_path"]).resolve())
        stay.recordings.append((instant, record, row))
    for stay in stays.values():
        # Stable source IDs disambiguate simultaneous valid ECGs independently of export order.
        stay.recordings.sort(key=lambda recording: recording[:2])
        stay.times = [time for time, _, _ in stay.recordings]
    return screened_out


class ControlIndex:
    def __init__(self, stays):
        by_ward = defaultdict(list)
        for stay in stays.values():
            if stay.age >= 18 and stay.recordings:
                for ward in {spell[2] for spell in stay.spells}:
                    by_ward[stay.sex, ward].append(stay)
        self.strata = {}
        for key, items in by_ward.items():
            items.sort(key=lambda s: (s.age, s.identifier))
            self.strata[key] = ([s.age for s in items], items)

    def nearest(self, case, ward, elapsed):
        ages, items = self.strata.get((case.sex, ward), ([], []))
        right = bisect_left(ages, case.age)
        left = right - 1
        best = None
        while left >= 0 or right < len(items):
            lower = case.age - ages[left] if left >= 0 else math.inf
            upper = ages[right] - case.age if right < len(ages) else math.inf
            distance = min(lower, upper)
            if best and distance > best[0]:
                break
            if lower <= upper:
                control = items[left]
                left -= 1
            else:
                control = items[right]
                right += 1
            index = control.admission + elapsed
            if (control.patient == case.patient or index >= control.risk_end
                    or control.department(index) != ward):
                continue
            row = control.latest(index)
            if row is not None and (best is None or control.identifier < best[1].identifier):
                best = (distance, control, row)
        return (best[1], best[2]) if best else None


def matched_row(case, stay, snapshot, ward, elapsed, label, parent):
    try:
        wave = os.path.relpath(snapshot["waveform_path"], parent)
    except ValueError:  # Windows paths on different volumes
        wave = snapshot["waveform_path"]
    return {**snapshot, "encounter_id": stay.identifier, "record_id": f"{case.identifier}_{label}",
            "source_record_id": snapshot["record_id"], "patient_id": stay.patient,
            "pair_id": case.identifier, "label": label, "waveform_path": wave,
            "admission_time": stay.admission.isoformat(), "at_risk_until": stay.risk_end.isoformat(),
            "index_time": (stay.admission + elapsed).isoformat(), "index_department": ward,
            "rhythm": case.rhythm if label else "", "etiology": case.etiology if label else ""}


def assemble(encounters, departments, ecgs, output, xml_map=None):
    dest = Path(output).resolve()
    if dest.exists():
        raise FileExistsError(f"Output already exists: {dest}")
    mapping = json.loads(Path(xml_map).read_text(encoding="utf-8-sig")) if xml_map else None
    stays = load_stays(encounters)
    load_departments(departments, stays)
    excluded = load_recordings(ecgs, stays, mapping)
    controls = ControlIndex(stays)
    rows = []
    missing_ecg = 0
    for key in sorted(stays):
        case = stays[key]
        if not case.eligible or case.age < 18:
            continue
        snapshot = case.latest(case.arrest)
        if snapshot is None:
            missing_ecg += 1
            continue
        ward = case.department(case.arrest)
        if ward is None:
            raise ValueError(f"{key}: missing department at the arrest time")
        elapsed = case.arrest - case.admission
        choice = controls.nearest(case, ward, elapsed)
        if choice is None:
            raise ValueError(f"{key}: no eligible control at the assigned elapsed time in {ward}")
        for label, stay, row in ((1, case, snapshot), (0, *choice)):
            rows.append(matched_row(case, stay, row, ward, elapsed, label, dest.parent))
    if not rows:
        raise ValueError("No eligible adult cases with a complete pre-index ECG")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8", newline="") as stream:
        try:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        except BaseException:
            stream.close()
            dest.unlink()
            raise
    return {"encounters": len(stays), "screened_out_ecgs": excluded,
            "cases_without_ecg": missing_ecg, "pairs": len(rows) // 2}
