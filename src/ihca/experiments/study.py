"""Align study inputs, saved partitions and record-level predictions."""

from contextlib import contextmanager
from functools import cached_property
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from ihca.data import load_data
from ihca.engine import read_ensemble
from ihca.schema import CLINICAL_FEATURES, ECG_FEATURES


def read_json(path):
    return Path(path).read_text(encoding="utf-8-sig")


def keyed_csv(path, ids):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("record_id")
    if not frame.index.is_unique or frame.index.hasnans or "" in frame.index:
        raise ValueError(f"Missing or duplicate record IDs in {path}")
    if set(frame.index) != set(ids):
        raise ValueError(f"Record IDs in {path} must exactly match the dataset")
    return frame.loc[list(ids)].copy()


def aligned_metadata(data, path):
    frame = keyed_csv(path, data["record_id"])
    for key in ("patient_id", "pair_id"):
        if key in frame:
            provided = frame[key].astype(str).str.strip().ne("")
            if not np.array_equal(frame.loc[provided, key].values, data[key][provided]):
                raise ValueError(f"{key} mismatch in {path}")
        frame[key] = data[key]
    if "label" in frame and not np.array_equal(
        pd.to_numeric(frame["label"]), data["label"], equal_nan=True
    ):
        raise ValueError(f"Outcome mismatch in {path}")
    frame["label"] = data["label"]
    frame["group_id"] = data["group_id"]
    for domain, names in (("clinical", CLINICAL_FEATURES), ("ecg", ECG_FEATURES)):
        for j, name in enumerate(names):
            values = data[domain][:, j]
            if name in frame:
                supplied = frame[name].astype(str).str.strip() != ""
                if not np.allclose(
                    pd.to_numeric(frame.loc[supplied, name]), values[supplied], rtol=1e-5
                ):
                    raise ValueError(f"{name} differs from prepared input in {path}")
            frame[name] = values
    return frame


def read_predictions(path, data):
    frame = keyed_csv(path, data["record_id"])
    for key in ("patient_id", "pair_id", "group_id"):
        if not np.array_equal(frame[key].values, data[key]):
            raise ValueError(f"{key} mismatch in {path}")
    for key in ("score", "label", "threshold", "prediction"):
        frame[key] = pd.to_numeric(frame[key])
    if (
        not np.array_equal(frame.label, data["label"])
        or not np.isfinite(frame[["score", "threshold"]]).all().all()
    ):
        raise ValueError(f"Invalid scores or outcome mismatch in {path}")
    if not frame.score.between(0, 1).all() or not np.array_equal(
        frame.prediction, frame.score >= frame.threshold
    ):
        raise ValueError(f"Invalid score or operating point in {path}")
    if "fold" in frame:
        frame["fold"] = pd.to_numeric(frame.fold)
    return frame


def fold_partitions(directory, data):
    ids = pd.Index(data["record_id"])
    splits = json.loads(read_json(Path(directory) / "splits.json"))
    if len(splits) < 2 or {s["fold"] for s in splits} != set(range(len(splits))):
        raise ValueError("Invalid saved fold numbering")
    coverage = np.zeros(len(ids), int)
    for split in sorted(splits, key=lambda row: row["fold"]):
        ix = [ids.get_indexer(split[k]) for k in ("fit", "inner", "heldout")]
        if any(not len(a) for a in ix) or not np.array_equal(
            np.sort(np.concatenate(ix)), np.arange(len(ids))
        ):
            raise ValueError("Saved split must partition all records exactly once")
        groups = [set(data["group_id"][a]) for a in ix]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("A matched group crosses fitting, inner or held-out partitions")
        coverage[ix[2]] += 1
        yield (split["fold"], *ix)
    if not np.all(coverage == 1):
        raise ValueError("Held-out partitions must cover every record exactly once")


@contextmanager
def new_result_directory(destination):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory: {destination}")
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".ihca-") as temporary:
        stage = Path(temporary) / "results"
        stage.mkdir()
        yield stage
        if destination.exists():
            raise FileExistsError(destination)
        stage.rename(destination)


class Study:
    def __init__(self, path):
        self.base = Path(path).resolve().parent
        self.cfg = json.loads(read_json(path))
        self.seed = int(self.cfg.get("seed", 42))
        self.repeats = int(self.cfg.get("bootstrap", 2000))
        if self.repeats < 2:
            raise ValueError("At least two bootstrap draws are required")

    def path(self, name):
        return self.base / self.cfg["paths"][name]

    def model(self, mode):
        return self.base / self.cfg["models"][mode]

    @cached_property
    def dev(self):
        data = load_data(self.path("development"))
        if str(data["cohort"]) != "development":
            raise ValueError("Expected a prepared development cohort")
        return data

    @cached_property
    def ext(self):
        data = load_data(self.path("external"))
        if str(data["cohort"]) != "external" or set(data["patient_id"]) & set(self.dev["patient_id"]):
            raise ValueError("An independent external cohort with distinct patient IDs is required")
        return data

    @cached_property
    def dm(self):
        return aligned_metadata(self.dev, self.path("development_metadata"))

    @cached_property
    def em(self):
        return aligned_metadata(self.ext, self.path("external_metadata"))

    @cached_property
    def folds(self):
        return list(fold_partitions(self.model("full"), self.dev))

    @cached_property
    def ensemble(self):
        return read_ensemble(self.model("full"))

    @cached_property
    def oof(self):
        frame = read_predictions(self.model("full") / "oof_predictions.csv", self.dev)
        for fold, _, _, held in self.folds:
            if not (frame.fold.iloc[held] == fold).all():
                raise ValueError("Prediction folds disagree with saved partitions")
        return frame

    @cached_property
    def external(self):
        return read_predictions(self.path("external_predictions"), self.ext)
