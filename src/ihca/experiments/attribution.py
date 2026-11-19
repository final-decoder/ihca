"""Interventional permutation SHAP for structured features at a fixed ECG waveform."""

import numpy as np
import pandas as pd
import torch

from ihca.engine import load_checkpoint
from ihca.schema import CLINICAL_FEATURES, ECG_FEATURES
from .plots import attribution as attribution_plot

FEATURES = [*ECG_FEATURES, *CLINICAL_FEATURES]


def permutation_shap(x, background, predict, permutations, rng):
    """Average forward/reverse marginal contributions over empirical reference rows."""
    if permutations < 1 or not len(background):
        raise ValueError("At least one permutation and reference record are required")
    p = len(x)
    phi = np.zeros(p, dtype=float)
    base = 0.0
    for _ in range(permutations):
        reference = background[rng.integers(len(background))]
        order = rng.permutation(p)
        for path in (order, order[::-1]):
            states = np.tile(reference, (p + 1, 1))
            for step, feature in enumerate(path, 1):
                states[step:, feature] = x[feature]
            scores = np.asarray(predict(states), dtype=float)
            if scores.shape != (p + 1,) or not np.isfinite(scores).all():
                raise ValueError("Non-finite or misaligned SHAP evaluations")
            phi[path] += np.diff(scores)
            base += scores[0]
    phi /= 2 * permutations
    base /= 2 * permutations
    score = float(predict(x[None, :])[0])
    if abs(base + phi.sum() - score) > 2e-5:
        raise ValueError("SHAP contributions do not reconstruct the model score")
    return base, phi, score


def structured(scaler, data, ix):
    return np.column_stack([scaler.transform(data, "ecg", ix), scaler.transform(data, "clinical", ix)])


@torch.inference_mode()
def explain_one(model, scaler, data, index, background, permutations, rng):
    if model.config.mode != "full":
        raise ValueError("Structured attribution requires the full multimodal model")
    device = next(model.parameters()).device
    waveform = torch.as_tensor(data["waveform"][index : index + 1], device=device)
    embedding = model.waveform(waveform)
    x = structured(scaler, data, [index])[0]

    def score(values):
        values = torch.as_tensor(values, device=device, dtype=torch.float32)
        ecg, clinical = values.split([len(ECG_FEATURES), len(CLINICAL_FEATURES)], dim=1)
        return (
            model.score_from_embedding(embedding.expand(len(values), -1), ecg, clinical)
            .sigmoid()
            .cpu()
            .numpy()
        )

    return permutation_shap(x, background, score, permutations, rng)


def reference_folds(study):
    entries = {entry["fold"]: entry for entry in study.ensemble["models"]}
    if set(entries) != {fold for fold, *_ in study.folds}:
        raise ValueError("Frozen ensemble folds differ from the saved development partitions")
    development = set(study.dev["patient_id"])
    for fold, fit, _, held in study.folds:
        entry = entries[fold]
        model, scaler, checkpoint = load_checkpoint(
            study.model("full") / entry["path"], torch.device("cpu")
        )
        fit_patients = set(study.dev["patient_id"][fit])
        if (
            checkpoint["fold"] != fold
            or checkpoint["threshold"] != entry["threshold"]
            or model.config.mode != "full"
            or development != set(checkpoint["development_patients"])
            or fit_patients != set(checkpoint["fit_patients"])
        ):
            raise ValueError("Reference cohort or fold does not match the training checkpoint")
        yield fold, held, model, scaler, structured(scaler, study.dev, fit)


def explain(study, out):
    rows = []
    for fold, held, model, scaler, background in reference_folds(study):
        rng = np.random.default_rng(study.seed + fold)
        for ix in held:
            base, phi, score = explain_one(
                model, scaler, study.dev, ix, background, study.cfg["permutations"], rng
            )
            if not np.isclose(score, study.oof.score.iloc[ix], atol=2e-5, rtol=1e-5):
                raise ValueError("Attribution target differs from the saved held-out prediction")
            rows.append(
                {
                    "record_id": study.dev["record_id"][ix],
                    "fold": fold,
                    "base_value": base,
                    "score": score,
                    **dict(zip(FEATURES, phi)),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "structured_shap.csv", index=False)
    rank = frame[FEATURES].abs().mean().rename("mean_abs_shap").rename_axis("feature").reset_index()
    rank = rank.sort_values("mean_abs_shap", ascending=False)
    rank.to_csv(out / "feature_ranking.csv", index=False)
    inputs = pd.DataFrame(
        np.column_stack((study.dev["ecg"], study.dev["clinical"])),
        index=study.dev["record_id"],
        columns=FEATURES,
    )
    attribution_plot(rank, frame.set_index("record_id"), inputs, out / "feature_ranking.png")
