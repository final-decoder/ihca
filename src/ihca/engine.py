"""Grouped cross-validation, checkpoint management and frozen inference."""

import csv
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ModelConfig, config_dict
from .data import ECGDataset, FeatureScaler, load_data
from .metrics import binary_metrics, bootstrap_intervals, youden_threshold
from .model import IHCAModel
from .schema import feature_schema
from .splits import inner_split, outer_folds, training_batches


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty prediction table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def predict_model(model, data, indices, scaler, batch_size, device):
    loader = DataLoader(ECGDataset(data, indices, scaler), batch_size=batch_size, shuffle=False)
    model.eval()
    scores = []
    with torch.inference_mode():
        for waveform, ecg, clinical, _ in loader:
            score = model(waveform.to(device), ecg.to(device), clinical.to(device)).sigmoid()
            if not torch.isfinite(score).all():
                raise FloatingPointError("Non-finite model predictions")
            scores.append(score.cpu().numpy())
    return np.concatenate(scores)


def fit_fold(data, train_ix, inner_ix, model_config, training, device, fold):
    seed_everything(training.seed + fold, training.threads)
    scaler = FeatureScaler.fit(data, train_ix)
    model = IHCAModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate,
                                  weight_decay=training.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    dataset = ECGDataset(data, train_ix, scaler)
    rng = np.random.default_rng(training.seed + fold)
    history, best_state, best_epoch, best_auc = [], None, 0, -float("inf")
    stale, updates, first_update = 0, 0, {}
    for epoch in range(1, training.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = training_batches(len(dataset), training.batch_size, rng)
        loader = DataLoader(dataset, batch_sampler=batches)
        for waveform, ecg, clinical, label in loader:
            optimizer.zero_grad(set_to_none=True)
            before = {n: p.detach().clone() for n, p in model.named_parameters()} if updates == 0 else None
            logits = model(waveform.to(device), ecg.to(device), clinical.to(device))
            loss = criterion(logits, label.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is None or not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"Missing or non-finite gradient: {name}")
            if updates == 0:
                for name, param in model.named_parameters():
                    prefix = name.split(".")[0]
                    first_update.setdefault(prefix, {"gradient_l2_squared": 0.0, "update_l2_squared": 0.0})
                    first_update[prefix]["gradient_l2_squared"] += float(param.grad.detach().square().sum())
            optimizer.step()
            if before is not None:
                for name, param in model.named_parameters():
                    first_update[name.split(".")[0]]["update_l2_squared"] += float((param.detach() - before[name]).square().sum())
            updates += 1
            total_loss += float(loss.detach()) * len(label)
        scores = predict_model(model, data, inner_ix, scaler, training.batch_size, device)
        metric = binary_metrics(data["label"][inner_ix], scores)
        value = metric["auroc"]
        if value is None:
            raise ValueError("Inner validation requires both classes")
        history.append({"epoch": epoch, "training_loss": total_loss / len(dataset), "inner_auroc": value})
        print(json.dumps({"fold": fold, **history[-1]}), flush=True)
        if value > best_auc + training.min_delta:
            best_auc, best_epoch, stale = value, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= training.patience:
            break
    if best_state is None:
        raise RuntimeError("No finite inner-validation checkpoint was selected")
    model.load_state_dict(best_state)
    inner_scores = predict_model(model, data, inner_ix, scaler, training.batch_size, device)
    threshold = youden_threshold(data["label"][inner_ix], inner_scores)
    for name, values in first_update.items():
        if values["gradient_l2_squared"] <= 0 or values["update_l2_squared"] <= 0:
            raise RuntimeError(f"Model branch did not participate in optimization: {name}")
    return model, scaler, threshold, {
        "history": history, "best_epoch": best_epoch, "best_inner_auroc": best_auc,
        "optimizer_updates": updates, "first_update": first_update,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def prediction_rows(data, scores, thresholds, predictions=None, folds=None):
    thresholds = np.broadcast_to(thresholds, scores.shape)
    predictions = scores >= thresholds if predictions is None else predictions
    for i, score in enumerate(scores):
        row = {"record_id": str(data["record_id"][i]), "patient_id": str(data["patient_id"][i]),
               "pair_id": str(data["pair_id"][i]), "group_id": str(data["group_id"][i]),
               "score": float(score), "threshold": float(thresholds[i]), "prediction": int(predictions[i])}
        if np.isfinite(data["label"][i]):
            row["label"] = int(data["label"][i])
        if folds is not None:
            row["fold"] = int(folds[i])
        yield row


def train_cv(data_path, output, model_config, training, device_name="auto"):
    data, output = load_data(data_path), Path(output)
    if str(data["cohort"]) != "development":
        raise ValueError("Cross-validation can fit only a development dataset")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = get_device(device_name)
    folds = outer_folds(data["group_id"], training.folds, training.seed)
    n = len(data["label"])
    oof, thresholds = np.full(n, np.nan), np.full(n, np.nan)
    fold_ids, coverage = np.full(n, -1), np.zeros(n, dtype=int)
    details, models, splits = [], [], []
    write_json(output / "config.json", config_dict(model_config, training))
    for fold, heldout in enumerate(folds):
        available = np.setdiff1d(np.arange(n), heldout)
        train_ix, inner_ix = inner_split(available, data["group_id"], training.inner_fraction,
                                         training.seed + fold)
        split = {name: data["record_id"][ix].tolist() for name, ix in
                 (("fit", train_ix), ("inner", inner_ix), ("heldout", heldout))}
        split["fold"] = fold
        splits.append(split)
        model, scaler, threshold, result = fit_fold(data, train_ix, inner_ix, model_config,
                                                    training, device, fold)
        filename = f"fold_{fold}.pt"
        checkpoint = {
            "format_version": 1, "schema": feature_schema(), "model_config": asdict(model_config),
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "scaler": scaler.state, "threshold": threshold, "fold": fold,
            "best_epoch": result["best_epoch"],
            "fit_patients": sorted(set(data["patient_id"][train_ix].tolist())),
            "development_patients": sorted(set(data["patient_id"].tolist())),
        }
        torch.save(checkpoint, output / filename)
        # Reload the serialized model before producing its held-out predictions.
        restored, restored_scaler, _ = load_checkpoint(output / filename, device)
        oof[heldout] = predict_model(restored, data, heldout, restored_scaler, training.batch_size, device)
        thresholds[heldout], fold_ids[heldout] = threshold, fold
        coverage[heldout] += 1
        models.append({"path": filename, "fold": fold, "threshold": threshold})
        details.append({"fold": fold, "fit_records": len(train_ix), "inner_records": len(inner_ix),
                        "heldout_records": len(heldout), "threshold": threshold, **result})
    if not np.isfinite(oof).all() or not np.all(coverage == 1):
        raise RuntimeError("Each record must receive exactly one held-out prediction")
    predicted = oof >= thresholds
    metrics = binary_metrics(data["label"], oof, predictions=predicted)
    metrics.update(bootstrap_intervals(data["label"], oof, predicted, data["group_id"],
                                       training.bootstrap, training.seed))
    ensemble = {"format_version": 1, "models": models,
                "threshold": float(np.median([x["threshold"] for x in models])),
                "aggregation": "arithmetic_mean", "schema": feature_schema()}
    write_json(output / "ensemble.json", ensemble)
    write_json(output / "splits.json", splits)
    write_json(output / "training.json", details)
    write_json(output / "oof_metrics.json", metrics)
    write_csv(output / "oof_predictions.csv", prediction_rows(data, oof, thresholds, folds=fold_ids))
    return {"folds": len(models), "records": n, "optimizer_updates": sum(d["optimizer_updates"] for d in details),
            "parameters": details[0]["parameters"], "oof_auroc": metrics["auroc"],
            "ensemble_threshold": ensemble["threshold"]}


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("schema") != feature_schema():
        raise ValueError("Unsupported checkpoint schema")
    model = IHCAModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    threshold = float(checkpoint["threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid checkpoint threshold")
    return model, FeatureScaler(checkpoint["scaler"]), checkpoint


def read_ensemble(model_dir):
    model_dir = Path(model_dir).resolve()
    ensemble = json.loads((model_dir / "ensemble.json").read_text(encoding="utf-8"))
    if ensemble.get("schema") != feature_schema() or ensemble.get("aggregation") != "arithmetic_mean":
        raise ValueError("Invalid ensemble definition")
    if len(ensemble.get("models", [])) < 2:
        raise ValueError("The frozen ensemble requires at least two fold models")
    paths, folds = set(), set()
    for entry in ensemble["models"]:
        path = (model_dir / entry["path"]).resolve()
        if path.parent != model_dir or path in paths or entry["fold"] in folds:
            raise ValueError("Ensemble checkpoint paths and folds must be distinct in the model directory")
        paths.add(path)
        folds.add(entry["fold"])
    thresholds = [entry["threshold"] for entry in ensemble["models"]]
    if ensemble["threshold"] != float(np.median(thresholds)):
        raise ValueError("Ensemble threshold must equal the median of saved inner-validation thresholds")
    return ensemble


def ensemble_predict(data, model_dir, batch_size=64, device_name="auto", threads=4):
    torch.set_num_threads(threads)
    model_dir, device = Path(model_dir).resolve(), get_device(device_name)
    ensemble = read_ensemble(model_dir)
    scores = []
    for entry in ensemble["models"]:
        path = model_dir / entry["path"]
        model, scaler, checkpoint = load_checkpoint(path, device)
        if set(data["patient_id"].tolist()) & set(checkpoint["development_patients"]):
            raise ValueError("Frozen evaluation/inference input overlaps development patients")
        scores.append(predict_model(model, data, np.arange(len(data["label"])), scaler, batch_size, device))
        if entry["fold"] != checkpoint["fold"] or entry["threshold"] != checkpoint["threshold"]:
            raise ValueError("Ensemble metadata does not match the saved fold checkpoint")
    return np.mean(np.stack(scores), axis=0), float(ensemble["threshold"])


def evaluate(data_path, model_dir, output, bootstrap=2000, seed=1234,
             batch_size=64, device_name="auto", threads=4):
    data, output = load_data(data_path), Path(output)
    if str(data["cohort"]) != "external":
        raise ValueError("Frozen evaluation requires an external labeled cohort")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output directory must be empty: {output}")
    score, threshold = ensemble_predict(data, model_dir, batch_size, device_name, threads)
    predictions = score >= threshold
    metrics = binary_metrics(data["label"], score, threshold)
    metrics["threshold"] = threshold
    metrics.update(bootstrap_intervals(data["label"], score, predictions, data["group_id"], bootstrap, seed))
    write_json(output / "metrics.json", metrics)
    write_csv(output / "predictions.csv", prediction_rows(data, score, threshold))
    return metrics


def predict(data_path, model_dir, output, batch_size=64, device_name="auto", threads=4):
    if Path(output).exists():
        raise FileExistsError(f"Prediction output already exists: {output}")
    data = load_data(data_path)
    score, threshold = ensemble_predict(data, model_dir, batch_size, device_name, threads)
    write_csv(output, prediction_rows(data, score, threshold))
    return {"records": len(score), "threshold": threshold}
