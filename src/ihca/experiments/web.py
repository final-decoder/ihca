"""Local research interface for XML ingestion, frozen inference and attribution."""

import csv
from pathlib import Path
import tempfile

from flask import Flask, Response, render_template, request
from markupsafe import escape
import numpy as np
import torch

from ihca.data import load_data, prepare_data
from ihca.engine import predict_model
from ihca.schema import CLINICAL_FEATURES, CLINICAL_BINARY, ECG_FEATURES
from .attribution import FEATURES, explain_one, reference_folds


def create_app(study):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 33 * 1024 * 1024
    torch.set_num_threads(study.cfg["threads"])
    folds = list(reference_folds(study))
    development_patients = set(study.dev["patient_id"])

    @app.get("/")
    def index():
        return render_template(
            "intake.html", clinical=CLINICAL_FEATURES, binary=CLINICAL_BINARY, ecg=ECG_FEATURES
        )

    @app.post("/predict")
    def score():
        try:
            patient = request.form["patient_id"].strip()
            if not patient or patient in development_patients:
                raise ValueError("Provide a distinct research participant ID")
            with tempfile.TemporaryDirectory(prefix="ihca_request_") as tmp:
                folder = Path(tmp)
                upload = request.files.get("waveform")
                if upload is None or not upload.filename:
                    raise ValueError("Select a calibrated 12-lead XML file")
                upload.save(folder / "record.xml")
                row = {
                    name: request.form.get(name, "").strip()
                    for name in (*CLINICAL_FEATURES, *ECG_FEATURES)
                }
                row.update(
                    record_id="assessment",
                    patient_id=patient,
                    waveform_path="record.xml",
                    sampling_rate_hz=500,
                    quality_score=request.form["quality_score"],
                    ecg_time=request.form["ecg_time"],
                    clinical_time=request.form["clinical_time"],
                )
                with (folder / "record.csv").open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=row)
                    writer.writeheader()
                    writer.writerow(row)
                prepare_data(folder / "record.csv", folder / "record.npz", "prediction")
                data = load_data(folder / "record.npz")
                scores, bases, effects = [], [], []
                for fold, _, model, scaler, background in folds:
                    base, phi, score_value = explain_one(
                        model,
                        scaler,
                        data,
                        0,
                        background,
                        study.cfg["permutations"],
                        np.random.default_rng(study.seed + fold),
                    )
                    direct = predict_model(model, data, [0], scaler, 1, torch.device("cpu"))[0]
                    if not np.isclose(direct, score_value, atol=2e-5):
                        raise ValueError("Inconsistent ensemble score")
                    scores.append(float(direct))
                    bases.append(base)
                    effects.append(phi)
            score_value = float(np.mean(scores))
            effect = np.mean(effects, axis=0)
            scale = max(float(np.max(np.abs(effect))), 1e-12)
            ordered = sorted(zip(FEATURES, effect), key=lambda item: abs(item[1]), reverse=True)
            features = [(key, float(val), 100 * abs(val) / scale) for key, val in ordered]
            kwargs = dict(
                patient=patient,
                score=score_value,
                threshold=float(study.ensemble["threshold"]),
                base=float(np.mean(bases)),
                features=features,
                folds=len(scores),
                date=row["ecg_time"],
                clinical_count=len(CLINICAL_FEATURES),
            )
            css = (Path(app.static_folder) / "research.css").read_text(encoding="utf-8")
            html = render_template(
                "assessment.html", **kwargs, css=css, download=request.args.get("download") == "1"
            )
            response = Response(html, mimetype="text/html")
            if request.args.get("download") == "1":
                response.headers["Content-Disposition"] = (
                    "attachment; filename=ihca-research-report.html"
                )
            response.headers["Cache-Control"] = "no-store"
            return response
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return f"Input error: {escape(str(exc))}", 400

    return app
