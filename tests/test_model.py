from dataclasses import asdict

import numpy as np
import pytest
import torch

from ihca.config import ModelConfig
from ihca.data import FeatureScaler
from ihca.engine import ensemble_predict, load_checkpoint, predict_model, write_json
from ihca.model import IHCAModel
from ihca.schema import feature_schema

torch.set_num_threads(2)


@pytest.mark.parametrize("mode", ["full", "clinical", "structured", "waveform"])
def test_active_branches_receive_gradients_and_update(mode):
    torch.manual_seed(42)
    model = IHCAModel(ModelConfig(mode=mode))
    x, e, c = torch.randn(2, 12, 5000), torch.randn(2, 13), torch.randn(2, 21)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    logits = model(x, e, c)
    assert logits.shape == (2,)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.])).backward()
    for param in model.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()
    optimizer.step()
    for prefix in {name.split(".")[0] for name in before}:
        assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n.startswith(prefix + "."))


def test_attention_weights_are_across_twelve_leads():
    model = IHCAModel(ModelConfig()).eval()
    with torch.no_grad():
        embedding, weights = model.waveform(torch.randn(2, 12, 5000), return_attention=True)
    assert embedding.shape == (2, 128)
    assert weights.shape == (2, 4, 12, 12)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 12))


def test_save_reload_preserves_full_model_outputs(tmp_path):
    torch.manual_seed(12)
    config = ModelConfig()
    model = IHCAModel(config).eval()
    checkpoint = {"format_version": 1, "schema": feature_schema(), "model_config": asdict(config),
                  "model_state": model.state_dict(), "threshold": 0.7, "scaler": {}}
    path = tmp_path / "fold.pt"; torch.save(checkpoint, path)
    restored, _, loaded = load_checkpoint(path, "cpu")
    values = (torch.randn(2, 12, 5000), torch.randn(2, 13), torch.randn(2, 21))
    with torch.no_grad():
        torch.testing.assert_close(model(*values), restored(*values), rtol=0, atol=0)
    assert loaded["threshold"] == 0.7


def test_frozen_ensemble_averages_scores_and_retains_threshold(tmp_path):
    rng = np.random.default_rng(1)
    data = {"waveform": rng.normal(size=(2, 12, 5000)).astype(np.float32),
            "ecg": rng.normal(size=(2, 13)).astype(np.float32),
            "clinical": rng.normal(size=(2, 21)).astype(np.float32),
            "label": np.array([0, 1], dtype=np.float32), "patient_id": np.array(["e1", "e2"])}
    scaler = FeatureScaler.fit(data, [0, 1])
    outputs, entries = [], []
    for i, threshold in enumerate([0.3, 0.9]):
        torch.manual_seed(i)
        config = ModelConfig(mode="clinical")
        model = IHCAModel(config).eval()
        path = tmp_path / f"fold_{i}.pt"
        torch.save({"format_version": 1, "schema": feature_schema(), "model_config": asdict(config),
                    "model_state": model.state_dict(), "threshold": threshold, "scaler": scaler.state,
                    "fold": i, "development_patients": ["d1", "d2"]}, path)
        entries.append({"path": path.name, "threshold": threshold, "fold": i})
        outputs.append(predict_model(model, data, [0, 1], scaler, 2, "cpu"))
    manifest = {"schema": feature_schema(), "aggregation": "arithmetic_mean", "models": entries, "threshold": 0.6}
    write_json(tmp_path / "ensemble.json", manifest)
    scores, threshold = ensemble_predict(data, tmp_path, 2, "cpu", 2)
    np.testing.assert_allclose(scores, (outputs[0] + outputs[1]) / 2)
    assert threshold == 0.6
    data["label"] = 1 - data["label"]
    altered, same_threshold = ensemble_predict(data, tmp_path, 2, "cpu", 2)
    np.testing.assert_array_equal(scores, altered)
    assert same_threshold == threshold
    data["patient_id"][0] = "d1"
    with pytest.raises(ValueError, match="overlaps"):
        ensemble_predict(data, tmp_path, 2, "cpu", 2)
