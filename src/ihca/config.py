"""Model and optimization configuration."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class ModelConfig:
    channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernels: list[int] = field(default_factory=lambda: [7, 15, 31])
    attention_dim: int = 128
    attention_heads: int = 4
    ecg_embedding: int = 256
    clinical_hidden: list[int] = field(default_factory=lambda: [64, 128])
    fusion_hidden: list[int] = field(default_factory=lambda: [128, 64])
    waveform_dropout: float = 0.3
    fusion_dropout: float = 0.5
    mode: str = "full"

    def __post_init__(self):
        if self.mode not in {"full", "clinical", "structured", "waveform"}:
            raise ValueError("mode must be full, clinical, structured or waveform")
        if len(self.channels) != 4 or any(c < len(self.kernels) for c in self.channels):
            raise ValueError("Four convolutional blocks with positive branch widths are required")
        if not self.kernels or any(k < 1 or k % 2 == 0 for k in self.kernels):
            raise ValueError("Convolution kernels must be positive odd integers")
        if self.attention_heads < 1 or self.attention_dim % self.attention_heads:
            raise ValueError("Attention dimension must be divisible by the head count")
        if len(self.clinical_hidden) != 2 or len(self.fusion_hidden) != 2:
            raise ValueError("Two clinical and two fusion layers are required")
        if not all(0 <= p < 1 for p in (self.waveform_dropout, self.fusion_dropout)):
            raise ValueError("Dropout probabilities must be in [0, 1)")


@dataclass
class TrainConfig:
    seed: int = 42
    folds: int = 5
    inner_fraction: float = 0.2
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    patience: int = 15
    min_delta: float = 0.0001
    bootstrap: int = 2000
    threads: int = 4

    def __post_init__(self):
        if self.folds < 2 or self.batch_size < 2 or self.epochs < 1:
            raise ValueError("folds >= 2, batch_size >= 2 and epochs >= 1 are required")
        if not 0 < self.inner_fraction < 0.5:
            raise ValueError("inner_fraction must be between 0 and 0.5")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.patience < 1:
            raise ValueError("Invalid optimizer or early-stopping settings")
        if self.bootstrap < 0 or self.threads < 1 or self.min_delta < 0:
            raise ValueError("Invalid bootstrap, thread or minimum-delta settings")


def read_config(path):
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(content) - {"model", "training"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    return ModelConfig(**content.get("model", {})), TrainConfig(**content.get("training", {}))


def config_dict(model, training):
    return {"model": asdict(model), "training": asdict(training)}
