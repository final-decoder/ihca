"""Multiscale residual ECG encoding, cross-lead attention and clinical fusion."""

import torch
from torch import nn

from .config import ModelConfig
from .schema import CLINICAL_FEATURES, ECG_FEATURES, LEADS


class MultiscaleResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernels):
        super().__init__()
        widths = [out_channels // len(kernels)] * len(kernels)
        for i in range(out_channels % len(kernels)):
            widths[i] += 1
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels, width, k, padding=k // 2, bias=False),
                          nn.BatchNorm1d(width), nn.GELU())
            for k, width in zip(kernels, widths)
        ])
        self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, bias=False),
                                      nn.BatchNorm1d(out_channels))
        self.activation = nn.GELU()
        self.pool = nn.MaxPool1d(2, stride=2)

    def forward(self, x):
        multiscale = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.pool(self.activation(multiscale + self.shortcut(x)))


class WaveformEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        blocks, previous = [], 1
        for width in config.channels:
            blocks.append(MultiscaleResidual(previous, width, config.kernels))
            previous = width
        self.blocks = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(config.waveform_dropout)
        self.lead_projection = nn.Linear(previous, config.attention_dim)
        self.lead_embedding = nn.Parameter(torch.empty(1, len(LEADS), config.attention_dim))
        nn.init.normal_(self.lead_embedding, std=0.02)
        self.attention = nn.MultiheadAttention(config.attention_dim, config.attention_heads,
                                               batch_first=True)

    def forward(self, waveform, return_attention=False):
        if waveform.ndim != 3 or tuple(waveform.shape[1:]) != (12, 5000):
            raise ValueError("Waveform input must have shape [batch, 12, 5000]")
        batch = waveform.shape[0]
        # Shared temporal filters preserve a separate representation for every lead.
        encoded = self.blocks(waveform.reshape(batch * len(LEADS), 1, 5000))
        tokens = self.dropout(encoded.mean(dim=-1)).reshape(batch, len(LEADS), -1)
        tokens = self.lead_projection(tokens) + self.lead_embedding
        attended, weights = self.attention(tokens, tokens, tokens,
                                           need_weights=return_attention,
                                           average_attn_weights=False)
        pooled = attended.mean(dim=1)
        return (pooled, weights) if return_attention else pooled


class IHCAModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.waveform = WaveformEncoder(config) if config.mode in {"full", "waveform"} else None
        self.clinical = None
        if config.mode != "waveform":
            a, b = config.clinical_hidden
            self.clinical = nn.Sequential(
                nn.Linear(len(CLINICAL_FEATURES), a), nn.BatchNorm1d(a), nn.GELU(),
                nn.Linear(a, b), nn.BatchNorm1d(b), nn.GELU(),
            )
        self.ecg_projection = None
        if config.mode in {"full", "structured"}:
            width = len(ECG_FEATURES) + (config.attention_dim if config.mode == "full" else 0)
            self.ecg_projection = nn.Sequential(nn.Linear(width, config.ecg_embedding), nn.GELU())
        if config.mode == "waveform":
            fusion_input = config.attention_dim
        elif config.mode == "clinical":
            fusion_input = config.clinical_hidden[-1]
        else:
            fusion_input = config.ecg_embedding + config.clinical_hidden[-1]
        a, b = config.fusion_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, a), nn.GELU(), nn.Dropout(config.fusion_dropout),
            nn.Linear(a, b), nn.GELU(), nn.Dropout(config.fusion_dropout), nn.Linear(b, 1),
        )

    def score_from_embedding(self, embedding, ecg, clinical):
        """Full-model logits with a previously computed waveform representation."""
        joined = torch.cat((embedding, ecg), dim=1)
        fused = torch.cat((self.ecg_projection(joined), self.clinical(clinical)), dim=1)
        return self.fusion(fused).squeeze(-1)

    def forward(self, waveform, ecg, clinical):
        mode = self.config.mode
        if ecg.ndim != 2 or ecg.shape[1] != len(ECG_FEATURES):
            raise ValueError("ECG measurements must contain the ordered structured ECG features")
        if clinical.ndim != 2 or clinical.shape[1] != len(CLINICAL_FEATURES):
            raise ValueError("Clinical input must contain 21 ordered features")
        if mode == "waveform":
            fused = self.waveform(waveform)
        elif mode == "clinical":
            fused = self.clinical(clinical)
        elif mode == "full":
            return self.score_from_embedding(self.waveform(waveform), ecg, clinical)
        else:
            fused = torch.cat((self.ecg_projection(ecg), self.clinical(clinical)), dim=1)
        # Return logits for numerically stable BCEWithLogitsLoss; sigmoid is applied at inference.
        return self.fusion(fused).squeeze(-1)
