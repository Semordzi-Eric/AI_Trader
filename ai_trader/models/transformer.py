"""Transformer encoder for directional bias.

Architecture: project features → add sinusoidal positional encoding → N transformer
encoder layers → take the last token's representation → MLP head → scalar logit.

Why a Transformer over LSTM here? Self-attention scales O(L²) with sequence length
but our windows are 64–256 bars, which is trivial. Attention also gives interpretable
per-bar weights, useful when debugging signals.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (non-learned)."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # x: (B, L, d)
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    """Encoder → mean-pooling → MLP → scalar (logit for sign, or regression target)."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, F)
        h = self.input_proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        # Mean-pool over time. Last-token pooling is also valid; mean is more stable.
        h = self.norm(h.mean(dim=1))
        return self.head(h).squeeze(-1)         # (B,) for output_dim=1
