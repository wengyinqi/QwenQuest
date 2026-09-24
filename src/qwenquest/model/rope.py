"""Rotary position embedding (NeoX / "rotate_half" layout) with optional YaRN.

Numerics follow ``transformers`` (``Qwen3MoeRotaryEmbedding`` and
``_compute_yarn_parameters``): frequencies and cos/sin in float32, then
cast to the activation dtype before rotating.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

__all__ = ["RotaryEmbedding", "apply_rotary", "compute_inv_freq", "rotate_half"]


def compute_inv_freq(
    head_dim: int,
    rope_theta: float,
    max_position_embeddings: int,
    rope_scaling: dict[str, Any] | None,
) -> tuple[torch.Tensor, float]:
    """Return ``(inv_freq [head_dim // 2] float32 on CPU, attention_scaling)``.

    Always computed on CPU, even under ``with torch.device("meta")`` (used by
    the checkpoint loader), because RoPE frequencies are not loaded weights.
    """
    dim = head_dim
    cpu = torch.device("cpu")
    pos_freqs = rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64, device=cpu).float() / dim)
    if not rope_scaling or rope_scaling.get("rope_type", "default") == "default":
        return 1.0 / pos_freqs, 1.0
    if rope_scaling["rope_type"] != "yarn":
        raise NotImplementedError(f"rope_type={rope_scaling['rope_type']!r}")

    original_max = rope_scaling.get("original_max_position_embeddings", max_position_embeddings)
    factor = rope_scaling.get("factor")
    if factor is None:
        factor = max_position_embeddings / original_max
    attention_factor = rope_scaling.get("attention_factor")
    mscale = rope_scaling.get("mscale")
    mscale_all_dim = rope_scaling.get("mscale_all_dim")

    def get_mscale(scale: float, m: float = 1.0) -> float:
        return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0

    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
        else:
            attention_factor = get_mscale(factor)

    beta_fast = rope_scaling.get("beta_fast") or 32
    beta_slow = rope_scaling.get("beta_slow") or 1
    truncate = rope_scaling.get("truncate", True)

    def correction_dim(num_rotations: float) -> float:
        return (dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (
            2 * math.log(rope_theta)
        )

    low, high = correction_dim(beta_fast), correction_dim(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001  # avoid a zero-width ramp
    ramp = (torch.arange(dim // 2, dtype=torch.float32, device=cpu) - low) / (high - low)
    ramp = ramp.clamp(0, 1)
    extrapolation_factor = 1 - ramp
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    inv_freq = (
        inv_freq_interpolation * (1 - extrapolation_factor)
        + inv_freq_extrapolation * extrapolation_factor
    )
    return inv_freq, float(attention_factor)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x``: ``[T, heads, head_dim]``; ``cos``/``sin``: ``[T, head_dim]``."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Holds no parameters or buffers, so meta-device model init stays trivial."""

    def __init__(
        self,
        head_dim: int,
        rope_theta: float,
        max_position_embeddings: int,
        rope_scaling: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self._inv_freq_cpu, self.attention_scaling = compute_inv_freq(
            head_dim, rope_theta, max_position_embeddings, rope_scaling
        )
        self._inv_freq_by_device: dict[torch.device, torch.Tensor] = {}

    def _inv_freq(self, device: torch.device) -> torch.Tensor:
        if device not in self._inv_freq_by_device:
            self._inv_freq_by_device[device] = self._inv_freq_cpu.to(device)
        return self._inv_freq_by_device[device]

    def cos_sin(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self._inv_freq(positions.device)
        freqs = positions.float().unsqueeze(1) * inv_freq.unsqueeze(0)  # [T, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype), sin.to(dtype)

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.cos_sin(positions, q.dtype)
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
