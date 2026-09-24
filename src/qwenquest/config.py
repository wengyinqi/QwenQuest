"""Configuration objects.

* :class:`Qwen3MoeConfig` - architecture of the Qwen3-MoE family
  (Qwen3-30B-A3B and its 2507 Instruct / Thinking refreshes).
  :meth:`Qwen3MoeConfig.from_pretrained` reads a checkpoint's ``config.json``
  in both the official Qwen layout (``num_experts``, ``rope_theta``,
  ``rope_scaling``) and the transformers>=5 layout (``num_local_experts``,
  ``rope_parameters``).
* :class:`SparseConfig` + :func:`parse_hisparse_config` - the JSON SGLang
  takes through ``--hisparse-config``, parsed with the same defaults and
  validation as the upstream ``hisparse_quest`` branch
  (``sglang/srt/mem_cache/sparsity/factory.py::_parse_sparse_config``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "PRESETS",
    "QWEN3_30B_A3B",
    "QWEN3_30B_A3B_2507",
    "Qwen3MoeConfig",
    "SparseConfig",
    "count_parameters",
    "parse_hisparse_config",
    "tiny_config",
]


# --------------------------------------------------------------------------- #
# Model architecture
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Qwen3MoeConfig:
    """Hyper-parameters of a Qwen3-MoE decoder (``model_type == "qwen3_moe"``).

    The defaults are Qwen3-30B-A3B-{Instruct,Thinking}-2507, the model HiSparse
    pairs with Quest: 48 layers, 32 query heads / 4 KV heads (GQA group of 8),
    head_dim 128, 128 experts with 8 routed per token (30.5B total / 3.3B active
    parameters).  The original April-2025 Qwen3-30B-A3B shares the architecture
    and differs only in ``rope_theta`` (1e6) and ``max_position_embeddings``
    (40960).  When a checkpoint is available, prefer :meth:`from_pretrained`.
    """

    vocab_size: int = 151_936
    hidden_size: int = 2048
    num_hidden_layers: int = 48
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    # Dense-MLP width, only used by layers listed in ``mlp_only_layers``
    # (none in Qwen3-30B-A3B: every layer is MoE).
    intermediate_size: int = 6144
    moe_intermediate_size: int = 768
    num_experts: int = 128
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple[int, ...] = ()
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    # None for plain RoPE, or e.g. {"rope_type": "yarn", "factor": 4.0,
    # "original_max_position_embeddings": 32768} for YaRN context extension.
    rope_scaling: dict[str, Any] | None = field(default=None, hash=False)
    max_position_embeddings: int = 262_144
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    bos_token_id: int | None = 151_643
    eos_token_id: int | tuple[int, ...] | None = 151_645

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
        if self.hidden_act != "silu":
            raise ValueError(f"Qwen3-MoE uses SiLU; got hidden_act={self.hidden_act!r}")
        if not 0 < self.num_experts_per_tok <= max(self.num_experts, 1):
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        rope_type = (self.rope_scaling or {}).get("rope_type", "default")
        if rope_type not in ("default", "yarn"):
            raise NotImplementedError(f"rope_type={rope_type!r} is not supported (default, yarn)")

    # ------------------------------------------------------------ derived

    @property
    def num_kv_groups(self) -> int:
        """Query heads per KV head (8 for Qwen3-30B-A3B)."""
        return self.num_attention_heads // self.num_key_value_heads

    def is_moe_layer(self, layer_id: int) -> bool:
        """Same rule as ``Qwen3MoeDecoderLayer`` in transformers / SGLang."""
        return (
            layer_id not in self.mlp_only_layers
            and self.num_experts > 0
            and (layer_id + 1) % self.decoder_sparse_step == 0
        )

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        if self.eos_token_id is None:
            return ()
        if isinstance(self.eos_token_id, int):
            return (self.eos_token_id,)
        return tuple(self.eos_token_id)

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """K + V bytes one token occupies across all layers."""
        return 2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim * dtype_bytes

    # ------------------------------------------------------------ (de)serialisation

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Qwen3MoeConfig:
        d = dict(raw)
        model_type = d.get("model_type", "qwen3_moe")
        if model_type != "qwen3_moe":
            raise ValueError(f"expected a qwen3_moe config, got model_type={model_type!r}")
        num_experts = d.get("num_experts", d.get("num_local_experts"))
        if num_experts is None:
            raise ValueError("config has neither 'num_experts' nor 'num_local_experts'")
        rope_theta, rope_scaling = _parse_rope(d)
        eos = d.get("eos_token_id")
        if isinstance(eos, list):
            eos = tuple(eos)
        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            head_dim=d.get("head_dim") or d["hidden_size"] // d["num_attention_heads"],
            intermediate_size=d.get("intermediate_size", 6144),
            moe_intermediate_size=d["moe_intermediate_size"],
            num_experts=num_experts,
            num_experts_per_tok=d["num_experts_per_tok"],
            norm_topk_prob=d.get("norm_topk_prob", False),
            decoder_sparse_step=d.get("decoder_sparse_step", 1),
            mlp_only_layers=tuple(d.get("mlp_only_layers") or ()),
            hidden_act=d.get("hidden_act", "silu"),
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=d.get("max_position_embeddings", 32_768),
            attention_bias=d.get("attention_bias", False),
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            bos_token_id=d.get("bos_token_id"),
            eos_token_id=eos,
        )

    @classmethod
    def from_pretrained(cls, path: str | os.PathLike[str]) -> Qwen3MoeConfig:
        """Read ``<path>/config.json`` (or ``path`` itself if it is a file)."""
        cfg_path = os.path.join(path, "config.json") if os.path.isdir(path) else os.fspath(path)
        with open(cfg_path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model_type"] = "qwen3_moe"
        d["architectures"] = ["Qwen3MoeForCausalLM"]
        d["mlp_only_layers"] = list(self.mlp_only_layers)
        if isinstance(self.eos_token_id, tuple):
            d["eos_token_id"] = list(self.eos_token_id)
        return d


def _parse_rope(d: Mapping[str, Any]) -> tuple[float, dict[str, Any] | None]:
    """Return ``(rope_theta, rope_scaling)`` from either config.json layout."""
    params = dict(d.get("rope_parameters") or {})
    theta = d.get("rope_theta", params.get("rope_theta", 10_000.0))
    scaling = d.get("rope_scaling")
    if scaling is None and params:
        scaling = {k: v for k, v in params.items() if k != "rope_theta"}
    if scaling is None:
        return float(theta), None
    scaling = dict(scaling)
    rope_type = scaling.pop("type", None) or scaling.get("rope_type") or "default"
    if rope_type == "default":
        return float(theta), None
    scaling["rope_type"] = rope_type
    return float(theta), scaling


QWEN3_30B_A3B_2507 = Qwen3MoeConfig()
"""Qwen3-30B-A3B-{Instruct,Thinking}-2507 (the model used in the HiSparse paper)."""

QWEN3_30B_A3B = Qwen3MoeConfig(rope_theta=1_000_000.0, max_position_embeddings=40_960)
"""The original Qwen3-30B-A3B release (April 2025)."""


def tiny_config(**overrides: Any) -> Qwen3MoeConfig:
    """A few-hundred-KB Qwen3-MoE with the same *shape* of computation.

    Keeps the ratios that matter for Quest (GQA group of 4, several KV heads)
    so every code path - GQA averaging, per-head bounds, MoE routing - is
    exercised on CPU in milliseconds.
    """
    base: dict[str, Any] = {
        "vocab_size": 256,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "intermediate_size": 128,
        "moe_intermediate_size": 32,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "rope_theta": 10_000.0,
        "max_position_embeddings": 4096,
        "bos_token_id": None,
        "eos_token_id": None,
    }
    base.update(overrides)
    return Qwen3MoeConfig(**base)


PRESETS: dict[str, Qwen3MoeConfig] = {
    "qwen3-30b-a3b": QWEN3_30B_A3B,
    "qwen3-30b-a3b-instruct-2507": QWEN3_30B_A3B_2507,
    "qwen3-30b-a3b-thinking-2507": QWEN3_30B_A3B_2507,
    "tiny": tiny_config(),
}


def count_parameters(cfg: Qwen3MoeConfig) -> tuple[int, int]:
    """Return ``(total, active_per_token)`` parameter counts."""
    h, d = cfg.hidden_size, cfg.head_dim
    attn = h * d * (cfg.num_attention_heads * 2 + cfg.num_key_value_heads * 2) + 2 * d
    if cfg.attention_bias:
        attn += d * (cfg.num_attention_heads * 2 + cfg.num_key_value_heads * 2)
    norms = 2 * h
    expert = 3 * h * cfg.moe_intermediate_size
    router = cfg.num_experts * h
    dense_mlp = 3 * h * cfg.intermediate_size
    total = active = 0
    for layer_id in range(cfg.num_hidden_layers):
        if cfg.is_moe_layer(layer_id):
            total += attn + norms + router + cfg.num_experts * expert
            active += attn + norms + router + cfg.num_experts_per_tok * expert
        else:
            total += attn + norms + dense_mlp
            active += attn + norms + dense_mlp
    embed = cfg.vocab_size * h
    head = 0 if cfg.tie_word_embeddings else cfg.vocab_size * h
    total += embed + head + h
    active += embed + head + h
    return total, active


# --------------------------------------------------------------------------- #
# HiSparse / Quest configuration  (``--hisparse-config`` in SGLang)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SparseConfig:
    """Mirror of ``sglang.srt.mem_cache.sparsity.core.sparse_coordinator.SparseConfig``.

    Only ``top_k``, ``device_buffer_size``, ``algorithm`` and
    ``quest_page_size`` influence this reference implementation;
    ``host_to_device_ratio`` / ``swap_in_block_size`` size SGLang's host pool and
    CUDA grid and are kept so a real ``--hisparse-config`` string round-trips.
    Unknown keys land in ``sparse_extra_config`` exactly like upstream; this
    package reads one of them, ``avoid_recent_overlap`` (see
    :class:`qwenquest.quest.QuestAlgorithm`).
    """

    top_k: int = 2048
    device_buffer_size: int = 4096
    host_to_device_ratio: int = 2
    swap_in_block_size: int = 960
    algorithm: str | None = None
    backend: str | None = None
    page_size: int | None = None
    min_sparse_prompt_len: int | None = None
    # Quest page granularity for the min/max key bounds.  Distinct from
    # ``page_size`` (KV-pool storage granularity).  Must divide ``top_k``.
    quest_page_size: int | None = None
    sparse_extra_config: dict[str, Any] = field(default_factory=dict, hash=False)

    @property
    def avoid_recent_overlap(self) -> bool:
        return bool(self.sparse_extra_config.get("avoid_recent_overlap", False))


def parse_hisparse_config(config: str | Mapping[str, Any] | SparseConfig | None) -> SparseConfig:
    """Parse ``--hisparse-config`` with upstream defaults and validation.

    Defaults: ``top_k=2048``, ``device_buffer_size=2*top_k``,
    ``host_to_device_ratio=2``, ``swap_in_block_size=960`` and - when
    ``algorithm == "quest"`` - ``quest_page_size=64``.
    """
    if isinstance(config, SparseConfig):
        return config
    if config is None:
        extra: dict[str, Any] = {}
    elif isinstance(config, str):
        try:
            extra = dict(json.loads(config))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse hisparse_config: {e}") from e
    else:
        extra = dict(config)

    top_k = extra.pop("top_k", 2048)
    device_buffer_size = extra.pop("device_buffer_size", 2 * top_k)
    host_to_device_ratio = extra.pop("host_to_device_ratio", 2)
    swap_in_block_size = extra.pop("swap_in_block_size", 960)

    if device_buffer_size < top_k:
        raise ValueError(
            f"device_buffer_size ({device_buffer_size}) must be no smaller than top_k ({top_k})"
        )
    if not isinstance(swap_in_block_size, int) or isinstance(swap_in_block_size, bool):
        raise ValueError(f"swap_in_block_size must be an integer, got {swap_in_block_size!r}")
    if swap_in_block_size <= 0 or swap_in_block_size > 1024:
        raise ValueError(
            f"swap_in_block_size ({swap_in_block_size}) must be in the range [1, 1024]"
        )

    algorithm = extra.pop("algorithm", None)
    backend = extra.pop("backend", None)
    min_sparse_prompt_len = extra.pop("min_sparse_prompt_len", None)
    page_size = extra.pop("page_size", None)
    quest_page_size = extra.pop("quest_page_size", None)

    if algorithm == "quest":
        if quest_page_size is None:
            quest_page_size = 64
        if quest_page_size <= 0 or top_k % quest_page_size != 0:
            raise ValueError(
                f"hisparse_config: top_k ({top_k}) must be divisible by "
                f"quest_page_size ({quest_page_size}) for algorithm='quest'."
            )

    return SparseConfig(
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        host_to_device_ratio=host_to_device_ratio,
        swap_in_block_size=swap_in_block_size,
        algorithm=algorithm,
        backend=backend,
        page_size=page_size,
        min_sparse_prompt_len=min_sparse_prompt_len,
        quest_page_size=quest_page_size,
        sparse_extra_config=extra,
    )
