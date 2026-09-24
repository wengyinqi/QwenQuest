"""Qwen3-MoE (Qwen3-30B-A3B) in plain PyTorch.

Parameter names match the Hugging Face checkpoints
(``model.layers.{i}.self_attn.q_proj.weight``,
``model.layers.{i}.mlp.experts.{e}.gate_proj.weight``, ...), so the official
safetensors load without renaming (see :mod:`qwenquest.model.loader`).

The model knows nothing about Quest.  Exactly like SGLang, attention layers
hand ``q, k, v`` - already through ``q_norm``/``k_norm`` and RoPE - to
``RadixAttention``, which forwards them to whichever attention backend the
batch carries ([Q1]).  Swapping the backend (dense / Quest / Quest+HiSparse)
is the whole integration; nothing else in the model changes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qwenquest.batch import ForwardBatch
from qwenquest.config import Qwen3MoeConfig
from qwenquest.model.rope import RotaryEmbedding

__all__ = [
    "Qwen3MoeAttention",
    "Qwen3MoeDecoderLayer",
    "Qwen3MoeForCausalLM",
    "Qwen3MoeMLP",
    "Qwen3MoeModel",
    "Qwen3MoeSparseMoeBlock",
    "RMSNorm",
    "RadixAttention",
]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x32.to(dtype)


class RadixAttention(nn.Module):
    """Weight-less attention "slot" of one layer (``sglang RadixAttention``).

    It only carries the layer's attention geometry and hands the call to the
    batch's attention backend.  This is the single seam through which Quest
    enters the model.
    """

    def __init__(
        self, layer_id: int, num_heads: int, num_kv_heads: int, head_dim: int, scaling: float
    ):
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = scaling

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, batch: ForwardBatch
    ) -> torch.Tensor:
        # [Q1] q: [T, 32, 128], k/v: [T, 4, 128] for Qwen3-30B-A3B.  In decode,
        # T == batch size and the backend decides which past tokens to read.
        return batch.attn_backend.forward(q, k, v, self, batch)


class Qwen3MoeAttention(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_id: int, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        h, d = config.hidden_size, config.head_dim
        bias = config.attention_bias
        self.q_proj = nn.Linear(h, self.num_heads * d, bias=bias)
        self.k_proj = nn.Linear(h, self.num_kv_heads * d, bias=bias)
        self.v_proj = nn.Linear(h, self.num_kv_heads * d, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * d, h, bias=bias)
        # Qwen3: RMSNorm over head_dim applied to every q / k head before RoPE.
        self.q_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.rotary_emb = rotary_emb
        self.attn = RadixAttention(layer_id, self.num_heads, self.num_kv_heads, d, scaling=d**-0.5)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, batch: ForwardBatch
    ) -> torch.Tensor:
        t = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(t, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(t, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        # Quest's bounds are built from exactly this k (normed + rotated) and
        # scored with exactly this q, because that is what the KV cache holds.
        o = self.attn(q, k, v, batch)  # [T, num_heads * head_dim]
        return self.o_proj(o)


class Qwen3MoeMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoeSparseMoeBlock(nn.Module):
    """128 experts, top-8 routing, renormalized top-k probabilities."""

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            Qwen3MoeMLP(config.hidden_size, config.moe_intermediate_size)
            for _ in range(config.num_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_logits = self.gate(x)  # [T, E]
        probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        weights, experts = torch.topk(probs, self.top_k, dim=-1)  # [T, k]
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights.to(router_logits.dtype)

        out = torch.zeros_like(x)
        for expert_id in torch.unique(experts).tolist():
            token_idx, slot = torch.where(experts == expert_id)
            y = self.experts[expert_id](x[token_idx]) * weights[token_idx, slot, None]
            out.index_add_(0, token_idx, y.to(out.dtype))
        return out


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_id: int, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.self_attn = Qwen3MoeAttention(config, layer_id, rotary_emb)
        self.mlp: nn.Module
        if config.is_moe_layer(layer_id):
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, batch: ForwardBatch
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(positions, self.input_layernorm(hidden_states), batch)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class Qwen3MoeModel(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            config.max_position_embeddings,
            config.rope_scaling,
        )
        self.layers = nn.ModuleList(
            Qwen3MoeDecoderLayer(config, i, rotary_emb) for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, batch: ForwardBatch) -> torch.Tensor:
        hidden_states = self.embed_tokens(batch.input_ids)
        for layer in self.layers:
            hidden_states = layer(batch.positions, hidden_states, batch)
        return self.norm(hidden_states)


class Qwen3MoeForCausalLM(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3MoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @property
    def device(self) -> torch.device:
        return self.lm_head.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.lm_head.weight.dtype

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, return_all_logits: bool = False) -> torch.Tensor:
        """Return float32 logits.

        DECODE: ``[bs, vocab]``.  EXTEND: the last token of every request
        (``[bs, vocab]``), or every token when ``return_all_logits``.
        """
        hidden_states = self.model(batch)
        if batch.forward_mode.is_extend() and not return_all_logits:
            ends = torch.tensor(batch.extend_token_offsets()[1:], device=hidden_states.device) - 1
            hidden_states = hidden_states[ends]
        return self.lm_head(hidden_states).float()
