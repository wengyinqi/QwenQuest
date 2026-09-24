"""A minimal serving loop: admit -> prefill -> batched decode -> finish.

It plays the part of SGLang's scheduler + ``ModelRunner`` just enough to
drive the three attention modes through the same lifecycle upstream uses:

* every forward: build a :class:`ForwardBatch`, call
  ``attn_backend.init_forward_metadata`` once, run the model;
* after a prefill forward: ``attn_backend.on_extend_finished`` (HiSparse
  admission) and, for HiSparse, free the prefill's token-pool slots;
* when a request ends: ``attn_backend.on_request_finished`` before the slot
  is reused.

Decoding is greedy by default (temperature/top-p sampling is available).
Like the upstream Quest modes there is no prefix cache and no chunked
prefill: each prompt is prefilled in one forward.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from qwenquest.attention import (
    ATTENTION_MODES,
    AttentionBackend,
    DenseBackend,
    QuestHiSparseBackend,
    QuestOnlyBackend,
)
from qwenquest.batch import ForwardBatch, ForwardMode
from qwenquest.config import SparseConfig, parse_hisparse_config
from qwenquest.memory import MHATokenToKVPool, ReqToTokenPool, TokenToKVPoolAllocator
from qwenquest.model.qwen3_moe import Qwen3MoeForCausalLM
from qwenquest.quest import QuestAlgorithm

__all__ = ["Engine", "EngineConfig", "GenerationOutput"]


@dataclass
class EngineConfig:
    #: "dense" (Mode 1), "quest" (Mode 2) or "quest_hisparse" (Mode 3).
    attention_mode: str = "dense"
    #: ``--hisparse-config`` JSON / dict; ``algorithm`` defaults to "quest"
    #: for the Quest modes.  Ignored in dense mode.
    hisparse_config: str | Mapping[str, Any] | SparseConfig | None = None
    max_running_requests: int = 4
    #: Longest prompt + generation; default ``min(max_position_embeddings, 32768)``.
    max_context_len: int | None = None
    #: Device token-pool size.  Default: enough for every running request at
    #: full length (dense / quest), or for one prefill (quest_hisparse, whose
    #: decode KV lives in the host pool + hot buffers).
    max_total_tokens: int | None = None
    #: HiSparse host tier ("CPU pinned memory" upstream).
    host_device: str = "cpu"


@dataclass
class GenerationOutput:
    input_ids: list[int]
    output_ids: list[int]
    finish_reason: str
    #: Per generated token float32 logits (only with ``return_logits=True``).
    logits: list[torch.Tensor] = field(default_factory=list)


@dataclass
class _Request:
    rid: int
    input_ids: list[int]
    max_new_tokens: int
    req_pool_idx: int = -1
    seq_len: int = 0  # tokens whose K/V is cached
    output_ids: list[int] = field(default_factory=list)
    logits: list[torch.Tensor] = field(default_factory=list)
    finish_reason: str | None = None


class Engine:
    def __init__(
        self, model: Qwen3MoeForCausalLM, config: EngineConfig | None = None, **kwargs: Any
    ):
        self.config = config or EngineConfig(**kwargs)
        cfg = self.config
        if cfg.attention_mode not in ATTENTION_MODES:
            raise ValueError(f"attention_mode must be one of {ATTENTION_MODES}")
        self.model = model
        mc = model.config
        device, dtype = model.device, model.dtype
        self.device = device

        self.max_context_len = cfg.max_context_len or min(mc.max_position_embeddings, 32_768)
        if cfg.max_total_tokens is not None:
            pool_size = cfg.max_total_tokens
        elif cfg.attention_mode == "quest_hisparse":
            pool_size = self.max_context_len
        else:
            pool_size = cfg.max_running_requests * self.max_context_len

        self.req_to_token_pool = ReqToTokenPool(
            cfg.max_running_requests, self.max_context_len, device
        )
        self.token_to_kv_pool = MHATokenToKVPool(
            pool_size, mc.num_hidden_layers, mc.num_key_value_heads, mc.head_dim, dtype, device
        )
        self.allocator = TokenToKVPoolAllocator(pool_size, device)
        self.attn_backend = self._make_backend()

    # ------------------------------------------------------------------ setup

    def _make_backend(self) -> AttentionBackend:
        cfg = self.config
        if cfg.attention_mode == "dense":
            return DenseBackend(self.req_to_token_pool, self.token_to_kv_pool)
        raw = cfg.hisparse_config
        if isinstance(raw, SparseConfig):
            sparse = raw
        else:
            # The Quest modes imply algorithm="quest" (explicit in SGLang).
            d = dict(json.loads(raw)) if isinstance(raw, str) else dict(raw or {})
            d.setdefault("algorithm", "quest")
            sparse = parse_hisparse_config(d)
        if cfg.attention_mode == "quest":
            return QuestOnlyBackend(self.req_to_token_pool, self.token_to_kv_pool, sparse)
        return QuestHiSparseBackend(
            self.req_to_token_pool,
            self.token_to_kv_pool,
            sparse,
            host_pool_size=cfg.max_running_requests * self.max_context_len,
            host_device=cfg.host_device,
        )

    @property
    def quest(self) -> QuestAlgorithm | None:
        return getattr(self.attn_backend, "quest", None)

    # --------------------------------------------------------------- generate

    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        max_new_tokens: int = 32,
        stop_token_ids: Sequence[int] | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        return_logits: bool = False,
    ) -> list[GenerationOutput]:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        stop = set(self.model.config.eos_token_ids if stop_token_ids is None else stop_token_ids)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        requests = [_Request(i, list(p), max_new_tokens) for i, p in enumerate(prompts)]
        waiting: deque[_Request] = deque(requests)
        running: list[_Request] = []

        def emit(req: _Request, logits: torch.Tensor) -> None:
            if return_logits:
                req.logits.append(logits.cpu())
            token = self._sample(logits, temperature, top_p, gen)
            req.output_ids.append(token)
            if token in stop:
                req.finish_reason = "stop"
            elif len(req.output_ids) >= req.max_new_tokens or req.seq_len >= self.max_context_len:
                # (the new token would be written at position seq_len)
                req.finish_reason = "length"

        while waiting or running:
            while waiting and len(running) < self.config.max_running_requests:
                req = waiting.popleft()
                emit(req, self._prefill(req))
                if req.finish_reason is None:
                    running.append(req)
                else:
                    self._finish(req)
            if running:
                logits = self._decode(running)
                for req, row in zip(running, logits, strict=True):
                    emit(req, row)
                for req in [r for r in running if r.finish_reason is not None]:
                    running.remove(req)
                    self._finish(req)

        return [
            GenerationOutput(r.input_ids, r.output_ids, r.finish_reason or "length", r.logits)
            for r in requests
        ]

    # ---------------------------------------------------------------- forward

    def _prefill(self, req: _Request) -> torch.Tensor:
        n = len(req.input_ids)
        if n == 0 or n >= self.max_context_len:
            raise ValueError(
                f"prompt length {n} must be in [1, max_context_len={self.max_context_len})"
            )
        req.req_pool_idx = self.req_to_token_pool.alloc()
        locs = self.allocator.alloc(n)
        self.req_to_token_pool.req_to_token[req.req_pool_idx, :n] = locs.to(torch.int32)
        dev = self.device
        batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            input_ids=torch.tensor(req.input_ids, dtype=torch.int64, device=dev),
            positions=torch.arange(n, dtype=torch.int64, device=dev),
            req_pool_indices=torch.tensor([req.req_pool_idx], dtype=torch.int64, device=dev),
            seq_lens=torch.tensor([n], dtype=torch.int64, device=dev),
            out_cache_loc=locs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            attn_backend=self.attn_backend,
            extend_prefix_lens=torch.zeros(1, dtype=torch.int64, device=dev),
            extend_seq_lens=torch.tensor([n], dtype=torch.int64, device=dev),
        )
        self.attn_backend.init_forward_metadata(batch)
        logits = self.model(batch)[0]
        self.attn_backend.on_extend_finished(batch)
        if self.attn_backend.frees_prefill_kv_after_admission:
            self.allocator.free(locs)
            self.req_to_token_pool.req_to_token[req.req_pool_idx, :n] = 0
        req.seq_len = n
        return logits

    def _decode(self, reqs: list[_Request]) -> torch.Tensor:
        dev = self.device
        req_idx = torch.tensor([r.req_pool_idx for r in reqs], dtype=torch.int64, device=dev)
        positions = torch.tensor([r.seq_len for r in reqs], dtype=torch.int64, device=dev)
        out_cache_loc = None
        if self.attn_backend.uses_token_pool_for_decode:
            out_cache_loc = self.allocator.alloc(len(reqs))
            self.req_to_token_pool.req_to_token[req_idx, positions] = out_cache_loc.to(torch.int32)
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            input_ids=torch.tensor([r.output_ids[-1] for r in reqs], dtype=torch.int64, device=dev),
            positions=positions,
            req_pool_indices=req_idx,
            seq_lens=positions + 1,
            out_cache_loc=out_cache_loc,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            attn_backend=self.attn_backend,
        )
        self.attn_backend.init_forward_metadata(batch)
        logits = self.model(batch)
        for r in reqs:
            r.seq_len += 1
        return logits

    def _finish(self, req: _Request) -> None:
        self.attn_backend.on_request_finished(req.req_pool_idx)
        self.allocator.free(self.req_to_token_pool.req_to_token[req.req_pool_idx, : req.seq_len])
        self.req_to_token_pool.free(req.req_pool_idx)

    # --------------------------------------------------------------- sampling

    @staticmethod
    def _sample(
        logits: torch.Tensor, temperature: float, top_p: float, gen: torch.Generator
    ) -> int:
        if temperature <= 0:
            return int(torch.argmax(logits))
        probs = torch.softmax(logits.float().cpu() / temperature, dim=-1)
        if top_p < 1.0:
            sorted_p, order = torch.sort(probs, descending=True)
            keep = torch.cumsum(sorted_p, dim=-1) - sorted_p < top_p
            probs = torch.zeros_like(probs).scatter(0, order[keep], sorted_p[keep])
        return int(torch.multinomial(probs / probs.sum(), 1, generator=gen))
