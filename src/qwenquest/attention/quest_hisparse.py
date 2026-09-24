"""Mode 3 - Quest selection + HiSparse hierarchical KV cache.

Stands in for SGLang's ``--enable-hisparse --decode-attention-backend
flashinfer_hisparse`` with ``--hisparse-config '{"algorithm": "quest", ...}'``
(``flashinfer_hisparse_backend.py`` on the ``hisparse_quest`` branch).

Per decode step::

    init_forward_metadata:  [H2] back up the previous token to host and fold its
                                 key into Quest's running bounds [Q3]
                            [Q4] quest.prepare_step(seq_lens)
    forward_decode (x48):   write new K/V into the hot buffer's newest slot
                            -> [Q6] quest.retrieve_topk(q)
                            -> [H3] swap_in_selected_pages: positions -> device rows
                               (hits stay, misses are copied host -> device)
                            -> [Q7] attention over those rows

Prefill runs dense on the regular token pool; right after it the request is
admitted [H1] (host backup + hot buffer + Quest prefill bounds) and the
engine frees the prefill's token-pool slots.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from qwenquest.attention.base import AttentionBackend
from qwenquest.attention.ops import decode_attention
from qwenquest.config import SparseConfig, parse_hisparse_config
from qwenquest.hisparse.coordinator import HiSparseCoordinator
from qwenquest.quest import QuestAlgorithm

if TYPE_CHECKING:
    from qwenquest.batch import ForwardBatch
    from qwenquest.memory import MHATokenToKVPool, ReqToTokenPool
    from qwenquest.model.qwen3_moe import RadixAttention

__all__ = ["QuestHiSparseBackend"]


class QuestHiSparseBackend(AttentionBackend):
    name = "flashinfer_hisparse"
    uses_token_pool_for_decode = False
    frees_prefill_kv_after_admission = True

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: MHATokenToKVPool,
        sparse_config: str | Mapping[str, Any] | SparseConfig | None,
        *,
        host_pool_size: int,
        start_layer: int = 0,
        end_layer: int | None = None,
        host_device: torch.device | str = "cpu",
    ):
        super().__init__(req_to_token_pool, token_to_kv_pool)
        cfg = parse_hisparse_config(sparse_config)
        if cfg.algorithm != "quest" or cfg.quest_page_size is None:
            raise ValueError("Quest backends require hisparse_config with algorithm='quest'")
        self.sparse_config = cfg
        device = token_to_kv_pool.device
        num_layers = token_to_kv_pool.num_layers
        self.quest = QuestAlgorithm(
            top_k=cfg.top_k,
            page_size=cfg.quest_page_size,
            device=device,
            avoid_recent_overlap=cfg.avoid_recent_overlap,
        )
        self.quest.init_storage(
            start_layer=start_layer,
            end_layer=num_layers if end_layer is None else end_layer,
            max_reqs=req_to_token_pool.size,
            max_context_len=req_to_token_pool.max_context_len,
            kv_heads=token_to_kv_pool.kv_heads,
            head_dim=token_to_kv_pool.head_dim,
        )
        self.coord = HiSparseCoordinator(
            quest=self.quest,
            num_layers=num_layers,
            kv_heads=token_to_kv_pool.kv_heads,
            head_dim=token_to_kv_pool.head_dim,
            max_reqs=req_to_token_pool.size,
            max_context_len=req_to_token_pool.max_context_len,
            top_k=cfg.top_k,
            device_buffer_size=cfg.device_buffer_size,
            host_pool_size=host_pool_size,
            dtype=token_to_kv_pool.dtype,
            device=device,
            host_device=host_device,
        )

    # ---------------------------------------------------------------- per step

    def init_forward_metadata(self, batch: ForwardBatch) -> None:
        if batch.forward_mode.is_decode():
            self.coord.prepare_decode_step(batch.req_pool_indices, batch.seq_lens)
            self.quest.prepare_step(batch.seq_lens)

    def on_extend_finished(self, batch: ForwardBatch) -> None:
        for req_idx, seq_len in zip(
            batch.req_pool_indices.tolist(), batch.seq_lens.tolist(), strict=True
        ):
            prefill_indices = self.req_to_token[req_idx, :seq_len]
            self.coord.admit_request(req_idx, seq_len, self.token_to_kv_pool, prefill_indices)

    # ------------------------------------------------------------------ decode

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        reqs, seq_lens = batch.req_pool_indices, batch.seq_lens
        # 1. The new token's K/V goes to its request's newest slot.
        self.coord.write_new_token(layer.layer_id, reqs, seq_lens, k, v)
        # 2. [Q6] Quest selection - identical to Mode 2.
        positions, actual_lens = self.quest.retrieve_topk(q, layer.layer_id, reqs, seq_lens)
        # 3. [H3] Bring the selected positions on-device; get their rows.
        rows = self.coord.swap_in_selected_pages(reqs, seq_lens, positions, layer.layer_id)
        # 4. [Q7] Sparse attention over the hot buffer.
        k_dev, v_dev = self.coord.device_k[layer.layer_id], self.coord.device_v[layer.layer_id]
        out = []
        for i in range(batch.batch_size):
            r = rows[i, : int(actual_lens[i])]
            out.append(decode_attention(q[i], k_dev[r], v_dev[r], layer.scaling))
        return torch.stack(out)

    # --------------------------------------------------------------- lifecycle

    def on_request_finished(self, req_pool_idx: int) -> None:
        self.coord.request_finished(req_pool_idx)  # [H4] + [Q8]
