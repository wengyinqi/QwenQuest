"""Mode 2 - Quest sparse decode, full KV cache on the device.

Stands in for SGLang's ``--attention-backend flashinfer_quest``
(``flashinfer_quest_backend.py`` on the ``hisparse_quest`` branch): the
"is sparsity itself the win, or the offloading?" baseline.  Selection is
identical to Mode 3; only where the selected K/V is read from differs.

Per decode step::

    init_forward_metadata:  [Q3] fold the previous token's K into Quest's running
                                 bounds (skipped on the first decode step after
                                 prefill, whose last token prefill already covered)
                            [Q4] quest.prepare_step(seq_lens)
    forward_decode (x48):   save new K/V -> [Q6] quest.retrieve_topk(q)
                            -> [Q7] logical positions -> req_to_token -> physical
                            -> attention over exactly those rows
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from qwenquest.attention.base import AttentionBackend
from qwenquest.attention.ops import decode_attention
from qwenquest.config import SparseConfig, parse_hisparse_config
from qwenquest.quest import QuestAlgorithm

if TYPE_CHECKING:
    from qwenquest.batch import ForwardBatch
    from qwenquest.memory import MHATokenToKVPool, ReqToTokenPool
    from qwenquest.model.qwen3_moe import RadixAttention

__all__ = ["QuestOnlyBackend"]


class QuestOnlyBackend(AttentionBackend):
    name = "flashinfer_quest"

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: MHATokenToKVPool,
        sparse_config: str | Mapping[str, Any] | SparseConfig | None,
        *,
        start_layer: int = 0,
        end_layer: int | None = None,
    ):
        super().__init__(req_to_token_pool, token_to_kv_pool)
        cfg = parse_hisparse_config(sparse_config)
        if cfg.algorithm != "quest" or cfg.quest_page_size is None:
            raise ValueError("Quest backends require hisparse_config with algorithm='quest'")
        self.sparse_config = cfg
        device = token_to_kv_pool.device
        # Quest at every layer (HiSparse passes the runner's full layer range).
        self.quest = QuestAlgorithm(
            top_k=cfg.top_k,
            page_size=cfg.quest_page_size,
            device=device,
            avoid_recent_overlap=cfg.avoid_recent_overlap,
        )
        self.quest.init_storage(
            start_layer=start_layer,
            end_layer=token_to_kv_pool.num_layers if end_layer is None else end_layer,
            max_reqs=req_to_token_pool.size,
            max_context_len=req_to_token_pool.max_context_len,
            kv_heads=token_to_kv_pool.kv_heads,
            head_dim=token_to_kv_pool.head_dim,
        )
        # Set when prefill bounds are built, cleared by the first decode step.
        self._skip_first_decode_update = torch.zeros(
            req_to_token_pool.size, dtype=torch.bool, device=device
        )

    # ---------------------------------------------------------------- per step

    def init_forward_metadata(self, batch: ForwardBatch) -> None:
        if batch.forward_mode.is_decode():
            self._do_quest_decode_step_update(batch.req_pool_indices, batch.seq_lens)
            self.quest.prepare_step(batch.seq_lens)

    def _do_quest_decode_step_update(
        self, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor
    ) -> None:
        """[Q3] The previous step's token sits at position ``seq_len - 2``."""
        skip = self._skip_first_decode_update[req_pool_indices]
        self._skip_first_decode_update[req_pool_indices] = False
        active_idx = req_pool_indices[~skip]
        prev_token_pos = (seq_lens[~skip] - 2).clamp(min=0).long()
        device_locs = self.req_to_token[active_idx, prev_token_pos]
        for layer_id in range(self.quest.start_layer, self.quest.end_layer):
            self.quest.update_decode_representations(
                layer_id, active_idx, self.token_to_kv_pool.get_key_buffer(layer_id), device_locs
            )
        self.quest.maybe_finalize_decode_representations(active_idx)

    # ----------------------------------------------------------------- prefill

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        out = super().forward_extend(q, k, v, layer, batch)
        # Bounds need every Quest layer's K, so build them after the last one.
        if layer.layer_id == self.quest.end_layer - 1:
            self._update_quest_for_extend(batch)
        return out

    def _update_quest_for_extend(self, batch: ForwardBatch) -> None:
        """[Q2] Per request: bounds of positions ``0 .. seq_len-1`` at every layer."""
        for req_idx, seq_len in zip(
            batch.req_pool_indices.tolist(), batch.seq_lens.tolist(), strict=True
        ):
            prefill_indices = self.req_to_token[req_idx, :seq_len]
            for layer_id in range(self.quest.start_layer, self.quest.end_layer):
                self.quest.update_prefill_representations(
                    layer_id,
                    req_idx,
                    self.token_to_kv_pool.get_key_buffer(layer_id),
                    prefill_indices,
                )
            self._skip_first_decode_update[req_idx] = True

    # ------------------------------------------------------------------ decode

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        # 1. Save the new token's K/V into the (full, on-device) token pool.
        assert batch.out_cache_loc is not None
        self.token_to_kv_pool.set_kv_buffer(layer.layer_id, batch.out_cache_loc, k, v)

        # 2. [Q6] Quest: which logical positions this layer reads.
        positions, actual_lens = self.quest.retrieve_topk(
            q, layer.layer_id, batch.req_pool_indices, batch.seq_lens
        )

        # 3. [Q7] logical -> physical (``quest_only_gather_scatter`` upstream)
        #    and attention over the selected rows (duplicates are kept, exactly
        #    like FlashInfer reading the packed kv_indices).
        k_buf, v_buf = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        out = []
        for i in range(batch.batch_size):
            n = int(actual_lens[i])
            req = int(batch.req_pool_indices[i])
            locs = self.req_to_token[req, positions[i, :n].long()].long()
            out.append(decode_attention(q[i], k_buf[locs], v_buf[locs], layer.scaling))
        return torch.stack(out)

    # --------------------------------------------------------------- lifecycle

    def on_request_finished(self, req_pool_idx: int) -> None:
        # [Q8] Upstream Mode 2 has no finish hook and relies on the next
        # prefill overwriting the slot; resetting here is strictly safer.
        self.quest.invalidate_request(req_pool_idx)
        self._skip_first_decode_update[req_pool_idx] = False
