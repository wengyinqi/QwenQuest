"""Mode 1 - dense decode (the accuracy / speed reference).

Stands in for SGLang's ``--attention-backend flashinfer``: every decode step
reads the request's whole KV history from the device token pool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qwenquest.attention.base import AttentionBackend
from qwenquest.attention.ops import decode_attention

if TYPE_CHECKING:
    from qwenquest.batch import ForwardBatch
    from qwenquest.model.qwen3_moe import RadixAttention

__all__ = ["DenseBackend"]


class DenseBackend(AttentionBackend):
    name = "flashinfer"

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        assert batch.out_cache_loc is not None
        self.token_to_kv_pool.set_kv_buffer(layer.layer_id, batch.out_cache_loc, k, v)
        k_buf, v_buf = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        out = []
        for i in range(batch.batch_size):
            req = int(batch.req_pool_indices[i])
            locs = self.req_to_token[req, : int(batch.seq_lens[i])].long()
            out.append(decode_attention(q[i], k_buf[locs], v_buf[locs], layer.scaling))
        return torch.stack(out)
