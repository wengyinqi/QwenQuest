"""Attention-backend interface (the role of SGLang's ``AttentionBackend``)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from qwenquest.attention.ops import extend_attention

if TYPE_CHECKING:
    from qwenquest.batch import ForwardBatch
    from qwenquest.memory import MHATokenToKVPool, ReqToTokenPool
    from qwenquest.model.qwen3_moe import RadixAttention

__all__ = ["AttentionBackend"]


class AttentionBackend(ABC):
    """Called by every ``RadixAttention`` of the model.

    Per forward the engine calls :meth:`init_forward_metadata` once, then the
    model calls :meth:`forward` once per layer.  Prefill (EXTEND) is dense in
    every mode - Quest only changes decode - so :meth:`forward_extend` lives
    here and subclasses implement :meth:`forward_decode`.
    """

    #: SGLang backend this class stands in for.
    name: str = ""
    #: Whether a decode token's K/V goes to the shared device token pool.
    #: HiSparse writes it to the request's hot buffer instead.
    uses_token_pool_for_decode: bool = True
    #: Whether the engine may free the prefill's token-pool slots after
    #: :meth:`on_extend_finished` (HiSparse moved the KV to host + hot buffer).
    frees_prefill_kv_after_admission: bool = False

    def __init__(self, req_to_token_pool: ReqToTokenPool, token_to_kv_pool: MHATokenToKVPool):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool

    @property
    def req_to_token(self) -> torch.Tensor:
        return self.req_to_token_pool.req_to_token

    # ------------------------------------------------------------ per step

    def init_forward_metadata(self, batch: ForwardBatch) -> None:  # noqa: B027 - optional hook
        """Per-forward preparation (once, before layer 0)."""

    # ------------------------------------------------------------ per layer

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        if batch.forward_mode.is_decode():
            return self.forward_decode(q, k, v, layer, batch)
        return self.forward_extend(q, k, v, layer, batch)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        """Dense causal prefill: save the new K/V, then attend per request."""
        assert batch.out_cache_loc is not None
        assert batch.extend_prefix_lens is not None
        self.token_to_kv_pool.set_kv_buffer(layer.layer_id, batch.out_cache_loc, k, v)
        k_buf, v_buf = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        offsets = batch.extend_token_offsets()
        outputs = []
        for i in range(batch.batch_size):
            req = int(batch.req_pool_indices[i])
            seq_len = int(batch.seq_lens[i])
            locs = self.req_to_token[req, :seq_len].long()
            outputs.append(
                extend_attention(
                    q[offsets[i] : offsets[i + 1]],
                    k_buf[locs],
                    v_buf[locs],
                    prefix_len=int(batch.extend_prefix_lens[i]),
                    scaling=layer.scaling,
                )
            )
        return torch.cat(outputs, dim=0)

    @abstractmethod
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        """``q``: ``[bs, H, D]``; ``k``/``v``: ``[bs, KV, D]`` of the new tokens -> ``[bs, H*D]``."""

    # ------------------------------------------------------------ lifecycle

    def on_extend_finished(self, batch: ForwardBatch) -> None:  # noqa: B027 - optional hook
        """Called by the engine after a whole prefill forward (all layers)."""

    def on_request_finished(self, req_pool_idx: int) -> None:  # noqa: B027 - optional hook
        """Called by the engine before a request slot is reused."""
