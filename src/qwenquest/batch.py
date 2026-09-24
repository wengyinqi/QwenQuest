"""The per-forward batch description passed from the engine through the model
into the attention backend (a trimmed-down ``sglang ForwardBatch``).

Tokens of all requests are flattened into one ``[num_tokens, ...]`` axis:

* EXTEND (prefill): request ``i`` contributes ``extend_seq_lens[i]`` new tokens
  at positions ``[extend_prefix_lens[i], seq_lens[i])``.
* DECODE: every request contributes exactly one token, at position
  ``seq_lens[i] - 1`` (``seq_lens`` already counts the token being decoded).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from qwenquest.attention.base import AttentionBackend
    from qwenquest.memory import MHATokenToKVPool, ReqToTokenPool

__all__ = ["ForwardBatch", "ForwardMode"]


class ForwardMode(enum.Enum):
    EXTEND = "extend"
    DECODE = "decode"

    def is_extend(self) -> bool:
        return self is ForwardMode.EXTEND

    def is_decode(self) -> bool:
        return self is ForwardMode.DECODE


@dataclass
class ForwardBatch:
    forward_mode: ForwardMode
    input_ids: torch.Tensor  # [num_tokens] int64
    positions: torch.Tensor  # [num_tokens] int64
    req_pool_indices: torch.Tensor  # [bs] int64
    seq_lens: torch.Tensor  # [bs] int64 - total length incl. the new tokens
    # Physical token-pool slots receiving the new tokens' K/V ([num_tokens]).
    # None when the backend keeps new decode tokens elsewhere (HiSparse).
    out_cache_loc: torch.Tensor | None
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: MHATokenToKVPool
    attn_backend: AttentionBackend
    extend_prefix_lens: torch.Tensor | None = None  # [bs] (EXTEND only)
    extend_seq_lens: torch.Tensor | None = None  # [bs] (EXTEND only)

    @property
    def batch_size(self) -> int:
        return int(self.req_pool_indices.shape[0])

    def extend_token_offsets(self) -> list[int]:
        """Start offset of every request's tokens in the flattened axis (+ total)."""
        assert self.extend_seq_lens is not None
        offsets = [0]
        for n in self.extend_seq_lens.tolist():
            offsets.append(offsets[-1] + int(n))
        return offsets
