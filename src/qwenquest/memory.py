"""KV-cache memory pools, mirroring SGLang's two-level addressing.

SGLang never stores KV "per request".  It keeps

* ``ReqToTokenPool.req_to_token[req_pool_idx, position]`` - for every running
  request slot, the *physical* token index that holds the KV of each
  *logical* position, and
* ``MHATokenToKVPool.k_buffer[layer][token_index]`` (and ``v_buffer``) - one
  flat ``[num_tokens, kv_heads, head_dim]`` tensor per layer.

Quest's upstream API is written against these two tables (for example
``update_prefill_representations(layer, req_pool_idx, k_buffer, prefill_indices)``
takes physical ``prefill_indices``), and ``retrieve_topk`` returns *logical*
positions that the attention backend must translate back through
``req_to_token``.  Keeping the same structure here makes the reference code
map one-to-one onto the upstream files; see docs/upstream_mapping.md.

Slot 0 of the token pool is reserved as a padding / dummy slot, as in SGLang.
"""

from __future__ import annotations

import torch

__all__ = ["MHATokenToKVPool", "ReqToTokenPool", "TokenToKVPoolAllocator"]


class ReqToTokenPool:
    """``req_to_token[req_pool_idx, position] -> physical token index``."""

    def __init__(self, size: int, max_context_len: int, device: torch.device | str):
        self.size = size
        self.max_context_len = max_context_len
        self.device = torch.device(device)
        self.req_to_token = torch.zeros(
            (size, max_context_len), dtype=torch.int32, device=self.device
        )
        self._free_slots = list(range(size))

    def available_size(self) -> int:
        return len(self._free_slots)

    def alloc(self) -> int:
        if not self._free_slots:
            raise RuntimeError("ReqToTokenPool is full; lower the number of concurrent requests")
        return self._free_slots.pop(0)

    def free(self, req_pool_idx: int) -> None:
        self.req_to_token[req_pool_idx].zero_()
        self._free_slots.append(req_pool_idx)


class TokenToKVPoolAllocator:
    """Free list over physical token indices ``[1, size]`` (0 = padding)."""

    def __init__(self, size: int, device: torch.device | str = "cpu"):
        self.size = size
        self.device = torch.device(device)
        self._free = torch.arange(1, size + 1, dtype=torch.int64, device=self.device)

    def available_size(self) -> int:
        return int(self._free.numel())

    def alloc(self, n: int) -> torch.Tensor:
        if n > self._free.numel():
            raise RuntimeError(
                f"KV pool out of memory: need {n} token slots, {self._free.numel()} free"
            )
        out, self._free = self._free[:n], self._free[n:]
        return out

    def free(self, indices: torch.Tensor) -> None:
        indices = indices.to(device=self.device, dtype=torch.int64).flatten()
        indices = indices[indices > 0]
        if indices.numel():
            self._free = torch.cat([self._free, indices])


class MHATokenToKVPool:
    """Per-layer K / V buffers of shape ``[size + 1, kv_heads, head_dim]``."""

    def __init__(
        self,
        size: int,
        num_layers: int,
        kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        self.size = size
        self.num_layers = num_layers
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        shape = (size + 1, kv_heads, head_dim)
        self.k_buffer = [
            torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(num_layers)
        ]
        self.v_buffer = [
            torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(num_layers)
        ]

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_buffer[layer_id], self.v_buffer[layer_id]

    def set_kv_buffer(
        self, layer_id: int, loc: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """Write ``k``/``v`` of shape ``[n, kv_heads, head_dim]`` at token indices ``loc``."""
        loc = loc.to(torch.int64)
        self.k_buffer[layer_id][loc] = k.to(self.dtype)
        self.v_buffer[layer_id][loc] = v.to(self.dtype)
