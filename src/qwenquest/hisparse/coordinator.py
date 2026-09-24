"""HiSparse hierarchical KV cache (Quest mode), reference implementation.

Upstream (SGLang, ``hisparse_quest`` @ ``e568f8a3``):

* ``python/sglang/srt/managers/hisparse_coordinator.py`` - admission,
  eager backup, device-buffer bookkeeping, request release.
* ``python/sglang/jit_kernel/csrc/hisparse.cuh`` - the fused swap-in kernel
  (hit detection -> LRU reordering -> host-to-device copy of misses).

Two memory tiers
----------------
* **host** (CPU pinned memory upstream): every request's *full* KV history,
  ``host_k/v[layer][host_index]``, addressed through
  ``req_to_host_pool[req, position]``.
* **device**: per request a small *hot buffer* of ``device_buffer_size`` LRU
  slots plus one reserved slot (index ``device_buffer_size``) for the token
  being decoded.  ``req_to_device_buffer[req, slot]`` gives the flat index into
  ``device_k/v[layer]``; ``buffer_tokens[layer, req, slot]`` records which
  position an LRU slot currently caches.

GPU memory per request is therefore bounded by ``device_buffer_size + 1``
tokens instead of the sequence length.  Quest only chooses *which* positions
attention reads; HiSparse makes sure they are on the device.

Deliberate difference from the CUDA kernel
------------------------------------------
Quest's layout can list a position twice (a selected page overlapping the
recent window).  The kernel's hash table keeps one index per token, so the
second copy is treated as a miss and loaded into another slot; on a later step
both slots count as hits and the derived miss count (``NUM_TOP_K - hits``)
falls short, leaving the last miss un-copied.  Here duplicates are resolved
first, so the device rows handed to attention are always exactly the KV of
the requested positions.  See docs/quest_walkthrough.md section 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qwenquest.memory import MHATokenToKVPool, TokenToKVPoolAllocator
from qwenquest.quest import QuestAlgorithm

__all__ = ["HiSparseCoordinator", "SwapInStats"]


@dataclass
class SwapInStats:
    """Hit/miss accounting of the LRU hot buffer (long requests only)."""

    hits: int = 0
    misses: int = 0
    fast_path_calls: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else float("nan")


def _unique_first_occurrence(x: torch.Tensor) -> torch.Tensor:
    """Unique values of a 1-D tensor, ordered by first occurrence."""
    if x.numel() == 0:
        return x
    uniq, inverse = torch.unique(x, return_inverse=True)
    first = torch.full((uniq.numel(),), x.numel(), dtype=torch.int64, device=x.device)
    first = first.scatter_reduce(
        0, inverse, torch.arange(x.numel(), device=x.device), reduce="amin"
    )
    return uniq[torch.argsort(first)]


class HiSparseCoordinator:
    def __init__(
        self,
        *,
        quest: QuestAlgorithm,
        num_layers: int,
        kv_heads: int,
        head_dim: int,
        max_reqs: int,
        max_context_len: int,
        top_k: int,
        device_buffer_size: int,
        host_pool_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
        host_device: torch.device | str = "cpu",
    ):
        if device_buffer_size < top_k:
            raise ValueError("device_buffer_size must be >= top_k")
        if quest.top_k != top_k:
            raise ValueError(f"QuestAlgorithm.top_k ({quest.top_k}) != top_k ({top_k})")
        self.quest = quest
        self.num_layers = num_layers
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = torch.device(device)
        self.host_device = torch.device(host_device)
        buf = device_buffer_size

        # ---- host tier: the full history of every request.
        self.host_allocator = TokenToKVPoolAllocator(host_pool_size, device=self.host_device)
        host_shape = (host_pool_size + 1, kv_heads, head_dim)
        self.host_k = [
            torch.zeros(host_shape, dtype=dtype, device=self.host_device) for _ in range(num_layers)
        ]
        self.host_v = [
            torch.zeros(host_shape, dtype=dtype, device=self.host_device) for _ in range(num_layers)
        ]
        self.req_to_host_pool = torch.full(
            (max_reqs, max_context_len), -1, dtype=torch.int64, device=self.device
        )

        # ---- device tier: per request `buf` LRU slots + 1 newest-token slot.
        dev_shape = (max_reqs * (buf + 1), kv_heads, head_dim)
        self.device_k = [
            torch.zeros(dev_shape, dtype=dtype, device=self.device) for _ in range(num_layers)
        ]
        self.device_v = [
            torch.zeros(dev_shape, dtype=dtype, device=self.device) for _ in range(num_layers)
        ]
        self.req_to_device_buffer = torch.arange(
            max_reqs * (buf + 1), dtype=torch.int64, device=self.device
        ).view(max_reqs, buf + 1)
        self.buffer_tokens = torch.full(
            (num_layers, max_reqs, buf), -1, dtype=torch.int64, device=self.device
        )
        self._lru_init = torch.arange(buf, dtype=torch.int64, device=self.device)
        self.lru_slots = self._lru_init.repeat(num_layers, max_reqs, 1)  # LRU -> MRU

        self._skip_first_backup = [False] * max_reqs
        self.stats = SwapInStats()

    # -------------------------------------------------- [H1] admission

    def admit_request(
        self,
        req_pool_idx: int,
        prefill_len: int,
        token_to_kv_pool: MHATokenToKVPool,
        prefill_indices: torch.Tensor,
    ) -> None:
        """[H1] Move a finished prefill into the two tiers.

        ``admit_request_into_staging`` + ``alloc_device_buffer`` upstream:
        back up all prefill KV to host, keep positions ``[0, device_buffer_size)``
        resident in the hot buffer, and build Quest's prefill bounds [Q2] from
        the keys while they are still on the device.  Afterwards the engine
        frees the prefill's token-pool slots.
        """
        prefill_indices = prefill_indices.long()
        host_idx = self.host_allocator.alloc(prefill_len)
        self.req_to_host_pool[req_pool_idx, :prefill_len] = host_idx.to(self.device)
        n_dev = min(prefill_len, self.device_buffer_size)
        dev_locs = self.req_to_device_buffer[req_pool_idx, :n_dev]
        for layer_id in range(self.num_layers):
            k_src, v_src = token_to_kv_pool.get_kv_buffer(layer_id)
            k_rows, v_rows = k_src[prefill_indices], v_src[prefill_indices]
            self.host_k[layer_id][host_idx] = k_rows.to(self.host_device)
            self.host_v[layer_id][host_idx] = v_rows.to(self.host_device)
            self.device_k[layer_id][dev_locs] = k_rows[:n_dev]
            self.device_v[layer_id][dev_locs] = v_rows[:n_dev]
        # Slot s caches position s (valid by the time seq_len > device_buffer_size).
        self.buffer_tokens[:, req_pool_idx, :] = self._lru_init
        self.lru_slots[:, req_pool_idx, :] = self._lru_init

        for layer_id in range(self.quest.start_layer, self.quest.end_layer):
            self.quest.update_prefill_representations(
                layer_id, req_pool_idx, token_to_kv_pool.get_key_buffer(layer_id), prefill_indices
            )
        # Prefill already covered the last prompt token (host copy + bounds).
        self._skip_first_backup[req_pool_idx] = True

    # -------------------------------------------------- [H2] per decode step

    def newest_slot(self, seq_lens: torch.Tensor) -> torch.Tensor:
        """Slot of the token being decoded: its own slot while the request fits
        the buffer, the reserved slot ``device_buffer_size`` afterwards."""
        return (seq_lens.long() - 1).clamp(max=self.device_buffer_size)

    def prepare_decode_step(self, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor) -> None:
        """[H2] ``map_last_loc_to_buffer`` / ``_eager_backup_previous_token``.

        The token decoded in the *previous* step is still in its device slot:
        fold its key into Quest's running bounds [Q3], then back it up to host
        so later steps can swap it back in after it leaves the reserved slot.
        """
        backup_reqs: list[int] = []
        backup_pos: list[int] = []
        for req, seq_len in zip(req_pool_indices.tolist(), seq_lens.tolist(), strict=True):
            if self._skip_first_backup[req]:
                self._skip_first_backup[req] = False
                continue
            backup_reqs.append(req)
            backup_pos.append(seq_len - 2)
        if not backup_reqs:
            return

        reqs = torch.tensor(backup_reqs, dtype=torch.int64, device=self.device)
        prev_pos = torch.tensor(backup_pos, dtype=torch.int64, device=self.device)
        dev_locs = self.req_to_device_buffer[reqs, prev_pos.clamp(max=self.device_buffer_size)]

        for layer_id in range(self.quest.start_layer, self.quest.end_layer):
            self.quest.update_decode_representations(
                layer_id, reqs, self.device_k[layer_id], dev_locs
            )
        self.quest.maybe_finalize_decode_representations(reqs)

        host_idx = self.host_allocator.alloc(len(backup_reqs))
        self.req_to_host_pool[reqs, prev_pos] = host_idx.to(self.device)
        for layer_id in range(self.num_layers):
            self.host_k[layer_id][host_idx] = self.device_k[layer_id][dev_locs].to(self.host_device)
            self.host_v[layer_id][host_idx] = self.device_v[layer_id][dev_locs].to(self.host_device)

    def write_new_token(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """``set_kv_buffer`` redirected into the hot buffer (``_resolve_write_loc``)."""
        locs = self.req_to_device_buffer[req_pool_indices.long(), self.newest_slot(seq_lens)]
        self.device_k[layer_id][locs] = k.to(self.device_k[layer_id].dtype)
        self.device_v[layer_id][locs] = v.to(self.device_v[layer_id].dtype)

    # -------------------------------------------------- [H3] swap-in

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """[H3] Make the selected positions device-resident; return their rows.

        Returns ``[bs, top_k]`` int64 indices into ``device_k/v[layer_id]``
        (``-1`` past each request's ``min(seq_len, top_k)`` entries).
        """
        bs = int(req_pool_indices.shape[0])
        out = torch.full((bs, self.top_k), -1, dtype=torch.int64, device=self.device)
        for i in range(bs):
            req = int(req_pool_indices[i])
            seq_len = int(seq_lens[i])
            n = min(seq_len, self.top_k)
            positions = top_k_result[i, :n].to(device=self.device, dtype=torch.int64)
            if seq_len <= self.device_buffer_size:
                # Fast path: the whole sequence is resident, slot == position.
                self.stats.fast_path_calls += 1
                out[i, :n] = self.req_to_device_buffer[req, positions]
            else:
                out[i, :n] = self._swap_in_long(layer_id, req, seq_len, positions)
        return out

    def _swap_in_long(
        self, layer_id: int, req: int, seq_len: int, positions: torch.Tensor
    ) -> torch.Tensor:
        buf = self.device_buffer_size
        locs = torch.empty_like(positions)
        is_newest = positions == seq_len - 1
        locs[is_newest] = self.req_to_device_buffer[req, buf]  # reserved slot

        wanted = positions[~is_newest]
        uniq = _unique_first_occurrence(wanted)  # kernel order: by top-k index
        tokens = self.buffer_tokens[layer_id, req]  # [buf] position cached per slot
        lru = self.lru_slots[layer_id, req]  # [buf] slots, least -> most recently used

        # 1. hit detection
        hit_in_lru = torch.isin(tokens[lru], uniq)
        hit_slots = lru[hit_in_lru]  # keep LRU scan order
        evictable = lru[~hit_in_lru]  # least recently used first
        misses = uniq[~torch.isin(uniq, tokens)]
        num_miss = int(misses.numel())
        if num_miss > evictable.numel():
            raise RuntimeError("device_buffer_size too small for this top-k")

        # 2. miss handling: evict the LRU slots and copy host -> device
        victims = evictable[:num_miss]
        if num_miss:
            host_idx = self.req_to_host_pool[req, misses]
            if bool((host_idx < 0).any()):
                raise AssertionError(f"req {req}: positions without host backup were selected")
            dst = self.req_to_device_buffer[req, victims]
            host_idx = host_idx.to(self.host_device)
            self.device_k[layer_id][dst] = self.host_k[layer_id][host_idx].to(self.device)
            self.device_v[layer_id][dst] = self.host_v[layer_id][host_idx].to(self.device)
            tokens[victims] = misses

        # 3. LRU reordering, as the kernel writes it back:
        #    [stale evictables (LRU end) | just-loaded misses | hits (MRU end)]
        self.lru_slots[layer_id, req] = torch.cat([evictable[num_miss:], victims, hit_slots])

        # 4. position -> slot for every requested entry (duplicates included)
        sorted_tokens, order = torch.sort(tokens)
        slot = order[torch.searchsorted(sorted_tokens, wanted)]
        locs[~is_newest] = self.req_to_device_buffer[req, slot]

        self.stats.hits += int(uniq.numel()) - num_miss
        self.stats.misses += num_miss
        return locs

    # -------------------------------------------------- [H4] release

    def request_finished(self, req_pool_idx: int) -> None:
        """[H4] Free the host rows and reset per-request state (+ [Q8])."""
        host_idx = self.req_to_host_pool[req_pool_idx]
        self.host_allocator.free(host_idx[host_idx >= 0].to(self.host_device))
        self.req_to_host_pool[req_pool_idx] = -1
        self.buffer_tokens[:, req_pool_idx, :] = -1
        self.lru_slots[:, req_pool_idx, :] = self._lru_init
        self._skip_first_backup[req_pool_idx] = False
        self.quest.invalidate_request(req_pool_idx)

    # -------------------------------------------------- introspection

    def device_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (*self.device_k, *self.device_v))

    def host_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (*self.host_k, *self.host_v))
