"""Quest page selection, exactly as HiSparse runs it for Qwen3-30B-A3B.

Upstream reference (SGLang, branch ``hisparse_quest`` @ ``e568f8a3``):
``python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py``.
This file keeps the upstream class/method names and semantics, but uses
plain PyTorch everywhere the upstream uses Triton (``quest_score_kernel``,
``quest_prefill_bounds_kernel``, ``quest_decode_bounds_kernel``) and drops
the CUDA-graph plumbing, so every step can be read (and unit tested) on CPU.

What Quest keeps per layer / request / page / KV head / channel
---------------------------------------------------------------
``page_k_bounds[layer, req, page, kv_head, d, 0|1] = min|max`` of the keys
(post-``k_norm``, post-RoPE, exactly what sits in the KV cache) of the
``page_size`` tokens of that page, stored in bf16.  Pages are *logical*:
page ``p`` of a request covers positions ``[p*P, (p+1)*P)``.  Unused pages
hold the sentinels ``min=+bf16max / max=-bf16max`` so that their score
collapses to (a very large) negative value without a separate validity mask.

Only *complete* pages get bounds.  The page that is still being filled by
decode accumulates into ``running_k_min/max`` and is copied into
``page_k_bounds`` ("finalized") once its ``P``-th token has been seen.

How one decode step selects tokens (per layer, per request)
-----------------------------------------------------------
For Qwen3-30B-A3B with the HiSparse defaults ``top_k=2048``, ``P=64``:

1. GQA reduction  - average the 8 query heads of each KV group:
   ``q_bar[h, d] = mean_g q[h*8 + g, d]`` -> ``[4 kv_heads, 128]``.
2. Criticality    - for each complete page ``p``:
   ``score[p] = sum_{h,d} max(q_bar[h,d] * kmin[p,h,d], q_bar[h,d] * kmax[p,h,d])``
   (implemented as ``where(q >= 0, q*kmax, q*kmin)``).  This upper-bounds
   ``sum_h q_bar_h . k_{t,h}`` for every token ``t`` of the page.  Scores are
   summed over the 4 KV heads, so **one page set is shared by all 32 heads**.
3. Top-k          - keep the ``top_k / P - 1 = 31`` best complete pages and
   append a *recent window* of the last ``P = 64`` positions
   (``[seq_len-64, seq_len)``), giving exactly ``top_k = 2048`` positions.
4. Short requests - if ``seq_len <= top_k`` the layout is simply
   ``[0, seq_len)``, i.e. dense attention.

The recent window is not page aligned.  When ``seq_len % P != 0`` it overlaps
the most recent complete page; if Quest also picks that page, the overlapping
positions appear twice and (upstream as here) are weighted twice by the
softmax.  ``avoid_recent_overlap=True`` removes such pages from the candidate
set, which keeps the budget at ``top_k`` *distinct* tokens.  The default
reproduces upstream.  See docs/quest_walkthrough.md section 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["QuestAlgorithm", "QuestSelection", "QuestTrace", "StepLayout"]

# Sentinels for running/page min-max buffers (they are bf16, so bf16 finfo).
_BF16_POS_INF = torch.finfo(torch.bfloat16).max
_BF16_NEG_INF = torch.finfo(torch.bfloat16).min


@dataclass
class StepLayout:
    """Layer-independent part of the selection, computed once per decode step.

    Upstream caches the same tensors in ``_step_*_buf`` inside
    ``QuestAlgorithm.prepare_step`` so the 48 per-layer ``retrieve_topk``
    calls do not recompute them.
    """

    seq_lens: torch.Tensor  # [bs] int64
    recent_positions: torch.Tensor  # [bs, P]      int64  last P positions
    short_layout: torch.Tensor  # [bs, top_k]  int64  0..seq_len-1 (clamped)
    is_short: torch.Tensor  # [bs, 1]      bool   seq_len <= top_k
    actual_lens: torch.Tensor  # [bs]         int32  min(seq_len, top_k)
    candidate_limit: torch.Tensor  # [bs, 1] int64  pages >= this are never candidates


@dataclass
class QuestSelection:
    """One ``retrieve_topk`` decision, recorded by :class:`QuestTrace`."""

    layer_id: int
    req_pool_idx: int
    seq_len: int
    dense: bool
    selected_pages: list[int]
    recent_window: tuple[int, int]
    page_scores: torch.Tensor | None  # [num_full_pages] float32, CPU
    positions: torch.Tensor  # [actual_len] int64, CPU (in attention order)


@dataclass
class QuestTrace:
    """Opt-in recorder: ``quest.trace = QuestTrace()`` to log every decision."""

    max_records: int | None = None
    records: list[QuestSelection] = field(default_factory=list)

    def add(self, record: QuestSelection) -> None:
        if self.max_records is None or len(self.records) < self.max_records:
            self.records.append(record)

    def for_layer(self, layer_id: int) -> list[QuestSelection]:
        return [r for r in self.records if r.layer_id == layer_id]


class QuestAlgorithm:
    """Quest page-wise sparse attention with bounds indexed by
    ``(layer, req_pool_idx, logical_page_in_req)``.

    Storage (bf16):

    * page bounds  ``[num_layers, max_reqs, max_pages_per_req, kv_heads, head_dim, 2]``
    * running bounds (one in-flight page per request)
      ``[num_layers, max_reqs, kv_heads, head_dim]`` x 2

    Lifecycle (who calls what):

    ======================================  =========================================
    after prefill (all layers' K written)   ``update_prefill_representations``  [Q2]
    every decode step, before the forward   ``update_decode_representations`` +
                                            ``maybe_finalize_decode_representations``
                                            for the *previous* token            [Q3]
    every decode step, before the forward   ``prepare_step``                    [Q4]
    every layer of every decode step        ``retrieve_topk``               [Q5, Q6]
    request finished                        ``invalidate_request``              [Q8]
    ======================================  =========================================
    """

    def __init__(
        self,
        top_k: int,
        page_size: int,
        device: torch.device | str,
        *,
        avoid_recent_overlap: bool = False,
    ):
        if page_size <= 0:
            raise ValueError(f"quest page_size must be > 0, got {page_size}")
        if top_k % page_size != 0:
            raise ValueError(
                f"top_k ({top_k}) must be divisible by quest_page_size ({page_size}); "
                "Quest emits whole pages worth of token positions."
            )
        self.top_k = top_k
        self.page_size = page_size
        self.top_k_pages = top_k // page_size
        self.device = torch.device(device)
        self.avoid_recent_overlap = avoid_recent_overlap

        # Set by init_storage().
        self.start_layer = 0
        self.end_layer = 0
        self.num_layers = 0
        self.max_reqs = 0
        self.max_pages_per_req = 0
        self.kv_heads = 0
        self.head_dim = 0
        self.page_k_bounds = torch.empty(0)
        self.running_k_min = torch.empty(0)
        self.running_k_max = torch.empty(0)
        self.running_token_count = torch.empty(0, dtype=torch.int32)
        self.running_page_idx = torch.empty(0, dtype=torch.int32)
        self._initialized = False

        self._step: StepLayout | None = None
        self.trace: QuestTrace | None = None

    # ------------------------------------------------------------------ setup

    def init_storage(
        self,
        start_layer: int,
        end_layer: int,
        max_reqs: int,
        max_context_len: int,
        kv_heads: int,
        head_dim: int,
    ) -> None:
        """Allocate bounds for layers ``[start_layer, end_layer)``.

        HiSparse passes the model's full layer range: Quest is applied at
        *every* layer of Qwen3-30B-A3B (the original Quest paper kept the
        first two layers dense; HiSparse does not).
        """
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.num_layers = end_layer - start_layer
        self.max_reqs = max_reqs
        self.max_pages_per_req = (max_context_len + self.page_size - 1) // self.page_size
        self.kv_heads = kv_heads
        self.head_dim = head_dim

        # Last axis indexes [min, max] so one gather brings both.
        bounds_shape = (self.num_layers, max_reqs, self.max_pages_per_req, kv_heads, head_dim, 2)
        self.page_k_bounds = torch.empty(bounds_shape, dtype=torch.bfloat16, device=self.device)
        self.page_k_bounds[..., 0].fill_(_BF16_POS_INF)
        self.page_k_bounds[..., 1].fill_(_BF16_NEG_INF)

        running_shape = (self.num_layers, max_reqs, kv_heads, head_dim)
        self.running_k_min = torch.full(
            running_shape, _BF16_POS_INF, dtype=torch.bfloat16, device=self.device
        )
        self.running_k_max = torch.full(
            running_shape, _BF16_NEG_INF, dtype=torch.bfloat16, device=self.device
        )
        self.running_token_count = torch.zeros(max_reqs, dtype=torch.int32, device=self.device)
        self.running_page_idx = torch.zeros(max_reqs, dtype=torch.int32, device=self.device)
        self._initialized = True

    @property
    def page_k_min(self) -> torch.Tensor:
        """View ``[num_layers, max_reqs, max_pages, kv_heads, head_dim]`` (writes go through)."""
        return self.page_k_bounds[..., 0]

    @property
    def page_k_max(self) -> torch.Tensor:
        return self.page_k_bounds[..., 1]

    @property
    def page_valid(self) -> torch.Tensor:
        """``[num_layers, max_reqs, max_pages]`` bool, derived from the sentinel."""
        return (self.page_k_max > _BF16_NEG_INF).any(dim=(-1, -2))

    def storage_bytes(self) -> int:
        n = self.page_k_bounds.numel() + self.running_k_min.numel() + self.running_k_max.numel()
        return n * 2

    # ------------------------------------------------------ [Q2] prefill bounds

    def update_prefill_representations(
        self,
        layer_id: int,
        req_pool_idx: int,
        k_buffer: torch.Tensor,
        prefill_indices: torch.Tensor,
    ) -> None:
        """[Q2] Bounds for one request's prefill keys at one layer.

        Args:
          layer_id: absolute layer id (offset by ``start_layer`` internally).
          req_pool_idx: the request's slot.
          k_buffer: ``[pool_size, kv_heads, head_dim]`` K buffer of this layer.
          prefill_indices: ``[prefill_len]`` physical addresses of the request's
            prefill tokens in ``k_buffer``, in position order.

        Full pages go to ``page_k_bounds``; a trailing partial page seeds the
        running bounds so decode continues where prefill stopped.  The two
        per-request counters are written once, on the last layer - the caller
        must invoke this for every layer in ``[start_layer, end_layer)``.
        """
        layer_offset = layer_id - self.start_layer
        prefill_len = int(prefill_indices.numel())
        P = self.page_size

        num_full_pages = prefill_len // P
        if num_full_pages > 0:
            full_k = k_buffer[prefill_indices[: num_full_pages * P].long()]
            paged = full_k.view(num_full_pages, P, self.kv_heads, self.head_dim)
            self.page_k_min[layer_offset, req_pool_idx, :num_full_pages] = paged.amin(dim=1).to(
                torch.bfloat16
            )
            self.page_k_max[layer_offset, req_pool_idx, :num_full_pages] = paged.amax(dim=1).to(
                torch.bfloat16
            )

        partial_count = prefill_len - num_full_pages * P
        if partial_count > 0:
            partial_k = k_buffer[prefill_indices[num_full_pages * P :].long()]
            self.running_k_min[layer_offset, req_pool_idx] = partial_k.amin(dim=0).to(
                torch.bfloat16
            )
            self.running_k_max[layer_offset, req_pool_idx] = partial_k.amax(dim=0).to(
                torch.bfloat16
            )
        # else: running buffers stay at the (+inf, -inf) sentinels = empty page.

        if layer_id == self.end_layer - 1:
            self.running_token_count[req_pool_idx] = partial_count
            self.running_page_idx[req_pool_idx] = num_full_pages

    # ------------------------------------------------------- [Q3] decode bounds

    def update_decode_representations(
        self,
        layer_id: int,
        req_indices: torch.Tensor,
        k_buffer: torch.Tensor,
        device_locs: torch.Tensor,
    ) -> None:
        """[Q3] Fold the just-decoded token's key into the running min/max.

        ``device_locs[i]`` is where request ``req_indices[i]``'s previous token
        lives in ``k_buffer``.  Counters advance once per step in
        :meth:`maybe_finalize_decode_representations`, after all layers ran.
        """
        layer_offset = layer_id - self.start_layer
        new_k = k_buffer[device_locs.long()].to(torch.bfloat16)
        cur_min = self.running_k_min[layer_offset, req_indices]
        cur_max = self.running_k_max[layer_offset, req_indices]
        self.running_k_min[layer_offset, req_indices] = torch.minimum(cur_min, new_k)
        self.running_k_max[layer_offset, req_indices] = torch.maximum(cur_max, new_k)

    def maybe_finalize_decode_representations(self, req_indices: torch.Tensor) -> None:
        """[Q3] Advance counters; finalize every page whose ``P``-th token just arrived.

        For a completing request (``running_token_count == P - 1`` before this
        step): copy running bounds into ``page_k_bounds[:, req, running_page_idx]``
        for all layers (this overwrites the sentinel, which is what makes the
        page selectable), reset the running bounds, advance ``running_page_idx``.
        """
        counts = self.running_token_count[req_indices]
        will_complete = counts == (self.page_size - 1)
        completing = req_indices[will_complete]
        page_indices = self.running_page_idx[completing].long()
        self.page_k_min[:, completing, page_indices] = self.running_k_min[:, completing]
        self.page_k_max[:, completing, page_indices] = self.running_k_max[:, completing]
        self.running_k_min[:, completing] = _BF16_POS_INF
        self.running_k_max[:, completing] = _BF16_NEG_INF
        self.running_page_idx[completing] = self.running_page_idx[completing] + 1
        self.running_token_count[req_indices] = (counts + 1) % self.page_size

    # ----------------------------------------------------- [Q4] per-step layout

    def prepare_step(self, seq_lens: torch.Tensor) -> StepLayout:
        """[Q4] Everything about the selection that does not depend on the layer.

        ``seq_lens[i]`` counts the token being decoded in this step (its K/V is
        written at position ``seq_len - 1`` during the forward).
        """
        P = self.page_size
        seq = seq_lens.to(device=self.device, dtype=torch.int64)
        last_valid = (seq - 1).clamp(min=0).unsqueeze(1)  # [bs, 1]

        recent_start = (seq - P).clamp(min=0).unsqueeze(1)
        offsets = torch.arange(P, device=self.device, dtype=torch.int64).unsqueeze(0)
        recent_positions = torch.minimum(recent_start + offsets, last_valid)  # [bs, P]

        all_positions = torch.arange(self.top_k, device=self.device, dtype=torch.int64)
        short_layout = torch.minimum(all_positions.unsqueeze(0), last_valid)  # [bs, top_k]

        is_short = (seq <= self.top_k).unsqueeze(1)
        actual_lens = seq.clamp(max=self.top_k).to(torch.int32)

        # Candidate pages: complete pages only (a page's bounds can be valid only
        # once all P tokens exist).  Optionally also drop pages that overlap the
        # recent window [seq_len - P, seq_len), i.e. page seq_len // P - 1.
        num_full_pages = seq // P
        if self.avoid_recent_overlap:
            candidate_limit = (num_full_pages - 1).clamp(min=0)
        else:
            candidate_limit = num_full_pages

        self._step = StepLayout(
            seq_lens=seq,
            recent_positions=recent_positions,
            short_layout=short_layout,
            is_short=is_short,
            actual_lens=actual_lens,
            candidate_limit=candidate_limit.unsqueeze(1),
        )
        return self._step

    # ---------------------------------------------------- [Q5] page criticality

    def group_queries(self, queries: torch.Tensor) -> torch.Tensor:
        """[Q5] ``[bs, q_heads*head_dim]`` or ``[bs, q_heads, head_dim]`` -> ``[bs, kv_heads, head_dim]``.

        Query head ``h`` reads KV head ``h // group`` (the layout used by
        ``repeat_kv``), so averaging over ``view(bs, kv_heads, group, d)``
        averages each KV head's own query heads.  Upstream calls this an
        "approximation for MQA/GQA"; it happens in the query dtype (bf16).
        """
        bs = queries.shape[0]
        if queries.dim() == 2:
            if queries.shape[1] % self.head_dim != 0:
                raise ValueError(
                    f"Query hidden {queries.shape[1]} not divisible by head_dim {self.head_dim}"
                )
            q = queries.view(bs, queries.shape[1] // self.head_dim, self.head_dim)
        elif queries.dim() == 3:
            q = queries
        else:
            raise ValueError(f"Unsupported query shape {tuple(queries.shape)}")
        q_heads = q.shape[1]
        if q_heads != self.kv_heads:
            if q_heads % self.kv_heads != 0:
                raise ValueError(f"q_heads {q_heads} not divisible by kv_heads {self.kv_heads}")
            group = q_heads // self.kv_heads
            q = q.reshape(bs, self.kv_heads, group, self.head_dim).mean(dim=2)
        return q

    def page_scores(
        self, grouped_q: torch.Tensor, layer_id: int, req_pool_indices: torch.Tensor
    ) -> torch.Tensor:
        """[Q5] Quest criticality of every page slot: ``[bs, max_pages_per_req]`` float32.

        ``score[p] = sum_{h,d} (q[h,d] >= 0 ? q[h,d]*kmax[p,h,d] : q[h,d]*kmin[p,h,d])``.
        Sentinel pages come out hugely negative (or ``-inf``), never positive.
        """
        layer_offset = layer_id - self.start_layer
        k_bounds = self.page_k_bounds[layer_offset, req_pool_indices.long()]  # [bs, P#, kv, d, 2]
        q = grouped_q.unsqueeze(1)  # [bs, 1, kv, d]
        k_chosen = torch.where(q >= 0, k_bounds[..., 1], k_bounds[..., 0])
        return (q.float() * k_chosen.float()).sum(dim=(2, 3))

    # ----------------------------------------------------------- [Q6] top-k

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """[Q6] Token positions this layer attends to.

        Returns ``(token_positions [bs, top_k] int32, actual_lens [bs] int32)``.
        Only the first ``actual_lens[i]`` entries of row ``i`` are meaningful.

        * long request (``seq_len > top_k``): slots ``[0, top_k - P)`` are the
          ``top_k/P - 1`` best complete pages expanded to positions (ordered by
          score), slots ``[top_k - P, top_k)`` are the recent window.
        * short request: ``0, 1, ..., seq_len - 1`` (dense).

        ``seq_lens`` is accepted for API parity with upstream; the layout uses
        the values cached by :meth:`prepare_step`.
        """
        step = self._step
        if step is None:
            raise RuntimeError("prepare_step() must be called once per decode step first")
        bs = queries.shape[0]
        if step.seq_lens.shape[0] != bs or seq_lens.shape[0] != bs:
            raise ValueError("batch size differs from the one given to prepare_step()")

        scores: torch.Tensor | None = None
        topk_pages: torch.Tensor | None = None
        if bool(step.is_short.all()):
            # Every request fits in the budget: dense, nothing to score.
            token_positions = step.short_layout
        else:
            q = self.group_queries(queries)
            scores = self.page_scores(q, layer_id, req_pool_indices)
            select_pages = self.top_k_pages - 1  # 0 if top_k == page_size
            if select_pages > 0:
                # Pages that cannot (yet) be candidates -> -inf.  Upstream keeps
                # this as a defensive mask against stale bounds left in a
                # reused request slot; with avoid_recent_overlap it also removes
                # the page overlapping the recent window.
                page_idx = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0)
                scores = scores.masked_fill(page_idx >= step.candidate_limit, float("-inf"))
                topk_pages = torch.topk(scores, k=select_pages, dim=1).indices  # [bs, k]
                offsets = torch.arange(self.page_size, device=scores.device)
                select_positions = (
                    topk_pages.unsqueeze(2) * self.page_size + offsets.view(1, 1, -1)
                ).reshape(bs, select_pages * self.page_size)
            else:
                select_positions = torch.empty((bs, 0), dtype=torch.int64, device=self.device)
            long_layout = torch.cat([select_positions, step.recent_positions], dim=1)
            token_positions = torch.where(step.is_short, step.short_layout, long_layout)

        if self.trace is not None:
            self._record(layer_id, req_pool_indices, step, token_positions, scores, topk_pages)
        return token_positions.to(torch.int32), step.actual_lens

    # ------------------------------------------------------ [Q8] request end

    def invalidate_request(self, req_pool_idx: int) -> None:
        """[Q8] Reset every per-request buffer when a slot is freed.

        Bounds are addressed by ``(req_pool_idx, page)``; without this reset
        the next request placed in the slot would inherit stale bounds.
        """
        self.page_k_min[:, req_pool_idx, :] = _BF16_POS_INF
        self.page_k_max[:, req_pool_idx, :] = _BF16_NEG_INF
        self.running_k_min[:, req_pool_idx] = _BF16_POS_INF
        self.running_k_max[:, req_pool_idx] = _BF16_NEG_INF
        self.running_token_count[req_pool_idx] = 0
        self.running_page_idx[req_pool_idx] = 0

    # ----------------------------------------------------------- tracing

    def _record(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        step: StepLayout,
        token_positions: torch.Tensor,
        scores: torch.Tensor | None,
        topk_pages: torch.Tensor | None,
    ) -> None:
        assert self.trace is not None
        for i in range(token_positions.shape[0]):
            seq_len = int(step.seq_lens[i])
            dense = bool(step.is_short[i, 0])
            n = int(step.actual_lens[i])
            full_pages = seq_len // self.page_size
            self.trace.add(
                QuestSelection(
                    layer_id=layer_id,
                    req_pool_idx=int(req_pool_indices[i]),
                    seq_len=seq_len,
                    dense=dense,
                    selected_pages=(
                        []
                        if dense or topk_pages is None
                        else sorted(int(p) for p in topk_pages[i].tolist())
                    ),
                    recent_window=(max(seq_len - self.page_size, 0), seq_len),
                    page_scores=(
                        None if dense or scores is None else scores[i, :full_pages].detach().cpu()
                    ),
                    positions=token_positions[i, :n].detach().to("cpu", torch.int64),
                )
            )
