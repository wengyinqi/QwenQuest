"""Quest's top-k, step by step, on synthetic keys (CPU, no model needed).

    python examples/01_topk_by_hand.py

Uses Qwen3-30B-A3B's attention geometry (32 query heads, 4 KV heads,
head_dim 128) with a scaled-down budget (top_k=64, page=16) so every
intermediate tensor can be printed.  A "needle" is planted in page 5: its
keys align with the query, so Quest must rank page 5 first even though it is
far from the recent window.
"""

from __future__ import annotations

import torch

from qwenquest.quest import QuestAlgorithm

Q_HEADS, KV_HEADS, HEAD_DIM = 32, 4, 128  # Qwen3-30B-A3B
TOP_K, PAGE = 64, 16  # HiSparse uses 2048 / 64; same structure, smaller numbers
SEQ_LEN = 200  # the token being decoded is position 199

torch.manual_seed(0)

# ---------------------------------------------------------------- the data
query = torch.randn(Q_HEADS, HEAD_DIM)  # q of the current token (post norm + RoPE)
keys = torch.randn(SEQ_LEN, KV_HEADS, HEAD_DIM) * 0.5  # the KV cache's keys
group_mean = query.view(KV_HEADS, Q_HEADS // KV_HEADS, HEAD_DIM).mean(1)
keys[80:96] += 0.8 * group_mean  # needle in page 5 = positions [80, 96)
keys = keys.to(torch.bfloat16)  # HiSparse requires a bf16 KV cache

quest = QuestAlgorithm(top_k=TOP_K, page_size=PAGE, device="cpu")
quest.init_storage(
    start_layer=0,
    end_layer=1,
    max_reqs=1,
    max_context_len=256,
    kv_heads=KV_HEADS,
    head_dim=HEAD_DIM,
)

# ------------------------------------------------ [Q2] page bounds (prefill)
# Pretend positions 0..198 came from prefill (token pool index == position).
quest.update_prefill_representations(0, 0, keys, torch.arange(SEQ_LEN - 1))
full = (SEQ_LEN - 1) // PAGE
print(
    f"[Q2] prefill of {SEQ_LEN - 1} tokens -> {full} complete pages with bounds, "
    f"{(SEQ_LEN - 1) % PAGE} tokens in the running page"
)
print(
    f"     page_k_bounds[layer=0, req=0] shape = {tuple(quest.page_k_bounds[0, 0].shape)}"
    "  (pages, kv_heads, head_dim, [min,max])"
)

# ------------------------------------------------ [Q4] per-step layout
step = quest.prepare_step(torch.tensor([SEQ_LEN]))
print(
    f"\n[Q4] seq_len={SEQ_LEN} > top_k={TOP_K}: sparse.  budget = "
    f"{TOP_K // PAGE - 1} pages + recent window {step.recent_positions[0, [0, -1]].tolist()}"
)

# ------------------------------------------------ [Q5] criticality
q_bar = quest.group_queries(query.unsqueeze(0))  # [1, 4, 128]
print(
    f"\n[Q5] GQA: {Q_HEADS} query heads averaged per group of {Q_HEADS // KV_HEADS} "
    f"-> q_bar {tuple(q_bar.shape[1:])}"
)
scores = quest.page_scores(q_bar, 0, torch.tensor([0]))[0, :full]
true_max = torch.einsum("hd,thd->t", q_bar[0], keys.float())[: full * PAGE]
true_max = true_max.view(full, PAGE).amax(1)
print("     page | Quest score (upper bound) | true max_t sum_h q_bar.k")
for p in range(full):
    flag = "  <- needle" if p == 5 else ""
    print(f"     {p:4d} | {scores[p]:25.2f} | {true_max[p]:10.2f}{flag}")
assert torch.all(scores >= true_max - 1e-3), "Quest score must upper-bound every token"

# ------------------------------------------------ [Q6] top-k
positions, actual = quest.retrieve_topk(
    query.view(1, -1), 0, torch.tensor([0]), torch.tensor([SEQ_LEN])
)
pages = sorted({int(p) // PAGE for p in positions[0, : TOP_K - PAGE]})
print(f"\n[Q6] selected pages {pages} + recent window -> {int(actual[0])} positions")
print(f"     first 16 slots: {positions[0, :16].tolist()}")
print(f"     last 16 slots : {positions[0, -16:].tolist()}")
assert 5 in pages, "the needle page must be selected"
dup = positions[0].numel() - positions[0].unique().numel()
print(f"     positions listed twice (selected page overlapping the window): {dup}")
