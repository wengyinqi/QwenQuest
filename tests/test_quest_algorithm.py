"""QuestAlgorithm unit tests.

The first half ports ``test/registered/unit/managers/test_quest_unit.py``
(``TestQuestAlgorithm``) from the upstream ``hisparse_quest`` branch to CPU;
the second half pins down properties the upstream tests leave implicit.
"""

from __future__ import annotations

import pytest
import torch

from qwenquest.quest import QuestAlgorithm, QuestTrace

TOP_K = 256
PAGE = 64
KV_HEADS = 4
HEAD_DIM = 32
LAYERS = 2
MAX_REQS = 8
MAX_CTX = 2048
BF16_MAX = torch.finfo(torch.bfloat16).max
BF16_MIN = torch.finfo(torch.bfloat16).min


def make_quest(
    top_k=TOP_K, page=PAGE, kv_heads=KV_HEADS, head_dim=HEAD_DIM, **kw
) -> QuestAlgorithm:
    q = QuestAlgorithm(top_k=top_k, page_size=page, device="cpu", **kw)
    q.init_storage(0, LAYERS, MAX_REQS, MAX_CTX, kv_heads, head_dim)
    return q


def make_k(pool_size, seed=0, dtype=torch.bfloat16, kv_heads=KV_HEADS, head_dim=HEAD_DIM):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(pool_size, kv_heads, head_dim, generator=gen).to(dtype)


def prefill(q: QuestAlgorithm, req: int, pool: torch.Tensor, n: int) -> None:
    idx = torch.arange(n, dtype=torch.int32)
    for layer in range(LAYERS):
        q.update_prefill_representations(layer, req, pool, idx)


def retrieve(q: QuestAlgorithm, queries, reqs, seq_lens, layer=0):
    reqs = torch.as_tensor(reqs, dtype=torch.int64)
    seq_lens = torch.as_tensor(seq_lens, dtype=torch.int64)
    q.prepare_step(seq_lens)
    return q.retrieve_topk(queries, layer, reqs, seq_lens)


# ------------------------------------------------------------ ported upstream


def test_invalid_top_k_page_size_combo_rejected():
    with pytest.raises(ValueError):
        QuestAlgorithm(top_k=100, page_size=64, device="cpu")
    with pytest.raises(ValueError):
        QuestAlgorithm(top_k=64, page_size=0, device="cpu")


def test_init_storage_shapes_and_sentinels():
    q = make_quest()
    assert q.page_k_min.shape == (LAYERS, MAX_REQS, MAX_CTX // PAGE, KV_HEADS, HEAD_DIM)
    assert q.page_k_bounds.dtype == torch.bfloat16
    assert q.page_valid.shape == (LAYERS, MAX_REQS, MAX_CTX // PAGE)
    assert not q.page_valid.any()
    assert torch.all(q.running_k_min == BF16_MAX) and torch.all(q.running_k_max == BF16_MIN)
    assert torch.all(q.running_token_count == 0) and torch.all(q.running_page_idx == 0)


def test_prefill_bounds_full_pages_only():
    q, pool = make_quest(), make_k(512)
    prefill(q, 2, pool, 192)  # exactly 3 pages
    assert q.page_valid[:, 2, :3].all() and not q.page_valid[:, 2, 3:].any()
    for layer in range(LAYERS):
        for p in range(3):
            slab = pool[p * PAGE : (p + 1) * PAGE]
            assert torch.equal(q.page_k_min[layer, 2, p], slab.amin(0))
            assert torch.equal(q.page_k_max[layer, 2, p], slab.amax(0))
    assert int(q.running_token_count[2]) == 0 and int(q.running_page_idx[2]) == 3


def test_prefill_bounds_partial_last_page_seeds_running():
    q, pool = make_quest(), make_k(512)
    prefill(q, 1, pool, 200)  # 3 full pages + 8 tokens
    assert q.page_valid[:, 1, :3].all() and not q.page_valid[:, 1, 3:].any()
    for layer in range(LAYERS):
        assert torch.equal(q.running_k_min[layer, 1], pool[192:200].amin(0))
        assert torch.equal(q.running_k_max[layer, 1], pool[192:200].amax(0))
    assert int(q.running_token_count[1]) == 8 and int(q.running_page_idx[1]) == 3


def test_prefill_bounds_short_seq_no_full_page():
    q, pool = make_quest(), make_k(128)
    prefill(q, 0, pool, 30)
    assert not q.page_valid[:, 0].any()
    assert int(q.running_token_count[0]) == 30 and int(q.running_page_idx[0]) == 0


def _decode(q: QuestAlgorithm, req: int, pool: torch.Tensor, token_locs) -> None:
    reqs = torch.tensor([req])
    for loc in token_locs:
        for layer in range(LAYERS):
            q.update_decode_representations(layer, reqs, pool, torch.tensor([loc]))
        q.maybe_finalize_decode_representations(reqs)


def test_decode_bounds_finalise_completed_page():
    q, pool = make_quest(), make_k(512, seed=11)
    prefill(q, 3, pool, 64)
    _decode(q, 3, pool, range(64, 128))
    assert q.page_valid[:, 3, :2].all()
    for layer in range(LAYERS):
        assert torch.equal(q.page_k_min[layer, 3, 1], pool[64:128].amin(0))
        assert torch.equal(q.page_k_max[layer, 3, 1], pool[64:128].amax(0))
    assert int(q.running_token_count[3]) == 0 and int(q.running_page_idx[3]) == 2
    assert torch.all(q.running_k_min[:, 3] == BF16_MAX)
    assert torch.all(q.running_k_max[:, 3] == BF16_MIN)


def test_decode_bounds_continue_partial_prefill():
    q, pool = make_quest(), make_k(256, seed=22)
    prefill(q, 4, pool, 50)
    _decode(q, 4, pool, range(50, 64))
    assert q.page_valid[:, 4, 0].all()
    assert torch.equal(q.page_k_min[0, 4, 0], pool[:64].amin(0))
    assert int(q.running_page_idx[4]) == 1 and int(q.running_token_count[4]) == 0


def test_page_is_not_valid_until_its_last_token_arrives():
    q, pool = make_quest(), make_k(256)
    prefill(q, 0, pool, 64)
    _decode(q, 0, pool, range(64, 127))  # 63 of 64 tokens
    assert not q.page_valid[:, 0, 1].any()
    _decode(q, 0, pool, [127])
    assert q.page_valid[:, 0, 1].all()


def test_invalidate_request_resets_all_per_request_state():
    q, pool = make_quest(), make_k(512)
    prefill(q, 5, pool, 150)
    q.invalidate_request(5)
    assert not q.page_valid[:, 5].any()
    assert int(q.running_token_count[5]) == 0 and int(q.running_page_idx[5]) == 0
    assert torch.all(q.running_k_min[:, 5] == BF16_MAX)
    assert torch.all(q.running_k_max[:, 5] == BF16_MIN)


def test_invalidate_does_not_affect_other_requests():
    q, pool = make_quest(), make_k(512)
    for r in (1, 2, 3):
        prefill(q, r, pool, 128)
    q.invalidate_request(2)
    assert q.page_valid[:, 1, :2].all() and q.page_valid[:, 3, :2].all()
    assert not q.page_valid[:, 2].any()


def test_retrieve_topk_contract():
    q, pool = make_quest(), make_k(2048, seed=99)
    prefill(q, 0, pool, 800)
    queries = torch.randn(1, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    positions, actual = retrieve(q, queries, [0], [800])
    assert positions.shape == (1, TOP_K) and positions.dtype == torch.int32
    assert actual.shape == (1,) and actual.dtype == torch.int32
    assert int(actual[0]) == TOP_K
    assert positions.min() >= 0 and positions.max() < 800


def test_retrieve_topk_long_seq_ends_with_recent_window():
    q, pool = make_quest(), make_k(2048, seed=99)
    prefill(q, 0, pool, 800)
    positions, _ = retrieve(q, torch.randn(1, KV_HEADS, HEAD_DIM), [0], [800])
    assert positions[0, -PAGE:].tolist() == list(range(800 - PAGE, 800))


def test_retrieve_topk_short_seq_is_dense():
    q, pool = make_quest(), make_k(256, seed=77)
    prefill(q, 0, pool, 128)
    positions, actual = retrieve(q, torch.randn(1, KV_HEADS, HEAD_DIM), [0], [128])
    assert int(actual[0]) == 128
    assert positions[0, :128].tolist() == list(range(128))
    assert int(positions[0, 128:].max()) == 127  # padding, ignored via actual_lens


def test_retrieve_topk_picks_high_score_pages():
    q = make_quest()
    pool = torch.zeros(2048, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    pool[320:384] = 5.0  # page 5
    prefill(q, 0, pool, 640)
    positions, _ = retrieve(q, torch.ones(1, KV_HEADS, HEAD_DIM), [0], [640])
    assert set(range(320, 384)) <= set(positions[0].tolist())


# ------------------------------------------------------------ properties


def test_score_upper_bounds_every_token_of_the_page():
    """Quest's guarantee: score[p] >= sum_h q_h . k_{t,h} for every token t in page p."""
    q = make_quest(top_k=128, page=16)
    pool = make_k(512, seed=5)  # bf16 keys -> bounds are exact
    prefill(q, 0, pool, 512)
    for seed in range(5):
        query = torch.randn(1, KV_HEADS, HEAD_DIM, generator=torch.Generator().manual_seed(seed))
        scores = q.page_scores(query, 0, torch.tensor([0]))[0, :32]
        per_token = torch.einsum("hd,thd->t", query[0], pool.float())  # [512]
        best_in_page = per_token.view(32, 16).amax(dim=1)
        assert torch.all(scores >= best_in_page - 1e-3)


def test_scores_match_a_naive_loop():
    q = make_quest(top_k=128, page=16)
    pool = make_k(256, seed=6)
    prefill(q, 0, pool, 256)
    query = torch.randn(1, KV_HEADS, HEAD_DIM)
    fast = q.page_scores(query, 1, torch.tensor([0]))[0, :16]
    for p in range(16):
        kmin, kmax = q.page_k_min[1, 0, p].float(), q.page_k_max[1, 0, p].float()
        naive = sum(
            max(float(query[0, h, d] * kmin[h, d]), float(query[0, h, d] * kmax[h, d]))
            for h in range(KV_HEADS)
            for d in range(HEAD_DIM)
        )
        assert abs(float(fast[p]) - naive) < 1e-3


def test_gqa_queries_are_averaged_per_kv_group():
    q = make_quest()
    queries = torch.randn(3, 4 * KV_HEADS, HEAD_DIM)  # group of 4
    grouped = q.group_queries(queries.view(3, -1))
    for h in range(KV_HEADS):
        assert torch.allclose(grouped[:, h], queries[:, 4 * h : 4 * h + 4].mean(1))
    with pytest.raises(ValueError):
        q.group_queries(torch.randn(3, 6, HEAD_DIM))


def test_unfinished_and_out_of_range_pages_are_never_selected():
    """Pages without bounds (sentinel) or beyond seq_len // P never win, however large q is."""
    q = make_quest(top_k=128, page=16)
    pool = make_k(1024, seed=3)
    big = 100 * torch.randn(1, KV_HEADS, HEAD_DIM, generator=torch.Generator().manual_seed(0))

    prefill(q, 0, pool, 176)  # pages 0..10 complete
    _decode(q, 0, pool, range(176, 191))  # page 11 holds 15/16 tokens: no bounds yet
    positions, _ = retrieve(q, big, [0], [192])  # token 191 is being decoded
    assert int(positions[0, : 128 - 16].max()) < 176

    prefill(q, 1, pool, 1000)  # stale bounds far beyond the current length
    positions, _ = retrieve(q, big, [1], [300])
    assert int(positions[0, : 128 - 16].max()) < (300 // 16) * 16


def test_recent_window_overlap_upstream_vs_avoid():
    """When seq_len % P != 0 the recent window overlaps the newest complete page.

    Upstream may select that page too, listing the overlap twice; with
    ``avoid_recent_overlap`` the positions are always distinct.
    """
    pool = torch.zeros(2048, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    pool[48:64] = 3.0  # make the newest complete page (page 3 of 16 tokens) win
    queries = torch.ones(1, KV_HEADS, HEAD_DIM)
    seq_len = 70  # recent window [54, 70) overlaps page 3 = [48, 64)

    upstream = make_quest(top_k=32, page=16)
    prefill(upstream, 0, pool, seq_len)
    pos, actual = retrieve(upstream, queries, [0], [seq_len])
    assert sorted(pos[0].tolist()) != sorted(set(pos[0].tolist()))  # duplicates
    assert set(range(54, 64)) <= set(pos[0, :16].tolist())

    fixed = make_quest(top_k=32, page=16, avoid_recent_overlap=True)
    prefill(fixed, 0, pool, seq_len)
    pos, actual = retrieve(fixed, queries, [0], [seq_len])
    assert len(set(pos[0].tolist())) == int(actual[0]) == 32
    assert pos[0, -16:].tolist() == list(range(54, 70))


def test_mixed_batch_of_short_and_long_requests():
    q, pool = make_quest(top_k=64, page=16), make_k(1024, seed=8)
    prefill(q, 0, pool, 40)  # short
    prefill(q, 1, pool, 300)  # long
    positions, actual = retrieve(q, torch.randn(2, KV_HEADS, HEAD_DIM), [0, 1], [40, 300])
    assert actual.tolist() == [40, 64]
    assert positions[0, :40].tolist() == list(range(40))
    assert positions[1, -16:].tolist() == list(range(284, 300))
    assert int(positions[1, :48].max()) < 288


def test_retrieve_topk_requires_prepare_step():
    q = make_quest()
    with pytest.raises(RuntimeError):
        q.retrieve_topk(torch.randn(1, KV_HEADS, HEAD_DIM), 0, torch.tensor([0]), torch.tensor([5]))


def test_trace_records_decisions():
    q, pool = make_quest(top_k=64, page=16), make_k(1024, seed=9)
    q.trace = QuestTrace()
    prefill(q, 0, pool, 500)
    positions, _ = retrieve(q, torch.randn(1, KV_HEADS, HEAD_DIM), [0], [500], layer=1)
    (rec,) = q.trace.records
    assert rec.layer_id == 1 and not rec.dense and len(rec.selected_pages) == 3
    assert rec.page_scores is not None and rec.page_scores.shape == (500 // 16,)
    assert rec.recent_window == (484, 500)
    assert torch.equal(rec.positions, positions[0].long())
