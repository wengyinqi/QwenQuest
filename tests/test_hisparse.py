"""HiSparseCoordinator: the hierarchical cache must be *exact*.

Whatever positions Quest selects, the device rows handed to attention must
hold exactly those positions' K/V - across admission, eager backup, LRU
eviction, the newest-token slot and duplicate selections.
"""

from __future__ import annotations

import pytest
import torch

from qwenquest.hisparse import HiSparseCoordinator
from qwenquest.memory import MHATokenToKVPool
from qwenquest.quest import QuestAlgorithm

LAYERS, KV, D = 2, 2, 8


def make(buf: int, top_k: int, page: int = 1, max_reqs: int = 2, max_ctx: int = 256):
    quest = QuestAlgorithm(top_k=top_k, page_size=page, device="cpu")
    quest.init_storage(0, LAYERS, max_reqs, max_ctx, KV, D)
    coord = HiSparseCoordinator(
        quest=quest,
        num_layers=LAYERS,
        kv_heads=KV,
        head_dim=D,
        max_reqs=max_reqs,
        max_context_len=max_ctx,
        top_k=top_k,
        device_buffer_size=buf,
        host_pool_size=max_reqs * max_ctx,
        dtype=torch.float32,
        device="cpu",
    )
    pool = MHATokenToKVPool(max_ctx, LAYERS, KV, D, torch.float32, "cpu")
    return quest, coord, pool


def truth(total: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    k = torch.randn(LAYERS, total, KV, D, generator=gen)
    v = torch.randn(LAYERS, total, KV, D, generator=gen)
    return k, v


def admit(coord, pool, req: int, n: int, k, v) -> None:
    locs = torch.arange(1, n + 1)  # token-pool slot of position p is p + 1
    for layer in range(LAYERS):
        pool.set_kv_buffer(layer, locs, k[layer, :n], v[layer, :n])
    coord.admit_request(req, n, pool, locs)


def decode_step(coord, req: int, pos: int, k, v) -> tuple[torch.Tensor, torch.Tensor]:
    """Begin the step that decodes ``pos`` and write its K/V; return (reqs, seq_lens)."""
    reqs, seq_lens = torch.tensor([req]), torch.tensor([pos + 1])
    coord.prepare_decode_step(reqs, seq_lens)
    for layer in range(LAYERS):
        coord.write_new_token(
            layer, reqs, seq_lens, k[layer, pos : pos + 1], v[layer, pos : pos + 1]
        )
    return reqs, seq_lens


@pytest.mark.parametrize(("prefill_len", "buf", "top_k"), [(10, 16, 8), (40, 16, 8), (5, 32, 32)])
def test_swap_in_is_exact_under_random_selections(prefill_len, buf, top_k):
    total = 120
    k, v = truth(total)
    _, coord, pool = make(buf, top_k)
    admit(coord, pool, 1, prefill_len, k, v)
    gen = torch.Generator().manual_seed(7)
    for pos in range(prefill_len, total):
        reqs, seq_lens = decode_step(coord, 1, pos, k, v)
        seq_len = pos + 1
        n = min(seq_len, top_k)
        for layer in range(LAYERS):
            if seq_len <= top_k:
                sel = torch.arange(n)
            else:  # random picks with duplicates, always including the newest token
                sel = torch.randint(0, seq_len, (n,), generator=gen)
                sel[-1] = seq_len - 1
                sel[0] = sel[1]
            rows = coord.swap_in_selected_pages(reqs, seq_lens, sel[None].int(), layer)
            got_k = coord.device_k[layer][rows[0, :n]]
            got_v = coord.device_v[layer][rows[0, :n]]
            assert torch.equal(got_k, k[layer, sel]), f"layer {layer} pos {pos}"
            assert torch.equal(got_v, v[layer, sel])
            if seq_len > top_k:
                assert rows[0, 0] == rows[0, 1]  # a duplicate maps to one row
    assert coord.stats.misses > 0


def test_every_decoded_token_is_backed_up_to_host():
    k, v = truth(64)
    _, coord, pool = make(buf=8, top_k=8)
    admit(coord, pool, 0, 20, k, v)
    for pos in range(20, 40):
        decode_step(coord, 0, pos, k, v)
    decode_step(coord, 0, 40, k, v)  # backs up position 39
    host_idx = coord.req_to_host_pool[0, :40]
    assert torch.all(host_idx >= 0)
    for layer in range(LAYERS):
        assert torch.equal(coord.host_k[layer][host_idx], k[layer, :40])
        assert torch.equal(coord.host_v[layer][host_idx], v[layer, :40])
    assert int(coord.req_to_host_pool[0, 40]) == -1  # newest token lives only on device


def test_lru_replacement_order():
    """buf=4 slots, prefill 10: slots 0..3 start with positions 0..3, LRU order [0,1,2,3]."""
    k, v = truth(32)
    _, coord, pool = make(buf=4, top_k=2)
    admit(coord, pool, 0, 10, k, v)

    def select(pos: int, positions: list[int]) -> None:
        reqs, seq_lens = decode_step(coord, 0, pos, k, v)
        coord.swap_in_selected_pages(reqs, seq_lens, torch.tensor([positions]).int(), 0)

    select(10, [5, 10])  # 10 = newest (reserved slot); 5 misses -> evicts slot 0
    assert coord.buffer_tokens[0, 0].tolist() == [5, 1, 2, 3]
    assert coord.lru_slots[0, 0].tolist() == [1, 2, 3, 0]

    select(11, [1, 5])  # both hit: no copy, they become most recently used
    assert (coord.stats.hits, coord.stats.misses) == (2, 1)
    assert coord.lru_slots[0, 0].tolist() == [2, 3, 1, 0]

    select(12, [7, 8])  # two misses evict the two least recently used slots (2, 3)
    assert coord.buffer_tokens[0, 0].tolist() == [5, 1, 7, 8]
    assert coord.lru_slots[0, 0].tolist() == [1, 0, 2, 3]
    assert (coord.stats.hits, coord.stats.misses) == (2, 3)


def test_fast_path_while_sequence_fits_the_buffer():
    k, v = truth(32)
    _, coord, pool = make(buf=16, top_k=16)
    admit(coord, pool, 0, 6, k, v)
    reqs, seq_lens = decode_step(coord, 0, 6, k, v)
    rows = coord.swap_in_selected_pages(reqs, seq_lens, torch.arange(16)[None].int(), 0)
    assert rows[0, :7].tolist() == coord.req_to_device_buffer[0, :7].tolist()
    assert torch.all(rows[0, 7:] == -1)
    assert coord.stats.fast_path_calls == 1 and coord.stats.misses == 0


def test_request_finished_releases_everything():
    k, v = truth(64)
    quest, coord, pool = make(buf=8, top_k=8, page=4)
    free_before = coord.host_allocator.available_size()
    admit(coord, pool, 1, 30, k, v)
    for pos in range(30, 36):
        decode_step(coord, 1, pos, k, v)
    assert quest.page_valid[:, 1].any()
    coord.request_finished(1)
    assert coord.host_allocator.available_size() == free_before
    assert torch.all(coord.req_to_host_pool[1] == -1)
    assert torch.all(coord.buffer_tokens[:, 1] == -1)
    assert not quest.page_valid[:, 1].any()


def test_coordinator_validates_sizes():
    quest = QuestAlgorithm(top_k=8, page_size=4, device="cpu")
    quest.init_storage(0, LAYERS, 1, 64, KV, D)
    common = {
        "quest": quest,
        "num_layers": LAYERS,
        "kv_heads": KV,
        "head_dim": D,
        "max_reqs": 1,
        "max_context_len": 64,
        "host_pool_size": 64,
        "dtype": torch.float32,
        "device": "cpu",
    }
    with pytest.raises(ValueError):
        HiSparseCoordinator(top_k=8, device_buffer_size=4, **common)
    with pytest.raises(ValueError):
        HiSparseCoordinator(top_k=16, device_buffer_size=32, **common)
