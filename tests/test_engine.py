"""End-to-end properties of the three attention modes on a tiny Qwen3-MoE."""

from __future__ import annotations

import pytest
import torch

from qwenquest import Engine, EngineConfig, QuestTrace
from qwenquest.attention.ops import decode_attention

# A Quest budget small enough that ~100-token prompts take the sparse path
# and a hot buffer small enough that HiSparse has to swap.
SMALL_QUEST = {"top_k": 32, "quest_page_size": 8, "device_buffer_size": 48}


def run(model, prompts, mode, hisparse_config=None, max_running=2, **gen_kwargs):
    engine = Engine(
        model,
        EngineConfig(
            attention_mode=mode,
            hisparse_config=hisparse_config,
            max_running_requests=max_running,
            max_context_len=256,
        ),
    )
    gen_kwargs.setdefault("max_new_tokens", 30)
    outs = engine.generate(prompts, return_logits=True, stop_token_ids=[], **gen_kwargs)
    return engine, outs


def logits(out) -> torch.Tensor:
    return torch.stack(out.logits)


def test_prefill_logits_are_identical_across_modes(tiny_model, prompts):
    """Quest only changes decode; prefill is dense in every mode."""
    _, dense = run(tiny_model, prompts, "dense", max_new_tokens=1)
    _, quest = run(tiny_model, prompts, "quest", SMALL_QUEST, max_new_tokens=1)
    for a, b in zip(dense, quest, strict=True):
        assert torch.equal(logits(a), logits(b))


def test_quest_equals_dense_when_the_budget_covers_the_sequence(tiny_model, prompts):
    """seq_len <= top_k takes the dense layout, so results match bit for bit."""
    cfg = {"top_k": 256, "quest_page_size": 16}
    _, dense = run(tiny_model, prompts, "dense")
    _, quest = run(tiny_model, prompts, "quest", cfg)
    for a, b in zip(dense, quest, strict=True):
        assert a.output_ids == b.output_ids
        assert torch.equal(logits(a), logits(b))


def test_quest_differs_from_dense_on_the_sparse_path(tiny_model, prompts):
    _, dense = run(tiny_model, prompts, "dense")
    _, quest = run(tiny_model, prompts, "quest", SMALL_QUEST)
    assert any(not torch.equal(logits(a), logits(b)) for a, b in zip(dense, quest, strict=True))


@pytest.mark.parametrize("avoid_recent_overlap", [False, True])
def test_hisparse_offload_is_exact(tiny_model, prompts, avoid_recent_overlap):
    """Mode 3 (host pool + LRU hot buffer) reproduces Mode 2 bit for bit."""
    cfg = {**SMALL_QUEST, "avoid_recent_overlap": avoid_recent_overlap}
    _, quest = run(tiny_model, prompts, "quest", cfg)
    engine, hisparse = run(tiny_model, prompts, "quest_hisparse", cfg)
    for a, b in zip(quest, hisparse, strict=True):
        assert a.output_ids == b.output_ids
        assert torch.equal(logits(a), logits(b))
    stats = engine.attn_backend.coord.stats
    assert stats.misses > 0 and stats.hits > 0  # the buffer really had to swap


def test_hisparse_device_memory_is_bounded(tiny_model, prompts):
    engine, _ = run(tiny_model, prompts, "quest_hisparse", SMALL_QUEST)
    coord = engine.attn_backend.coord
    per_request_rows = coord.device_k[0].shape[0] // engine.req_to_token_pool.size
    assert per_request_rows == SMALL_QUEST["device_buffer_size"] + 1
    # Decode tokens never touch the shared token pool; everything was returned.
    assert engine.allocator.available_size() == engine.token_to_kv_pool.size


def test_batching_does_not_change_results(tiny_model, prompts):
    _, batched = run(tiny_model, prompts, "quest", SMALL_QUEST, max_running=3)
    _, one_by_one = run(tiny_model, prompts, "quest", SMALL_QUEST, max_running=1)
    for a, b in zip(batched, one_by_one, strict=True):
        assert a.output_ids == b.output_ids
        assert torch.allclose(logits(a), logits(b), atol=1e-5)


@pytest.mark.parametrize("mode", ["quest", "quest_hisparse"])
def test_request_slots_are_reused_cleanly(tiny_model, prompts, mode):
    """A slot's Quest bounds are reset between requests ([Q8])."""
    _, sequential = run(tiny_model, prompts * 2, mode, SMALL_QUEST, max_running=1)
    for first, second in zip(sequential[:3], sequential[3:], strict=True):
        assert first.output_ids == second.output_ids
        assert torch.equal(logits(first), logits(second))


def test_attention_reads_exactly_the_selected_tokens(tiny_model, prompts):
    """[Q7] Every sparse decode output equals attention over the traced positions."""
    engine = Engine(
        tiny_model,
        EngineConfig(attention_mode="quest", hisparse_config=SMALL_QUEST, max_context_len=256),
    )
    quest = engine.quest
    assert quest is not None
    quest.trace = QuestTrace()
    backend = engine.attn_backend
    original = backend.forward_decode
    checked: list[bool] = []

    def spy(q, k, v, layer, batch):
        out = original(q, k, v, layer, batch)
        rec = quest.trace.records[-1]  # the decision this call just made
        assert rec.layer_id == layer.layer_id and not rec.dense
        req = int(batch.req_pool_indices[0])
        slots = engine.req_to_token_pool.req_to_token[req, rec.positions].long()
        k_buf, v_buf = engine.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        expected = decode_attention(q[0], k_buf[slots], v_buf[slots], layer.scaling)
        checked.append(torch.allclose(out[0], expected, atol=1e-6))
        return out

    backend.forward_decode = spy  # type: ignore[method-assign]
    engine.generate([prompts[1]], max_new_tokens=3, stop_token_ids=[])
    assert len(checked) == 2 * tiny_model.config.num_hidden_layers and all(checked)


def test_engine_rejects_bad_input(tiny_model):
    with pytest.raises(ValueError):
        Engine(tiny_model, EngineConfig(attention_mode="nope"))
    engine = Engine(tiny_model, EngineConfig(max_context_len=16))
    with pytest.raises(ValueError):
        engine.generate([list(range(16))])
    with pytest.raises(ValueError):
        engine.generate([[1, 2, 3]], max_new_tokens=0)


def test_generation_stops_at_max_context(tiny_model):
    engine = Engine(tiny_model, EngineConfig(max_context_len=20))
    (out,) = engine.generate([list(range(10))], max_new_tokens=100, stop_token_ids=[])
    assert out.finish_reason == "length"
    assert len(out.output_ids) == 11  # positions 10..19 decoded + the token after the last


def test_sampling_is_seeded(tiny_model, prompts):
    engine = Engine(tiny_model, EngineConfig(max_context_len=256))
    a = engine.generate(prompts[:1], max_new_tokens=8, temperature=1.0, top_p=0.9, seed=3)
    b = engine.generate(prompts[:1], max_new_tokens=8, temperature=1.0, top_p=0.9, seed=3)
    assert a[0].output_ids == b[0].output_ids
