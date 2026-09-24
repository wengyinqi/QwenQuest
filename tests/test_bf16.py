"""bf16 end to end (the dtype HiSparse requires for its KV cache).

Runs on CPU everywhere and additionally on CUDA when a GPU is present.
"""

from __future__ import annotations

import pytest
import torch

from qwenquest import Engine, EngineConfig, Qwen3MoeForCausalLM, init_random_weights, tiny_config

QUEST = {"top_k": 32, "quest_page_size": 8, "device_buffer_size": 40}

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.gpu,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
        ],
    ),
]


@pytest.mark.parametrize("device", DEVICES)
def test_bf16_quest_and_hisparse_agree(device):
    model = init_random_weights(Qwen3MoeForCausalLM(tiny_config()), seed=1, std=0.05)
    model = model.to(device=device, dtype=torch.bfloat16)
    gen = torch.Generator().manual_seed(0)
    prompts = [torch.randint(0, 256, (n,), generator=gen).tolist() for n in (70, 45)]
    results = {}
    for mode in ("quest", "quest_hisparse"):
        engine = Engine(
            model,
            EngineConfig(attention_mode=mode, hisparse_config=QUEST, max_context_len=128),
        )
        assert engine.token_to_kv_pool.dtype == torch.bfloat16
        results[mode] = engine.generate(prompts, max_new_tokens=20, return_logits=True)
    for a, b in zip(results["quest"], results["quest_hisparse"], strict=True):
        assert a.output_ids == b.output_ids
        assert torch.equal(torch.stack(a.logits), torch.stack(b.logits))
