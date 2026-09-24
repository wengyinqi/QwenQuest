from __future__ import annotations

import pytest
import torch

from qwenquest import Qwen3MoeForCausalLM, init_random_weights, tiny_config


@pytest.fixture(scope="session")
def tiny_model() -> Qwen3MoeForCausalLM:
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(tiny_config(num_hidden_layers=3))
    return init_random_weights(model, seed=0, std=0.05)


@pytest.fixture(scope="session")
def prompts() -> list[list[int]]:
    gen = torch.Generator().manual_seed(1234)
    return [torch.randint(0, 256, (n,), generator=gen).tolist() for n in (90, 107, 64)]
