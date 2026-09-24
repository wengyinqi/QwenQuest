"""Numerical parity with ``transformers``' Qwen3-MoE.

A random tiny HF model is saved with ``save_pretrained`` (the checkpoint
layout of the official Qwen weights) and loaded through our loader; prefill
logits and incremental dense decode must match HF's full-sequence forward.
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from qwenquest import Engine, EngineConfig, load_qwen3_moe  # noqa: E402
from qwenquest.model.loader import read_safetensors  # noqa: E402

pytestmark = pytest.mark.hf

TINY = {
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 3,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "moe_intermediate_size": 32,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "norm_topk_prob": True,
    "max_position_embeddings": 512,
    "mlp_only_layers": [1],  # also exercise the dense-MLP branch
}


def hf_config(rope: dict):
    """Build a Qwen3MoeConfig on either side of the transformers 5 rope refactor."""
    cfg_cls = transformers.Qwen3MoeConfig
    if int(transformers.__version__.split(".")[0]) >= 5:
        return cfg_cls(**TINY, rope_parameters={"rope_theta": 1e6, **rope})
    legacy = None if rope["rope_type"] == "default" else rope
    return cfg_cls(**TINY, rope_theta=1e6, rope_scaling=legacy)


@pytest.fixture(
    params=[
        {"rope_type": "default"},
        {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 128},
    ],
    ids=["rope", "yarn"],
)
def hf_checkpoint(request, tmp_path):
    torch.manual_seed(0)
    model = transformers.Qwen3MoeForCausalLM(hf_config(request.param)).eval()
    model.config._attn_implementation = "eager"
    model.save_pretrained(tmp_path)
    return model, tmp_path


def test_prefill_logits_match(hf_checkpoint):
    hf, path = hf_checkpoint
    ours = load_qwen3_moe(path, device="cpu", dtype=torch.float32)
    ids = torch.randint(0, 256, (70,), generator=torch.Generator().manual_seed(1))

    from qwenquest.batch import ForwardBatch, ForwardMode

    engine = Engine(ours, EngineConfig(max_running_requests=1, max_context_len=128))
    req = engine.req_to_token_pool.alloc()
    locs = engine.allocator.alloc(70)
    engine.req_to_token_pool.req_to_token[req, :70] = locs.int()
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        input_ids=ids,
        positions=torch.arange(70),
        req_pool_indices=torch.tensor([req]),
        seq_lens=torch.tensor([70]),
        out_cache_loc=locs,
        req_to_token_pool=engine.req_to_token_pool,
        token_to_kv_pool=engine.token_to_kv_pool,
        attn_backend=engine.attn_backend,
        extend_prefix_lens=torch.tensor([0]),
        extend_seq_lens=torch.tensor([70]),
    )
    got = ours(batch, return_all_logits=True)
    with torch.no_grad():
        want = hf(ids[None]).logits[0].float()
    torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-4)


def test_incremental_decode_matches_full_forward(hf_checkpoint):
    hf, path = hf_checkpoint
    ours = load_qwen3_moe(path, device="cpu", dtype=torch.float32)
    prompt = torch.randint(0, 256, (50,), generator=torch.Generator().manual_seed(2)).tolist()
    engine = Engine(ours, EngineConfig(max_running_requests=1, max_context_len=128))
    (out,) = engine.generate([prompt], max_new_tokens=20, return_logits=True, stop_token_ids=[])
    with torch.no_grad():
        full = hf(torch.tensor([prompt + out.output_ids])).logits[0].float()
    torch.testing.assert_close(torch.stack(out.logits), full[49:69], atol=2e-5, rtol=1e-4)


def test_fused_expert_layout_is_split(hf_checkpoint, tmp_path_factory):
    """transformers>=5 keeps experts fused in memory; the loader accepts that layout too."""
    from safetensors.torch import save_file

    _, path = hf_checkpoint
    state = read_safetensors(path, "cpu", torch.float32)
    fused = {}
    for key, value in state.items():
        if ".mlp.experts." not in key:
            fused[key] = value
    for layer in (0, 2):  # layer 1 is a dense MLP
        prefix = f"model.layers.{layer}.mlp.experts."
        gate = torch.stack([state[f"{prefix}{e}.gate_proj.weight"] for e in range(8)])
        up = torch.stack([state[f"{prefix}{e}.up_proj.weight"] for e in range(8)])
        down = torch.stack([state[f"{prefix}{e}.down_proj.weight"] for e in range(8)])
        fused[prefix + "gate_up_proj"] = torch.cat([gate, up], dim=1)
        fused[prefix + "down_proj"] = down
    out_dir = tmp_path_factory.mktemp("fused")
    save_file(fused, str(out_dir / "model.safetensors"))
    (out_dir / "config.json").write_text((path / "config.json").read_text())
    a = load_qwen3_moe(path, device="cpu", dtype=torch.float32)
    b = load_qwen3_moe(out_dir, device="cpu", dtype=torch.float32)
    for (name, pa), (_, pb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert torch.equal(pa, pb), name


def test_loader_rejects_foreign_checkpoints(hf_checkpoint, tmp_path_factory):
    from safetensors.torch import save_file

    _, path = hf_checkpoint
    state = read_safetensors(path, "cpu", torch.float32)
    state.pop("model.norm.weight")
    bad = tmp_path_factory.mktemp("bad")
    save_file(state, str(bad / "model.safetensors"))
    (bad / "config.json").write_text((path / "config.json").read_text())
    with pytest.raises(RuntimeError, match="missing"):
        load_qwen3_moe(bad, device="cpu", dtype=torch.float32)
