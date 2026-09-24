"""Load Hugging Face Qwen3-MoE checkpoints (safetensors) into our model.

The official checkpoints (and ``transformers`` ``save_pretrained``) store one
tensor per expert (``mlp.experts.{e}.gate_proj.weight`` ...) - the layout our
module tree uses - so loading is a plain ``load_state_dict``.  The fused
``mlp.experts.gate_up_proj`` / ``down_proj`` layout used inside
transformers>=5 is split back into per-expert tensors if encountered.

The model is built on the ``meta`` device and tensors are *assigned* straight
from safetensors onto the target device, so Qwen3-30B-A3B (61 GB in bf16)
is never materialised twice.
"""

from __future__ import annotations

import glob
import os

import torch

from qwenquest.config import Qwen3MoeConfig
from qwenquest.model.qwen3_moe import Qwen3MoeForCausalLM

__all__ = ["init_random_weights", "load_qwen3_moe", "read_safetensors"]


def read_safetensors(
    path: str | os.PathLike[str], device: torch.device | str, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    files = sorted(glob.glob(os.path.join(os.fspath(path), "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors under {path}")
    state: dict[str, torch.Tensor] = {}
    for file in files:
        with safe_open(file, framework="pt", device=str(device)) as f:
            for key in f.keys():  # noqa: SIM118 - safe_open is not a Mapping
                t = f.get_tensor(key)
                state[key] = t.to(dtype) if t.is_floating_point() else t
    return state


def _split_fused_experts(state: dict[str, torch.Tensor]) -> None:
    for key in list(state):
        if key.endswith("mlp.experts.gate_up_proj"):
            prefix = key[: -len("gate_up_proj")]
            gate, up = state.pop(key).chunk(2, dim=1)  # [E, 2I, H] -> 2 x [E, I, H]
            for e in range(gate.shape[0]):
                state[f"{prefix}{e}.gate_proj.weight"] = gate[e]
                state[f"{prefix}{e}.up_proj.weight"] = up[e]
        elif key.endswith("mlp.experts.down_proj"):
            prefix = key[: -len("down_proj")]
            down = state.pop(key)  # [E, H, I]
            for e in range(down.shape[0]):
                state[f"{prefix}{e}.down_proj.weight"] = down[e]


def load_qwen3_moe(
    path: str | os.PathLike[str],
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    config: Qwen3MoeConfig | None = None,
) -> Qwen3MoeForCausalLM:
    """Build ``Qwen3MoeForCausalLM`` from a local HF checkpoint directory."""
    config = config or Qwen3MoeConfig.from_pretrained(path)
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(config)
    state = read_safetensors(path, device, dtype)
    _split_fused_experts(state)
    if config.tie_word_embeddings:
        state.pop("lm_head.weight", None)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if config.tie_word_embeddings:
        missing = [k for k in missing if k != "lm_head.weight"]
        model.lm_head.weight = model.model.embed_tokens.weight
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match Qwen3MoeForCausalLM: missing={missing[:8]} "
            f"unexpected={unexpected[:8]}"
        )
    return model.eval()


@torch.no_grad()
def init_random_weights(
    model: Qwen3MoeForCausalLM, seed: int = 0, std: float = 0.02
) -> Qwen3MoeForCausalLM:
    """Deterministic HF-style init (normal(0, std), norms = 1) for demos/tests."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for name, param in model.named_parameters():
        if name.endswith("norm.weight"):
            param.fill_(1.0)
        else:
            param.copy_(torch.randn(param.shape, generator=gen) * std)
    return model.eval()
