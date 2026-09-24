"""Generate with the real Qwen3-30B-A3B in the three modes (needs a GPU with ~64 GB).

    python examples/02_generate.py /path/to/Qwen3-30B-A3B-Thinking-2507

The checkpoint directory is the Hugging Face snapshot (config.json,
*.safetensors, tokenizer files).  Loading takes ~61 GB of GPU memory in bf16.
"""

from __future__ import annotations

import sys
import time

import torch
from transformers import AutoTokenizer

from qwenquest import Engine, EngineConfig, load_qwen3_moe

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-30B-A3B-Thinking-2507"
HISPARSE = {"algorithm": "quest", "top_k": 2048, "quest_page_size": 64, "device_buffer_size": 4096}

tok = AutoTokenizer.from_pretrained(MODEL)
model = load_qwen3_moe(MODEL, device="cuda", dtype=torch.bfloat16)

prompt = "Explain in three sentences how query-aware sparse attention decides which tokens to read."
enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
ids = [int(t) for t in (enc["input_ids"] if hasattr(enc, "keys") else enc)]

for mode in ("dense", "quest", "quest_hisparse"):
    engine = Engine(
        model,
        EngineConfig(
            attention_mode=mode,
            hisparse_config=HISPARSE,
            max_running_requests=1,
            max_context_len=len(ids) + 512 + 1,
        ),
    )
    t0 = time.time()
    (out,) = engine.generate([ids], max_new_tokens=512)
    print(f"===== {mode}  ({len(out.output_ids)} tokens, {time.time() - t0:.1f}s)")
    print(tok.decode(out.output_ids))
    # Prompts shorter than top_k decode densely; use 03_passkey.py for long contexts.
