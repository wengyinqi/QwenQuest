"""Long-context passkey retrieval: dense vs Quest vs Quest+HiSparse.

    python examples/03_passkey.py /path/to/Qwen3-30B-A3B-Thinking-2507 --lengths 4096 16384 32768

A random 5-digit passkey is hidden at several depths of a filler text; each
mode must repeat it.  With top_k=2048 Quest reads ~2048 of the N context
tokens per layer per step, so this directly shows whether the page bounds
find the needle.  Also reports the HiSparse hot-buffer hit rate.

Thinking models reason before answering; the check accepts the passkey
anywhere in the generated text, and --max-new-tokens bounds the reasoning.
"""

from __future__ import annotations

import argparse
import random
import re

import torch
from transformers import AutoTokenizer

from qwenquest import Engine, EngineConfig, load_qwen3_moe

FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
)


def build_prompt(tok, n_tokens: int, depth: float, passkey: str) -> list[int]:
    needle = f" The pass key is {passkey}. Remember it. {passkey} is the pass key. "
    unit = len(tok(FILLER)["input_ids"])
    n_units = max(1, (n_tokens - 200) // unit)
    parts = [FILLER] * n_units
    parts.insert(int(depth * n_units), needle)
    question = (
        "There is an important pass key hidden inside a lot of irrelevant text. Find it.\n\n"
        + "".join(parts)
        + "\n\nWhat is the pass key? Answer with the number only."
    )
    enc = tok.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return [int(t) for t in (enc["input_ids"] if hasattr(enc, "keys") else enc)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--modes", nargs="+", default=["dense", "quest", "quest_hisparse"])
    ap.add_argument("--hisparse-config", default='{"algorithm": "quest", "top_k": 2048}')
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_qwen3_moe(args.model, device="cuda", dtype=torch.bfloat16)
    rng = random.Random(0)
    for n in args.lengths:
        for mode in args.modes:
            engine = Engine(
                model,
                EngineConfig(
                    attention_mode=mode,
                    hisparse_config=args.hisparse_config,
                    max_running_requests=1,
                    max_context_len=n + args.max_new_tokens + 64,
                ),
            )
            correct = 0
            for depth in args.depths:
                key = str(rng.randint(10000, 99999))
                ids = build_prompt(tok, n, depth, key)
                (out,) = engine.generate([ids], max_new_tokens=args.max_new_tokens)
                answer = tok.decode(out.output_ids, skip_special_tokens=True)
                correct += key in re.findall(r"\d{5}", answer)
            extra = ""
            if mode == "quest_hisparse":
                extra = f"  hot-buffer hit rate {engine.attn_backend.coord.stats.hit_rate:.1%}"
            print(f"len={n:6d}  {mode:15s} {correct}/{len(args.depths)} correct{extra}")


if __name__ == "__main__":
    main()
