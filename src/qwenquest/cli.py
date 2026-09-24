"""``qwenquest`` command line.

* ``qwenquest info``      - shapes, memory and top-k budget for a model + config
* ``qwenquest demo``      - watch Quest pick pages on a tiny random model (CPU)
* ``qwenquest generate``  - run a real checkpoint in dense / quest / quest_hisparse mode
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from qwenquest._version import __version__
from qwenquest.config import (
    PRESETS,
    Qwen3MoeConfig,
    SparseConfig,
    count_parameters,
    parse_hisparse_config,
    tiny_config,
)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _gib(n: float) -> str:
    return f"{n / 2**30:,.2f} GiB"


def _quest_config(raw: str | None) -> SparseConfig:
    d: dict[str, Any] = json.loads(raw) if raw else {}
    d.setdefault("algorithm", "quest")
    return parse_hisparse_config(d)


# --------------------------------------------------------------------- info


def cmd_info(args: argparse.Namespace) -> int:
    cfg = Qwen3MoeConfig.from_pretrained(args.model) if args.model else PRESETS[args.preset]
    sc = _quest_config(args.hisparse_config)
    assert sc.quest_page_size is not None
    total, active = count_parameters(cfg)
    n = args.context_len
    kv_tok = cfg.kv_bytes_per_token(2)
    page = sc.quest_page_size
    bounds_tok = cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * 2 * 2 / page
    print(f"model            : {args.model or args.preset}")
    print(
        f"architecture     : {cfg.num_hidden_layers} layers, {cfg.num_attention_heads} q heads / "
        f"{cfg.num_key_value_heads} kv heads (GQA group {cfg.num_kv_groups}), head_dim {cfg.head_dim}"
    )
    print(
        f"MoE              : {cfg.num_experts} experts, top-{cfg.num_experts_per_tok}, "
        f"expert width {cfg.moe_intermediate_size}"
    )
    print(f"parameters       : {total / 1e9:.2f}B total, {active / 1e9:.2f}B active per token")
    print(
        f"rope             : theta={cfg.rope_theta:g}, scaling={cfg.rope_scaling}, "
        f"max_position_embeddings={cfg.max_position_embeddings}"
    )
    print(
        f"KV cache (bf16)  : {kv_tok / 1024:.0f} KiB per token -> {_gib(kv_tok * n)} for {n} tokens"
    )
    print()
    print(
        f"Quest            : top_k={sc.top_k}, quest_page_size={page}, applied at all "
        f"{cfg.num_hidden_layers} layers"
    )
    print(
        f"  per layer/step : {sc.top_k // page - 1} best complete pages + last {page} tokens "
        f"= {sc.top_k} tokens (dense while seq_len <= {sc.top_k})"
    )
    print(
        f"  scoring input  : q averaged over {cfg.num_kv_groups} heads/group -> "
        f"[{cfg.num_key_value_heads}, {cfg.head_dim}], scores summed over kv heads"
    )
    print(f"  density at {n:>6}: {min(1.0, sc.top_k / n):.2%} of the context is attended")
    print(
        f"  bounds (bf16)  : {bounds_tok:.0f} B per token ({bounds_tok / kv_tok:.2%} of KV) -> "
        f"{_gib(bounds_tok * n)} per {n}-token request"
    )
    print()
    buf = sc.device_buffer_size
    print(f"HiSparse         : device_buffer_size={buf} (+1 newest slot) per request")
    print(
        f"  device KV      : {_gib(kv_tok * (buf + 1))} per request (vs {_gib(kv_tok * n)} dense)"
    )
    print(f"  host KV        : {_gib(kv_tok * n)} per request (full history)")
    return 0


# --------------------------------------------------------------------- demo


def cmd_demo(args: argparse.Namespace) -> int:
    from qwenquest.engine import Engine, EngineConfig
    from qwenquest.model import Qwen3MoeForCausalLM, init_random_weights
    from qwenquest.quest import QuestTrace

    torch.manual_seed(args.seed)
    cfg = tiny_config(num_hidden_layers=args.layers)
    model = init_random_weights(Qwen3MoeForCausalLM(cfg), seed=args.seed, std=0.05)
    hc = {
        "algorithm": "quest",
        "top_k": args.top_k,
        "quest_page_size": args.page_size,
        "device_buffer_size": args.device_buffer_size or 2 * args.top_k,
    }
    gen = torch.Generator().manual_seed(args.seed)
    prompt = torch.randint(0, cfg.vocab_size, (args.prompt_len,), generator=gen).tolist()
    ctx = args.prompt_len + args.new_tokens + 8

    print(
        f"tiny Qwen3-MoE: {cfg.num_hidden_layers} layers, {cfg.num_attention_heads} q / "
        f"{cfg.num_key_value_heads} kv heads, head_dim {cfg.head_dim}; hisparse_config={hc}\n"
    )
    outputs = {}
    for mode in ("dense", "quest", "quest_hisparse"):
        eng = Engine(
            model,
            EngineConfig(
                attention_mode=mode, hisparse_config=hc, max_running_requests=1, max_context_len=ctx
            ),
        )
        if mode == "quest" and eng.quest is not None:
            eng.quest.trace = QuestTrace()
        outputs[mode] = eng.generate(
            [prompt], max_new_tokens=args.new_tokens, stop_token_ids=[], return_logits=True
        )[0]
        if mode == "quest" and eng.quest is not None and eng.quest.trace is not None:
            print("Quest decisions (layer 0; one line per decode step):")
            for rec in eng.quest.trace.for_layer(0):
                if rec.dense:
                    print(f"  seq_len={rec.seq_len:4d}  dense: positions 0..{rec.seq_len - 1}")
                    continue
                assert rec.page_scores is not None
                best = torch.topk(rec.page_scores, k=min(3, rec.page_scores.numel()))
                top = ", ".join(
                    f"p{int(i)}:{float(s):+.2f}"
                    for s, i in zip(best.values, best.indices, strict=True)
                )
                dup = rec.positions.numel() - rec.positions.unique().numel()
                note = f"  <- {dup} positions listed twice" if dup else ""
                print(
                    f"  seq_len={rec.seq_len:4d}  pages={rec.selected_pages}  + recent "
                    f"[{rec.recent_window[0]}, {rec.recent_window[1]})  best {top}{note}"
                )
            print(
                "  (a selected page that overlaps the recent window is attended twice, as "
                "upstream;\n   set sparse_extra_config avoid_recent_overlap=true to exclude it)\n"
            )
        if mode == "quest_hisparse":
            st = eng.attn_backend.coord.stats  # type: ignore[attr-defined]
            if st.hits + st.misses == 0:
                print(
                    "HiSparse hot buffer: every step took the fast path (the sequence fits "
                    f"in device_buffer_size={hc['device_buffer_size']})\n"
                )
            else:
                print(
                    f"HiSparse hot buffer: {st.hits} hits / {st.misses} misses "
                    f"(hit rate {st.hit_rate:.1%}) across all layers and steps\n"
                )

    q, h, d = (torch.stack(outputs[m].logits) for m in ("quest", "quest_hisparse", "dense"))
    print(
        f"quest vs quest_hisparse : max |logit diff| = {(q - h).abs().max():.3g}  (exact offload)"
    )
    print(
        f"quest vs dense          : max |logit diff| = {(q - d).abs().max():.3g}  (sparsity error)"
    )
    agree = sum(
        a == b
        for a, b in zip(outputs["quest"].output_ids, outputs["dense"].output_ids, strict=True)
    )
    print(
        f"greedy tokens equal to dense: {agree}/{len(outputs['dense'].output_ids)} "
        "(random weights give near-flat logits, so argmax flips easily)"
    )
    return 0


# ------------------------------------------------------------------ generate


def _encode_prompt(tok: Any, text: str, chat: bool) -> list[int]:
    if chat:
        enc: Any = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
        )
    else:
        enc = tok(text)
    if isinstance(enc, Mapping):  # BatchEncoding
        enc = enc["input_ids"]
    return [int(t) for t in enc]


def cmd_generate(args: argparse.Namespace) -> int:
    from qwenquest.engine import Engine, EngineConfig
    from qwenquest.model import load_qwen3_moe

    try:
        from transformers import AutoTokenizer
    except ImportError:  # pragma: no cover - optional dependency
        print("`generate` needs the tokenizer: pip install 'qwenquest[hf]'", file=sys.stderr)
        return 2

    tok = AutoTokenizer.from_pretrained(args.model)
    text = args.prompt
    if text is None:
        text = Path(args.prompt_file).read_text(encoding="utf-8")
    ids = _encode_prompt(tok, text, args.chat)

    t0 = time.time()
    model = load_qwen3_moe(args.model, device=args.device, dtype=_DTYPES[args.dtype])
    print(f"loaded in {time.time() - t0:.1f}s; prompt has {len(ids)} tokens", file=sys.stderr)
    eng = Engine(
        model,
        EngineConfig(
            attention_mode=args.mode,
            hisparse_config=args.hisparse_config,
            max_running_requests=1,
            max_context_len=args.max_context_len or len(ids) + args.max_new_tokens + 1,
        ),
    )
    t0 = time.time()
    out = eng.generate(
        [ids],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )[0]
    dt = time.time() - t0
    print(tok.decode(out.output_ids, skip_special_tokens=False))
    print(
        f"\n[{args.mode}] {len(out.output_ids)} tokens in {dt:.1f}s "
        f"({len(out.output_ids) / max(dt, 1e-9):.2f} tok/s), finish={out.finish_reason}",
        file=sys.stderr,
    )
    if args.mode == "quest_hisparse":
        st = eng.attn_backend.coord.stats  # type: ignore[attr-defined]
        print(
            f"HiSparse hit rate {st.hit_rate:.1%} ({st.hits} hits, {st.misses} misses)",
            file=sys.stderr,
        )
    return 0


# ---------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qwenquest", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--version", action="version", version=f"qwenquest {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="shapes, memory and Quest budget")
    src = info.add_mutually_exclusive_group()
    src.add_argument("--model", help="checkpoint directory (reads config.json)")
    src.add_argument("--preset", default="qwen3-30b-a3b-thinking-2507", choices=sorted(PRESETS))
    info.add_argument("--hisparse-config", default=None, help="JSON, e.g. '{\"top_k\": 2048}'")
    info.add_argument("--context-len", type=int, default=131_072)
    info.set_defaults(func=cmd_info)

    demo = sub.add_parser("demo", help="Quest top-k on a tiny random model (CPU, seconds)")
    demo.add_argument("--prompt-len", type=int, default=150)
    demo.add_argument("--new-tokens", type=int, default=12)
    demo.add_argument("--top-k", type=int, default=64)
    demo.add_argument("--page-size", type=int, default=16)
    demo.add_argument("--device-buffer-size", type=int, default=0)
    demo.add_argument("--layers", type=int, default=2)
    demo.add_argument("--seed", type=int, default=0)
    demo.set_defaults(func=cmd_demo)

    g = sub.add_parser("generate", help="run a real Qwen3-MoE checkpoint")
    g.add_argument("--model", required=True, help="HF checkpoint directory")
    prompt = g.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file")
    g.add_argument("--mode", default="quest", choices=["dense", "quest", "quest_hisparse"])
    g.add_argument("--hisparse-config", default='{"algorithm": "quest", "top_k": 2048}')
    g.add_argument("--chat", action="store_true", help="wrap the prompt with the chat template")
    g.add_argument("--max-new-tokens", type=int, default=256)
    g.add_argument("--max-context-len", type=int, default=0)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--dtype", default="bfloat16", choices=sorted(_DTYPES))
    g.set_defaults(func=cmd_generate)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
