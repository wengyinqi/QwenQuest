# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(see [docs/versioning.md](docs/versioning.md)).

## [Unreleased]

## [0.1.0] - 2026-09-24

First baseline, mirroring SGLang `hisparse_quest` @ `e568f8a3`
(the Quest path of HiSparse, arXiv:2608.07009).

### Added
- `Qwen3MoeForCausalLM`: plain-PyTorch Qwen3-MoE (Qwen3-30B-A3B and the 2507
  Instruct/Thinking refreshes) with q/k-norm, RoPE/YaRN, 128-expert top-8 MoE;
  loads the official safetensors directly and matches `transformers` to ~1e-7
  in float32.
- `QuestAlgorithm`: HiSparse's Quest selector - bf16 per-page key min/max
  bounds, running bounds for the page being decoded, GQA-averaged query,
  criticality summed over KV heads, `top_k/P - 1` pages + a `P`-token recent
  window, dense below `top_k`; applied at every layer. Optional
  `avoid_recent_overlap` removes the page that overlaps the recent window.
- Three decode modes behind one attention seam (`RadixAttention` ->
  backend): `dense` (flashinfer), `quest` (flashinfer_quest) and
  `quest_hisparse` (flashinfer_hisparse) with a reference `HiSparseCoordinator`
  (host pool, per-request LRU hot buffer, eager backup, exact swap-in).
- `parse_hisparse_config` with upstream defaults (`top_k=2048`,
  `quest_page_size=64`, `device_buffer_size=2*top_k`) and validation.
- `Engine` serving loop, `QuestTrace` recorder, `qwenquest info|demo|generate` CLI.
- Tests (upstream Quest unit tests ported to CPU, Quest upper-bound property,
  exact-offload and HF-parity checks), GitHub Actions CI and tag-driven releases.

[Unreleased]: https://github.com/wengyinqi/QwenQuest/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/wengyinqi/QwenQuest/releases/tag/v0.1.0
