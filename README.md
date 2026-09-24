# QwenQuest

[![CI](https://github.com/wengyinqi/QwenQuest/actions/workflows/ci.yml/badge.svg)](https://github.com/wengyinqi/QwenQuest/actions/workflows/ci.yml)

**HiSparse 论文中「Qwen3-30B-A3B + Quest」方案的可读 baseline**：用纯 PyTorch 复现
SGLang `hisparse_quest` 分支（commit `e568f8a3`）里 Quest 接入 Qwen3-MoE、逐层选出 top-k token、
以及 HiSparse 分层 KV 缓存的完整数据流。CPU 上即可运行和测试。代码中用 `[Q1]`…`[Q8]`、`[H1]`…`[H4]`
标出每个关键步骤，与 [docs/quest_walkthrough.md](docs/quest_walkthrough.md) 一一对应。

*A readable PyTorch baseline of the Quest sparse-decoding path HiSparse uses for Qwen3-30B-A3B
(SGLang `hisparse_quest` @ `e568f8a3`), with three decode modes, exact HiSparse offload, CPU tests and CI.*

## 一句话看懂

Qwen3 的每个注意力层把（q_norm/k_norm + RoPE 之后的）`q, k, v` 交给 attention backend；
Quest 只替换 **decode** 时的 backend：

1. 每 64 个 token 为一页，维护页内 key 的逐通道 `min / max`（bf16）；
2. 当前 query 在每个 GQA 组内取平均（32 头 → 4 头），对每页算上界分数
   `Σ max(q·kmin, q·kmax)`（4 个 KV 头求和，所有头共享一组选择）；
3. 取分数最高的 **31 个完整页 + 最近 64 个位置 = 2048 个 token**，序列不超过 2048 时直接稠密；
4. 只对这 2048 个 token 做注意力。48 层全部如此；prefill 始终稠密。

## 三种模式

| `attention_mode` | 对应 SGLang 后端 | decode 读取的 KV |
|---|---|---|
| `dense` | `flashinfer` | 全部历史 token |
| `quest` | `flashinfer_quest` | Quest top-2048（全量 KV 在 GPU） |
| `quest_hisparse` | `flashinfer_hisparse` + `--enable-hisparse` | Quest top-2048；全量 KV 在 host，GPU 只放每请求 4096+1 个 token 的 LRU 热缓冲 |

`quest` 与 `quest_hisparse` 的输出**逐位相同**（有测试保证）：HiSparse 只改变 KV 放在哪里，不改变结果。

> 定位：这是用来**读懂和验证**算法的参考实现（逐请求循环、无 Triton/CUDA graph），
> 精度对齐上游，速度不对齐；测吞吐请用 SGLang 上游实现。

## 快速开始

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # 有 GPU 则装对应 CUDA 版
pip install -e ".[dev]"

python examples/01_topk_by_hand.py   # 合成数据上逐步演示：包围盒、打分、上界、选页
qwenquest demo                       # 随机小模型：每步每层选了哪些页、HiSparse 命中率、三种模式对比
qwenquest info --context-len 131072  # Qwen3-30B-A3B 的参数量、KV/包围盒/热缓冲显存
pytest                               # 全部测试（CPU，约 10 秒）
```

真实模型（Hugging Face 权重目录，bf16 约需 61 GB 显存）：

```bash
qwenquest generate --model /path/to/Qwen3-30B-A3B-Thinking-2507 --chat \
    --prompt "..." --mode quest_hisparse \
    --hisparse-config '{"algorithm": "quest", "top_k": 2048, "quest_page_size": 64, "device_buffer_size": 4096}'

python examples/03_passkey.py /path/to/Qwen3-30B-A3B-Thinking-2507 --lengths 8192 32768   # 长上下文 passkey
```

Python API：

```python
import torch
from qwenquest import Engine, EngineConfig, QuestTrace, load_qwen3_moe

model = load_qwen3_moe("/path/to/Qwen3-30B-A3B-Thinking-2507", device="cuda", dtype=torch.bfloat16)
engine = Engine(
    model,
    EngineConfig(
        attention_mode="quest",  # 或 "dense" / "quest_hisparse"
        hisparse_config={"top_k": 2048, "quest_page_size": 64},
        max_running_requests=1,
        max_context_len=32768,  # 决定 GPU 上 KV 池的大小
    ),
)
engine.quest.trace = QuestTrace()  # 可选：记录每层每步选了哪些页
out = engine.generate([input_ids], max_new_tokens=256)[0]
```

## 代码结构（建议阅读顺序）

```
src/qwenquest/
├── model/qwen3_moe.py        Qwen3-MoE；RadixAttention 是 Quest 唯一的接入点 [Q1]
├── quest.py                  QuestAlgorithm：包围盒 [Q2][Q3]、每步布局 [Q4]、打分 [Q5]、top-k [Q6]、重置 [Q8]
├── attention/
│   ├── base.py / ops.py      后端接口；稠密 prefill；按给定行做 decode 注意力
│   ├── dense.py              Mode 1
│   ├── quest_only.py         Mode 2：位置 → req_to_token → KV 行 [Q7]
│   └── quest_hisparse.py     Mode 3：位置 → swap-in → 热缓冲行 [Q7]
├── hisparse/coordinator.py   host 池 + LRU 热缓冲 + 立即备份 + swap-in [H1]-[H4]
├── engine.py                 prefill → 批量 decode → 结束 的最小服务循环
├── memory.py                 ReqToTokenPool / MHATokenToKVPool（与 SGLang 相同的两级寻址）
├── model/loader.py           直接加载官方 safetensors（meta 设备 + assign，不重复占显存）
├── config.py                 Qwen3MoeConfig、--hisparse-config 解析（与上游默认值/校验一致）
└── cli.py                    qwenquest info | demo | generate
```

## 已验证的性质

* **与 transformers 数值对齐**：随机初始化的 Qwen3-MoE（含 YaRN、稠密 MLP 层）在 float32 下
  prefill 与逐步 decode 的 logits 误差约 1e-7（`tests/test_hf_parity.py`）。
* **Quest 上界性质**：每页分数 ≥ 页内任意 token 的 `Σ_h q̄_h·k_h`（`tests/test_quest_algorithm.py`）。
* **上游用例**：`hisparse_quest` 分支的 13 个 `TestQuestAlgorithm` 用例全部移植并通过。
* **精确性**：预算覆盖整段序列时 `quest == dense`；`quest_hisparse == quest` 逐位相同；
  每层每步的注意力输出都等于对 trace 中位置手工计算的结果（`tests/test_engine.py`）。
* 参数量：30.53B 总参 / 3.35B 激活，与官方模型卡一致。

## 与上游的差异和需要注意的地方

* 选择逻辑与上游一致（默认）。上游有一个特性：最近 64 个位置的窗口可能与被选中的最新完整页重叠，
  重叠位置会被 softmax 计两次；可以用 `"avoid_recent_overlap": true` 关闭（见 walkthrough 6.1）。
* 读上游 HiSparse swap-in kernel 的代码可以推断：出现上述重复位置时，它的 miss 计数会少算，可能读到旧 KV
  （walkthrough 6.2，未在 GPU 上验证）。本仓库的 swap-in 先去重，保证精确。
* 其余差异（Triton/CUDA → PyTorch、同步执行、固定大小热缓冲等）见
  [docs/upstream_mapping.md](docs/upstream_mapping.md)。

## 版本管理与 CI

* SemVer，版本号唯一来源 `src/qwenquest/_version.py`，改动记录在 [CHANGELOG.md](CHANGELOG.md)；
  `scripts/release.py bump|check|notes` 负责升级与校验。
* GitHub Actions：ruff、mypy、Python 3.10–3.13 测试矩阵、最低依赖版本测试、构建与 wheel 安装检查、
  版本一致性检查；推送 `vX.Y.Z` tag 自动发布 GitHub Release。详见 [docs/versioning.md](docs/versioning.md)。

## 参考

* HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management, arXiv:2608.07009
* Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference, ICML 2024, arXiv:2406.10774
* SGLang `hisparse_quest` 分支：<https://github.com/sgl-project/sglang/tree/hisparse_quest>

Apache-2.0。部分逻辑源自 SGLang（Apache-2.0），见 [NOTICE](NOTICE)。
