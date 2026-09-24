# 与上游 SGLang 代码的对应关系

上游固定版本：<https://github.com/sgl-project/sglang> 分支 `hisparse_quest`，
commit `e568f8a362b4ae3a597b28b75716309814aa33fb`（2026-05-06，两次提交：
`b3b36671 hisparse for quest`、`e568f8a3 tests for hisparse quest`）。
该常量也写在 `qwenquest.UPSTREAM_COMMIT` 里；升级上游版本时请同步更新本表（见 docs/versioning.md）。

行号均指上述 commit。

## Quest 算法

| 上游 | 本仓库 | 说明 |
|---|---|---|
| `srt/mem_cache/sparsity/algorithms/quest_algorithm.py:71` `QuestAlgorithm` | `src/qwenquest/quest.py` `QuestAlgorithm` | 方法名、参数、存储布局、哨兵值一致 |
| `:134` `init_storage` | `init_storage` | 同 |
| `:234` `update_prefill_representations` | `update_prefill_representations` [Q2] | 同（纯 PyTorch 路径） |
| `:294` `update_prefill_representations_fused` + `quest_prefill_bounds_kernel.py` | —— | Triton 融合版，只为性能；语义等同于逐层调用上面的函数 |
| `:349` `update_decode_representations` | `update_decode_representations` [Q3] | 同 |
| `:378` `update_decode_representations_fused` + `quest_decode_bounds_kernel.py` | —— | Triton 融合版 |
| `:413` `maybe_finalize_decode_representations` | 同名 [Q3] | 同 |
| `:461` `prepare_step` | `prepare_step` [Q4] | 返回 `StepLayout` 而非写入预分配 buffer（上游为了 CUDA graph） |
| `:536` `retrieve_topk`（GQA 平均 + 打分 + top-k + 布局） | `group_queries` [Q5]、`page_scores` [Q5]、`retrieve_topk` [Q6] | 拆成三个函数便于阅读和测试；全短序列时跳过打分（结果相同） |
| `quest_score_kernel.py` | `page_scores` | Triton 打分 kernel 的纯 PyTorch 等价（上游保留的 PyTorch 参考路径） |
| `:518` `invalidate_request` | `invalidate_request` [Q8] | 同 |
| `srt/mem_cache/sparsity/factory.py:61` `_parse_sparse_config` | `src/qwenquest/config.py` `parse_hisparse_config` | 默认值与校验一致（`quest_page_size` 默认 64，`top_k % quest_page_size == 0`） |
| `sparse_coordinator.py` `SparseConfig` | `config.py` `SparseConfig` | 字段一致，多一个只读属性 `avoid_recent_overlap` |

## 注意力后端

| 上游 | 本仓库 | 说明 |
|---|---|---|
| `srt/layers/attention/flashinfer_backend.py`（dense） | `attention/dense.py` `DenseBackend` | Mode 1 |
| `flashinfer_quest_backend.py:147` `FlashInferQuestDecodeBackend` | `attention/quest_only.py` `QuestOnlyBackend` | Mode 2 |
| `:371` `init_forward_metadata` / `:408` `_do_quest_decode_step_update` | `init_forward_metadata` / `_do_quest_decode_step_update` | 同；逐层循环代替融合 kernel |
| `:454` `forward_extend` / `:481` `_update_quest_for_extend` | 同名 | 最后一层 extend 后构建包围盒 |
| `:514` `forward_decode`（retrieve → `quest_only_gather_scatter` → FlashInfer） | `forward_decode` [Q6][Q7] | FlashInfer plan/run、CUDA graph 相关代码省略 |
| `flashinfer_hisparse_backend.py:66` `FlashInferHiSparseDecodeBackend` | `attention/quest_hisparse.py` `QuestHiSparseBackend` | Mode 3 |
| `:486` `forward_decode`（set_kv → retrieve → swap_in → scatter-pack → FlashInfer） | `forward_decode` [Q6][H3][Q7] | 同顺序 |
| `quest_scatter_pack_kernel.py` | —— | 只为把变长结果紧凑写入 `kv_indices`；这里直接按 `actual_lens` 切片 |
| `srt/layers/radix_attention.py:138` | `model/qwen3_moe.py` `RadixAttention.forward` [Q1] | 模型与后端的唯一接口 |
| `srt/models/qwen3_moe.py:633-695`（q/k norm → RoPE → `self.attn`） | `model/qwen3_moe.py` `Qwen3MoeAttention.forward` | 同 |

## HiSparse

| 上游 | 本仓库 | 说明 |
|---|---|---|
| `srt/managers/hisparse_coordinator.py` `HiSparseCoordinator(mode="quest")` | `hisparse/coordinator.py` `HiSparseCoordinator` | 只实现 Quest 模式 |
| `:232` `admit_request_into_staging` + `:290` `_update_quest_prefill_representations` + `:357` `alloc_device_buffer` + `:397` `collect_ready_reqs` | `admit_request` [H1] | 同步执行；上游用 staging stream / bounds stream 异步，并提前释放 prefill 的逻辑槽位 |
| `:557` `map_last_loc_to_buffer` → `:583` `_eager_backup_previous_token` → `:668` `_update_quest_decode_representations` | `prepare_decode_step` [H2]（+ `write_new_token`） | 同：先更新 Quest running bounds，再备份到 host |
| `:897` `swap_in_selected_pages` + `jit_kernel/csrc/hisparse.cuh:88` `load_cache_to_device_buffer_kernel` | `swap_in_selected_pages` / `_swap_in_long` [H3] | 快路径、保留槽位、命中检测、LRU 淘汰顺序与 kernel 一致；**先去重**（见下） |
| `:853` `request_finished` | `request_finished` [H4] | 同，含 `quest.invalidate_request` |
| `srt/mem_cache/hisparse_memory_pool.py` `HiSparseMHATokenToKVPool` / allocator | `hisparse/coordinator.py`（每请求固定热缓冲）+ `memory.py` | 上游热缓冲从共享设备池按页增长分配；这里每请求预留 `device_buffer_size + 1` 行，语义相同 |

## 有意的差异（都不改变 Quest 选中哪些位置）

1. **Triton / CUDA / FlashInfer → 纯 PyTorch**，并去掉 CUDA graph、多 stream、plan/run 分离。
   这些只影响速度；数学上与上游的 PyTorch 参考路径一致。
2. **swap-in 先去重**：上游 kernel 假设 top-k 位置互不相同，而 Quest 的布局可能重复（walkthrough 6.1、6.2）。
   这里保证 Mode 3 与 Mode 2 逐位一致。
3. **Mode 2 在请求结束时也重置 Quest 状态**：上游 Mode 2 没有结束钩子（walkthrough 第 9 节）。
4. **可选 `avoid_recent_overlap`**（默认关闭 = 与上游一致）。
5. **引擎是最小实现**：一次 prefill 一个请求、无 prefix cache、无 chunked prefill，与上游 Quest 模式的
   约束一致（上游强制 `--disable-radix-cache` 并关闭 chunked prefill）。

## 上游测试的移植

`test/registered/unit/managers/test_quest_unit.py::TestQuestAlgorithm` 的 13 个用例已移植到
`tests/test_quest_algorithm.py`（CPU 上运行）；需要真实 `HiSparseCoordinator` / FlashInfer 的用例由
`tests/test_hisparse.py` 与 `tests/test_engine.py` 中的等价性质测试代替。
