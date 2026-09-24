# Quest 是怎样接入 Qwen3-30B-A3B 并选出 top-k 的

> 对应上游：SGLang `hisparse_quest` 分支 @ `e568f8a3`（HiSparse 作者提交的 “hisparse for quest”），
> 即 HiSparse 论文（arXiv:2608.07009）里 **Qwen3-30B-A3B-Thinking-2507 + Quest** 这组实验所用的实现。
>
> 代码中的 `[Q1]`…`[Q8]`、`[H1]`…`[H4]` 标签与本文小节一一对应：
> `grep -rn "\[Q6\]" src/` 就能跳到对应代码。

---

## 0. 一张图看完一次 decode

```
一次 decode step（batch 里每个请求生成 1 个 token）
│
├─ engine: attn_backend.init_forward_metadata(batch)            —— 每步 1 次，所有层共享
│    [Q3] 把「上一步生成的 token」的 K 并入 running min/max；
│         凑满 64 个 token 的页被「定稿」写入 page_k_bounds
│    [Q4] quest.prepare_step(seq_lens)：最近窗口、是否短序列、actual_lens
│
├─ model: 48 层，每层
│    q,k,v = q_proj/k_proj/v_proj(h) → q_norm / k_norm → RoPE
│    [Q1] RadixAttention → attn_backend.forward_decode(q, k, v)  —— Quest 的唯一接入点
│         ├─ 写入当前 token 的 K/V
│         ├─ [Q5] q̄ = 每组 8 个 query 头取平均（32 头 → 4 个 KV 头）
│         │       score[p] = Σ_h Σ_d max(q̄·kmin[p], q̄·kmax[p])     （每页一个标量）
│         ├─ [Q6] 取分数最高的 31 个完整页（×64）+ 最近 64 个位置 = 2048 个位置
│         │       seq_len ≤ 2048 时直接取 0..seq_len-1（稠密）
│         ├─ [Q7] 位置 → K/V 行：Mode 2 查 req_to_token；Mode 3 走 [H3] swap-in
│         └─      只对这 2048 行做注意力（scale = 1/√128）
│
└─ 采样下一个 token；请求结束时 [Q8] invalidate_request
```

---

## 1. HiSparse 里的「Qwen + Quest」是什么

| 项 | 取值 | 上游出处（@e568f8a3） |
|---|---|---|
| 模型 | Qwen3-30B-A3B（论文用 Thinking-2507）：48 层，32 个 Q 头 / 4 个 KV 头（GQA 组大小 8），head_dim 128，128 专家 top-8，30.5B 总参 / 3.3B 激活 | `config.json` |
| 选择器 | Quest，training-free，**48 层全部使用**（不像原始 Quest 那样保留前两层稠密） | `model_runner.py:727` `init_storage(start_layer, end_layer)` |
| `top_k` | 2048 个 token（每请求、每层、每步） | `factory.py` 默认值 |
| `quest_page_size` | 64 | `factory.py:96` |
| 选择粒度 | 每请求每层**一组**页，32 个头共享 | `quest_algorithm.py::retrieve_topk` |
| KV cache | 必须 bf16 | `server_args.py` |
| `device_buffer_size` | 默认 `2 * top_k = 4096`（HiSparse 每请求 GPU 热缓冲） | `factory.py` |

上游有三种运行方式（本仓库用 `attention_mode` 复现）：

| 本仓库 | 上游启动参数 | decode 读哪些 KV |
|---|---|---|
| `dense` | `--attention-backend flashinfer` | 全部历史 token（全在 GPU） |
| `quest` | `--attention-backend flashinfer_quest --hisparse-config '{"algorithm":"quest"}'` | Quest 选出的 2048 个（全量 KV 仍在 GPU） |
| `quest_hisparse` | `--enable-hisparse --hisparse-config '{"algorithm":"quest"}' --prefill-attention-backend flashinfer --decode-attention-backend flashinfer_hisparse` | 同上 2048 个，但全量 KV 在 host，GPU 只放每请求 4096+1 个 token 的热缓冲 |

`quest` 与 `quest_hisparse` 的**选择逻辑完全相同**，区别只在 KV 放在哪里；上游把 Mode 2 称为
「is sparsity itself the win, or is it the offloading?」的归因基线。

---

## 2. [Q1] 接入点：模型本身完全不知道 Quest

`src/qwenquest/model/qwen3_moe.py`：

```python
class Qwen3MoeAttention(nn.Module):
    def forward(self, positions, hidden_states, batch):
        q = self.q_proj(hidden_states).view(t, 32, 128)
        k = self.k_proj(hidden_states).view(t, 4, 128)
        v = self.v_proj(hidden_states).view(t, 4, 128)
        q = self.q_norm(q)  # Qwen3：每个头先做 RMSNorm
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v, batch)  # RadixAttention
        return self.o_proj(o)


class RadixAttention(nn.Module):
    def forward(self, q, k, v, batch):
        return batch.attn_backend.forward(q, k, v, self, batch)  # [Q1]
```

和上游 `qwen3_moe.py:695`（`self.attn(q, k, v, ...)`）→ `radix_attention.py:138`
（`forward_batch.attn_backend.forward(...)`）是同一条路径。要点：

* **接入 = 换 attention backend**。embedding、MoE、RMSNorm、lm_head 一行不改。
* Quest 看到的 `q`、`k` 是**做完 q_norm/k_norm 和 RoPE 之后**的，也就是 KV cache 里真正存的东西；
  所以页的包围盒是在「带位置编码的 key」上算的。
* Prefill（EXTEND）在所有模式下都是**稠密**的，Quest 只改变 decode。
  这可以用测试验证：`tests/test_engine.py::test_prefill_logits_are_identical_across_modes`。

---

## 3. [Q2] [Q3] 每页 key 的包围盒（page bounds）

`src/qwenquest/quest.py::QuestAlgorithm`

**存储**（bf16）：

```
page_k_bounds[layer, req, page, kv_head, d, 0] = 该页 64 个 token 的 key 在 (kv_head, d) 上的最小值
page_k_bounds[layer, req, page, kv_head, d, 1] = 最大值
形状: [48, max_reqs, ceil(max_context_len / 64), 4, 128, 2]
```

* 页是**逻辑页**：第 `p` 页 = 位置 `[64p, 64p+64)`，与 KV 在物理内存中放在哪里无关。
* 还没有包围盒的页存哨兵值 `min = +bf16max, max = -bf16max`，这样打分公式自然给出极小值 / `-inf`，
  不需要单独的 valid 掩码（`page_valid` 属性只是给测试看的）。
* 正在被 decode 填充的那一页不进 `page_k_bounds`，而是累积在 `running_k_min / running_k_max`，
  配合两个计数器 `running_token_count`（这一页已有几个 token）和 `running_page_idx`（这是第几页）。
* 显存开销：48 层 × 4 头 × 128 × 2(min,max) × 2 B / 64 = **1536 B/token**，是 KV（96 KiB/token）的 1.56%。

**[Q2] prefill 之后**（`update_prefill_representations`，对每一层调用一次）：

```
prefill_len = 200 为例（页大小 64）：
  完整页 0,1,2  → 直接写入 page_k_bounds
  剩下 8 个 token → 写入 running_k_min/max，running_token_count = 8，running_page_idx = 3
```

调用时机：Mode 2 在最后一层的 `forward_extend` 里（此时所有层的 K 都已写入）；
Mode 3 在请求准入 HiSparse 时（`HiSparseCoordinator.admit_request`，上游放在独立 CUDA stream 上）。

**[Q3] 每个 decode step 开始时**（`update_decode_representations` + `maybe_finalize_decode_representations`）：

1. 取出「上一步生成的 token」（位置 `seq_len - 2`）的 K，对 running min/max 做逐元素 min/max。
2. 如果这一页因此凑满 64 个 token：把 running 拷进 `page_k_bounds[:, req, running_page_idx]`（覆盖哨兵
   → 该页从此可以被选中），running 重置为哨兵，`running_page_idx += 1`。
3. `running_token_count = (count + 1) % 64`。

细节：

* prefill 之后的**第一个** decode step 要跳过（上游 `_skip_first_decode_update / _skip_first_backup`），
  因为「上一个 token」就是 prompt 的最后一个 token，prefill 已经算进去了。
* 于是第 `p` 页在它的第 64 个 token 被 decode 出来之后的**下一步**才能被选中；
  这之前它的内容由下面的「最近窗口」覆盖。

---

## 4. [Q4] 每步只算一次的布局：`prepare_step`

与层无关的部分每步只算一次（上游缓存在 `_step_*_buf` 里，避免 48 层各算一遍）：

| 量 | 公式 | 含义 |
|---|---|---|
| `recent_positions` | `[seq_len-64, seq_len)` | 最近窗口，永远被读 |
| `is_short` | `seq_len <= top_k` | 短序列直接稠密 |
| `short_layout` | `0, 1, …, seq_len-1` | 稠密布局 |
| `actual_lens` | `min(seq_len, top_k)` | 每个请求真正读多少个位置 |

这里的 `seq_len` 已经包含当前正在 decode 的 token（它的 K/V 在本步 forward 中写入位置 `seq_len-1`）。

---

## 5. [Q5] 打分：包围盒给出的上界（criticality）

`QuestAlgorithm.group_queries` + `QuestAlgorithm.page_scores`：

1. **GQA 归约**：Qwen3-30B-A3B 有 32 个 query 头、4 个 KV 头。第 `h` 个 query 头读第 `h // 8` 个 KV 头，
   所以把每个 KV 头对应的 8 个 query 头取平均：`q̄[h] = mean(q[8h : 8h+8])`，得到 `[4, 128]`
   （上游注释称之为 MQA/GQA 的近似）。
2. **每页分数**：

   ```
   score[p] = Σ_{h=0..3} Σ_{d=0..127}  max( q̄[h,d]·kmin[p,h,d],  q̄[h,d]·kmax[p,h,d] )
            = Σ_{h,d}  ( q̄[h,d] ≥ 0 ? q̄[h,d]·kmax[p,h,d] : q̄[h,d]·kmin[p,h,d] )
   ```

3. **为什么是上界**：页内任一 token `t` 满足 `kmin ≤ k_t ≤ kmax`（逐元素），而 `x ↦ q·x` 对每个分量单调，
   所以 `q̄[h,d]·k_t[h,d] ≤ max(q̄·kmin, q̄·kmax)`；求和得到
   `Σ_h q̄_h·k_{t,h} ≤ score[p]` 对页内**每个** token 都成立。
   分数高 ⇔ 这一页「可能」有 token 和当前 query 很对齐。
   （`tests/test_quest_algorithm.py::test_score_upper_bounds_every_token_of_the_page` 验证了这一点。）
4. 分数在 4 个 KV 头上**求和**，因此一个请求在一层只有一组页，32 个头共享；
   这与原始 Quest 每个头各选各的不同（见第 10 节）。

---

## 6. [Q6] top-k 布局：31 页 + 最近 64 个位置

`QuestAlgorithm.retrieve_topk` 返回 `(token_positions [bs, 2048], actual_lens [bs])`：

| 情况 | slot `[0, 1984)` | slot `[1984, 2048)` |
|---|---|---|
| 长序列 `seq_len > 2048` | 分数最高的 `2048/64 - 1 = 31` 个**完整页**展开成位置（按分数排序） | 最近窗口 `[seq_len-64, seq_len)` |
| 短序列 `seq_len ≤ 2048` | `0, 1, …, seq_len-1`（稠密），其余 slot 为填充，靠 `actual_lens` 忽略 | |

候选页规则：

* 只有**完整页**（`p < seq_len // 64`）是候选；更靠后的页即使残留旧请求的包围盒也会被置为 `-inf`
  （上游称为防御性掩码）。
* 完整但还没定稿的页（哨兵）分数为极小值，永远排不进前 31。
* 正在填充的最后一页不打分，由最近窗口覆盖——Quest 的包围盒只对完整页有意义，
  而刚生成的上下文对 decode 最重要。

举例：`seq_len = 10000` → 完整页 0…155（156 页），第 156 页只有 16 个 token（9984…9999）。
最近窗口 = `[9936, 10000)`，Quest 在 156 页里挑 31 页。整个注意力只读 2048 / 10000 ≈ 20% 的上下文；
128K 上下文时只读 1.56%。

### 6.1 最近窗口与最新完整页重叠（上游行为；可选修正）

最近窗口不按页对齐。当 `seq_len % 64 = r ≠ 0` 时，窗口 `[seq_len-64, seq_len)` 覆盖了最新完整页
（第 `seq_len//64 - 1` 页）的后 `64 - r` 个位置。如果 Quest 恰好也选中了这一页（它很新，常常分数高），
这些位置就在 2048 个 slot 里出现两次。上游 FlashInfer 按 `kv_indices` 原样读取，softmax 会把它们算两次，
相当于给这些 token 加了权重。

* 默认行为与上游一致（为了做**忠实的 baseline**）。`qwenquest demo` 会打印每步重复了多少个位置。
* 在 `--hisparse-config` 的 JSON 里加 `"avoid_recent_overlap": true`（上游会把未知键放进
  `sparse_extra_config`，本仓库从那里读取）即可把这一页从候选中去掉，保证 2048 个位置互不相同：
  `tests/test_quest_algorithm.py::test_recent_window_overlap_upstream_vs_avoid`。

### 6.2 重复位置对上游 HiSparse swap-in kernel 的影响（读代码推断，未在 GPU 上验证）

上游 `hisparse.cuh::load_cache_to_device_buffer_kernel` 默认 top-k 里的 token 互不相同：

1. 用哈希表把 token → **一个** top-k 下标（`atomicCAS`，第 201 行）；重复的另一个下标得不到命中标记，
   被当成 miss，从 host 再拷一份到另一个槽位——热缓冲里于是有两个槽位缓存同一个 token。
2. 之后某一步这个 token 再被选中时，两个槽位都算 hit，`s_total_hits` 比真正被命中的 top-k 条目多 1。
3. 第 339 行用 `total_misses = NUM_TOP_K - s_total_hits - s_newest_hit` 推算 miss 数，因此少算 1，
   拷贝循环会漏掉**按下标顺序的最后一个 miss**。在长序列布局里，它通常是最近窗口中的位置 `seq_len-2`
   （上一步的 token，刚从保留槽位备份到 host，不在 LRU 缓冲里）。于是那一层读到的是被淘汰 token 的旧 KV。

本仓库的 `HiSparseCoordinator._swap_in_long` 先去重再做命中检测，所以 Mode 3 与 Mode 2 逐位相同
（`tests/test_engine.py::test_hisparse_offload_is_exact`）。若要在上游确认，可在长输出上对比
`flashinfer_quest` 与 `flashinfer_hisparse` 的 logits；使用 `avoid_recent_overlap` 后不存在重复，这个问题也随之消失。

---

## 7. [Q7] 位置 → K/V 行 → 稀疏注意力

`retrieve_topk` 给出的是**逻辑位置**，注意力需要的是 KV 的**物理行**：

* **Mode 2**（`attention/quest_only.py`）：`rows = req_to_token[req, positions]`，读完整的 GPU KV 池。
  上游用 Triton kernel `quest_only_gather_scatter` 把它们紧凑地写进 FlashInfer 的 `kv_indices`
  （`page_size = 1`，每请求长度 `actual_lens`）。
* **Mode 3**（`attention/quest_hisparse.py`）：`rows = coord.swap_in_selected_pages(...)`，
  保证这些位置都在 GPU 热缓冲里，返回热缓冲的行号（见第 8 节）。

然后两种模式都调用 `attention/ops.py::decode_attention(q, k[rows], v[rows], 1/√128)`：
对**恰好这些行**做 softmax（有重复就重复计入），与 FlashInfer decode 读 `kv_indices` 的语义一致。
`tests/test_engine.py::test_attention_reads_exactly_the_selected_tokens` 在每一层、每一步都手工重算并比对。

---

## 8. HiSparse 分层 KV：[H1]–[H4]

`src/qwenquest/hisparse/coordinator.py`（上游 `hisparse_coordinator.py` + `hisparse.cuh`）。

| 层级 | 放什么 | 以 128K 上下文为例（bf16） |
|---|---|---|
| host（上游是 CPU pinned memory） | 每个请求**全部**历史 KV | 12 GiB / 请求 |
| device 热缓冲 | 每请求 `device_buffer_size = 4096` 个 LRU 槽位 + 1 个「最新 token」保留槽位 | 0.38 GiB / 请求 |

* **[H1] 准入**（prefill 结束后）：全部 prefill KV 备份到 host；位置 `[0, 4096)` 留在热缓冲
  （槽位 s 缓存位置 s）；趁 K 还在 GPU 上算 Quest 的 prefill 包围盒 [Q2]；然后释放 prefill 占用的 GPU KV。
* **[H2] 每步开始**（上游 `map_last_loc_to_buffer` → `_eager_backup_previous_token`）：
  上一步的 token 还在它的槽位里，先把它的 K 并入 Quest running bounds [Q3]，再把它的 K/V 备份到 host。
* **[H3] swap-in**（每层）：
  * `seq_len ≤ 4096`：快路径，整段序列都在热缓冲里，槽位 = 位置。
  * 否则：当前 token → 保留槽位；其余位置先做**命中检测**，未命中的从 host 拷进
    **最久未用**的槽位；最后按 `[未用到的旧槽位 | 刚拷入的 | 命中的]` 重排 LRU 顺序（与 kernel 回写顺序一致）。
  * `SwapInStats` 统计命中率——Quest 的选择在相邻步之间高度重合，这正是 HiSparse 用 LRU 获益的原因。
* **[H4] 结束**：释放 host 行、重置表、调用 `quest.invalidate_request` [Q8]。

---

## 9. [Q8] 请求结束

包围盒按 `(req_pool_idx, page)` 寻址，槽位会被下一个请求复用，所以结束时要把该槽位的所有页、
running 缓冲和计数器恢复成哨兵（`invalidate_request`）。上游 Mode 2 没有结束钩子，依赖下次 prefill 覆盖；
若新 prompt 长度恰为 64 的整数倍，running 缓冲不会被覆盖，第一张 decode 页会混入上一个请求的 min/max。
本仓库在两种模式下都会重置（`tests/test_engine.py::test_request_slots_are_reused_cleanly`）。

---

## 10. 与原始 Quest 的差异

| | 原始 Quest（论文 / 官方实现，数值以论文为准） | HiSparse 的 Quest（本仓库复现） |
|---|---|---|
| 目标模型 | MHA 模型（LongChat、Llama-2 系列） | GQA 的 Qwen3-30B-A3B（32Q / 4KV） |
| 页大小 | 16 | 64 |
| 选择粒度 | 每个注意力头各自选页 | 每请求每层一组，所有头共享；query 在组内取平均，分数在 KV 头上求和 |
| 哪些层 | 前两层保持稠密 | 48 层全部稀疏 |
| 最近上下文 | 当前（最后）一页 | 最近 64 个位置（不对齐，可能与最新完整页重叠） |
| 预算 | token 预算（如 2048） | `top_k = 2048`，包含最近窗口 |
| 包围盒 | 每页每通道 min/max | 相同（bf16，哨兵表示无效页） |
| 与 KV 管理的关系 | KV 全在 GPU | 可选 HiSparse：全量 KV 在 host，GPU 只放 LRU 热缓冲 |

---

## 11. 自己动手看

```bash
python examples/01_topk_by_hand.py   # 合成数据：逐步打印包围盒、分数、上界、选中的页
qwenquest demo                       # 随机小模型：每步选了哪些页、重复位置、HiSparse 命中率
qwenquest info --context-len 131072  # 30B-A3B 的显存/预算计算
pytest tests/test_quest_algorithm.py -v
```

在自己的代码里记录每一次选择：

```python
from qwenquest import Engine, EngineConfig, QuestTrace

engine = Engine(model, EngineConfig(attention_mode="quest", hisparse_config={"top_k": 2048}))
engine.quest.trace = QuestTrace()
engine.generate([ids], max_new_tokens=32)
for rec in engine.quest.trace.for_layer(0):
    print(rec.seq_len, rec.selected_pages, rec.recent_window)
```
