# 2026-05-28 AttnPredict-Offload 会话上下文

本文用于给新会话初始化背景。默认使用中文交流；运行命令必须使用项目虚拟环境 `.venv`，不要用系统 Python。

## 0. 必须遵循的 skill 和环境约束

新会话继续做 `attnpredict-offload` 时，必须先遵循仓库内 skill：

- `$add-sparse-method`：保持 Sparse-vLLM 的 cache-manager-first 架构。`attnpredict-offload` 的状态和核心逻辑应放在 `src/sparsevllm/engine/cache_manager/attnpredict_offload.py`，跨层调度/attention score 收集由 `src/sparsevllm/engine/sparse_controller.py` 通过通用 hook 触发，`src/sparsevllm/layers/attention.py` 尽量保持方法无关。
- `python-code-slim`：只保留行为必要、性能有证据的代码。不要新增无必要配置项，不要保留无收益实验代码，不要为了兜底加入理论上不会走到的复杂分支，不要把热路径张量操作改成 Python loop。注释保留中文短注释，说明方法作用、关键变量含义、shape/stream/lease/residency 等非显然约束。

环境和实验约束：

- 必须使用 `.venv`，不要使用系统 Python 包。
- benchmark 脚本里调用 `python` 时，应在命令前加：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH
```

- 直接跑 Python 脚本时优先用：

```bash
.venv/bin/python ...
```

- 不要重新下载依赖；`.venv` 里的包已经齐全。
- 验证代码先跑 `py_compile`，再跑小 smoke，最后跑目标 benchmark。
- 当前主要目标 benchmark 是 `128k / batch_size=2 / output_len=64 / gpu_memory_utilization=0.7 / reuse=4 / max_stale=6`。
- 注意工作区可能有无关 dirty 文件，不要回退或覆盖 `.gitignore`、`AGENTS.md`、`.vscode`、论文、`hfd.sh` 等无关改动。

Claude Code 审查约束：

- 服务器已安装 Claude Code CLI，命令为 `/usr/bin/claude`，当前验证版本为 `2.1.153 (Claude Code)`。
- 每次写完代码后，先跑本地验证，再让 Claude Code 对本次 diff 做只读审查。推荐形式：

```bash
git diff -- <相关文件> | claude -p "请审查这个 diff。重点关注行为是否变化、CUDA stream/prefetch/lease/resident KV 时序是否有竞态、decode 性能是否可能回退、是否违反 python-code-slim。只输出必须修的问题、代码位置和理由。"
```

- 不要让 Claude Code 自动改代码；它只作为第二审查员。
- 是否采纳 Claude Code 的意见由 Codex 判断。应采纳明确指出正确性、竞态、shape/device、性能回退、benchmark 参数错误的问题；谨慎采纳能缩短代码且不改变 hot path 的建议；拒绝泛泛重构、无证据性能猜测、增加复杂兜底、或违背 `python-code-slim` / `$add-sparse-method` 的建议。
- 采纳后必须重新跑 `py_compile`、必要 smoke、以及对应 benchmark；最终汇报 Claude Code 提了什么、采纳/拒绝了什么、理由是什么。

## 1. 会话目标

本轮围绕 `attnpredict-offload`（AttentionPredictor 的 KV cache offload 版本，即用 predictor 预测 hot tokens，同时把完整 KV 放在 CPU，GPU 只保留 active KV）做性能和代码整理。

主要目标：

- 从一版代码量较大、策略较杂的实现中回退。
- 以 `3b2071d9591fe9d085dccd5a2a35af5c33245ccd` 为基线，该提交名为“回退至跨步复用版本”。
- 在该版本上只保留/重做必要策略，先验证跨步复用，再添加“4 层跨层复用一个预测结果”。
- 用 `128k / batch_size=2 / output_len=64` 做 vanilla 与 `attnpredict-offload` 对比。
- 判断当前 `attnpredict-offload` 是否能在速度上超过 vanilla 基线。

## 2. 关键术语

- `reuse_steps`（跨步复用步数）：一份 predictor 预测出来的 hot token 结果连续复用多少个 decode step。
- `max_stale_steps`（最大陈旧步数）：后台 predictor 没算完时，旧预测结果最多允许继续复用多少步，超过后需要等待新结果。
- `layer_reuse_stride`（跨层复用跨度）：每多少层共享一个 predictor 结果。本轮固定为 `4`，不做配置项。
- `lease`（租约）：当前层正在使用的一份 hot token 预测结果，包括每个 row 的 hot positions 以及这份结果从哪个 decode 位置开始使用。
- `prefetch`（预取）：后台提前把下一步需要的 KV 从 CPU 搬到 GPU。
- `packed view`（打包视图）：把当前 attention 要看的 sink/hot/recent/current token 压成二维 slots 表，交给 decode kernel 使用。

## 3. 回退版本和当前代码状态

回退目标提交：

```text
3b2071d9591fe9d085dccd5a2a35af5c33245ccd
```

该提交本身已经包含最小跨步复用：

```text
attnpredict_reuse_steps = 4
attnpredict_max_stale_steps = 6
```

回退后覆盖过的相关 runtime/benchmark 文件：

```text
scripts/bench_attnpredict_vs_vanilla_128k.sh
src/sparsevllm/config.py
src/sparsevllm/engine/cache_manager/attnpredict.py
src/sparsevllm/engine/cache_manager/attnpredict_offload.py
src/sparsevllm/engine/cache_manager/base.py
src/sparsevllm/engine/sparse_controller.py
src/sparsevllm/layers/attention.py
src/sparsevllm/triton_kernel/context_flashattention_nopad.py
```

当前最终代码状态：

- `src/sparsevllm/config.py` 保留 `attnpredict_reuse_steps=4` 和 `attnpredict_max_stale_steps=6`。
- `src/sparsevllm/engine/cache_manager/attnpredict_offload.py` 在回退版本基础上新增固定：

```python
self._layer_reuse_stride = 4
```

- 没有新增 `attnpredict_refresh_layer_stride` 配置项。
- 没有保留 batch predictor、full-resident fast path、fused decode kernel、recent slots 缓存等前面实验性代码。
- 当前工作区中相关代码改动只有：

```text
M src/sparsevllm/engine/cache_manager/attnpredict_offload.py
```

另外还有无关 dirty 文件，不要随意修改或回退：

```text
M .gitignore
M AGENTS.md
?? .vscode/settings.json
?? hfd.sh
?? 论文/*.md
```

## 4. 当前保留的优化策略

### 4.1 跨步复用

位置：

```text
src/sparsevllm/engine/cache_manager/attnpredict_offload.py
```

核心行为：

- 只有当当前 lease 复用到 `reuse_steps=4` 时，组首层才重新收集 decode attention score。
- 如果后台 predictor 还没完成，则继续使用旧 lease。
- 如果达到 `max_stale_steps=6`，消费该层时会强制等待后台结果。

关键方法：

```text
should_collect_decode_attn_score()
_lease_age_reached()
_consume_prefetch()
get_layer_store_view()
```

### 4.2 固定 4 层跨层复用

当前实现是固定策略，不加配置项，符合 `python-code-slim` 的“不要为内部常量新增配置”要求。

行为：

- 第 `0, 4, 8, ...` 层是真正跑 predictor 的 source layer。
- 同组后 3 层复用组首层预测出来的 hot positions。
- prefill predictor 初始化也只在组首层收集 tail-score，非组首层直接跳过。
- decode refresh 也只在组首层收集 attention score。
- source layer 的后台 future 完成后，一次性给同组 4 层准备 residency 并提交 lease。

新增/修改方法：

```text
_is_layer_reuse_source()
_layer_reuse_targets()
_expand_layer_results()
_ensure_layer_results_resident()
_commit_layer_results()
```

重要时序约束：

- 没有把同一个 future 立刻挂到同组后 3 层，避免它们在同一个 decode token 内提前消费“下一步”的预测结果。
- 只有 source layer 在下一步消费 future 时，才提交整组 4 层的新 lease。

## 5. 已经尝试但当前已撤销的策略

这些策略曾经实现或实验过，但当前代码没有保留。

### 5.1 分层错峰 refresh + 可配置 refresh_layer_stride

曾加过 `attnpredict_refresh_layer_stride` 配置项，后来撤销。原因：

- 代码复杂度上升。
- 在 64k/128k 下仍未超过 vanilla。
- 用户希望先回到更干净的跨步复用版本。

### 5.2 batched predictor refresh

曾把一个 decode step 内多个层的 predictor refresh 合成批量任务。当前撤销。

原因：

- 有一定改善，但代码量和状态管理变复杂。
- 后续和其他策略叠加后难以判断收益来源。

### 5.3 full-resident fast path

曾尝试当完整历史 KV 已在 GPU 时跳过 offload 驱逐/H2D。当前撤销。

原因：

- 改动面较大。
- 在目标实验下不足以让 offload 超过 vanilla。
- 用户希望当前版本尽量干净。

### 5.4 fused decode kernel

曾写过单阶段 no-score decode kernel，试图替代 stage1 + stage2 两段 kernel。当前已删除。

结果：

```text
128k / bs=2 / output=64 / reuse=4 / max_stale=6 / stride=32
fused kernel 实验 DecTP 约 45.5 tok/s
```

没有带来突破，且自写 kernel 并行度不足，因此撤销。

### 5.5 recent slots 缓存

曾尝试缓存 full-resident fast path 下的 recent slots，仅追加当前 token。收益接近噪声，已撤销。

## 6. 实验记录

所有 benchmark 均使用 `.venv`，不要用系统环境。

### 6.1 回退到只跨步复用后的 128k 实验

配置：

```text
LENGTHS=128000
BATCH_SIZES=2
OUTPUT_LEN=64
GPU_MEMORY_UTILIZATION=0.7
attnpredict_reuse_steps=4
attnpredict_max_stale_steps=6
无跨层复用
```

结果：

```text
vanilla
TTFT 42.65s | PreTP 6003.1 | DecTP 46.8 | ITL 42.75ms | Mem 66.49GB | 1.00x

attnpredict-offload
TTFT 53.21s | PreTP 4811.1 | DecTP 4.9 | ITL 410.71ms | Mem 53.72GB | 0.10x
```

结论：

- 只跨步复用能省显存，但 decode 极慢。
- 说明每层 predictor refresh / prefetch / packed view 的固定成本仍然很重。

### 6.2 添加固定 4 层跨层复用后的 smoke

配置：

```text
length=4096
batch_size=2
output_len=8
reuse=4
max_stale=6
layer_reuse_stride=4
```

结果：

```text
attnpredict-offload
TTFT 0.98s | PreTP 8334.5 | DecTP 21.6 | ITL 92.49ms | Mem 16.80GB
```

结论：

- smoke 通过。
- 日志确认 `layer_reuse_stride=4` 生效。

### 6.3 添加固定 4 层跨层复用后的 128k 实验

配置：

```text
LENGTHS=128000
BATCH_SIZES=2
OUTPUT_LEN=64
GPU_MEMORY_UTILIZATION=0.7
attnpredict_reuse_steps=4
attnpredict_max_stale_steps=6
layer_reuse_stride=4
```

结果：

```text
vanilla
TTFT 42.67s | PreTP 5999.3 | DecTP 46.9 | ITL 42.67ms | Mem 66.49GB | 1.00x

attnpredict-offload
TTFT 55.57s | PreTP 4606.9 | DecTP 14.0 | ITL 143.25ms | Mem 50.19GB | 0.30x
```

结论：

- 4 层跨层复用有效：DecTP 从 `4.9` 提到 `14.0`。
- 显存进一步下降：`53.72GB` 到 `50.19GB`。
- 但仍明显低于 vanilla 的 `46.9 tok/s`。

### 6.4 更早期、已撤销复杂版本上的实验参考

这些结果只作为方向判断，当前代码不包含对应策略。

64k / bs=2 / output=64：

```text
vanilla DecTP 约 61-65 tok/s

stride=1   offload DecTP 15.44 tok/s
stride=4   offload DecTP 30.61 tok/s
stride=8   offload DecTP 38.26 tok/s
stride=16  offload DecTP 42.23 tok/s
stride=32  offload DecTP 44.07 tok/s
```

诊断实验：

```text
reuse=10000 / max_stale=10000 / stride=32
64k / bs=2 / output=64
offload DecTP 51.06 tok/s
```

解释：

- 这个配置不保证输出质量，不可作为正式策略。
- 即使几乎不做 decode refresh，在 64k 下也没有超过 vanilla，说明瓶颈不只是 predictor refresh。

预算实验：

```text
num_top_tokens=2048 -> offload DecTP 45.42 tok/s
num_top_tokens=1024 -> offload DecTP 45.86 tok/s
```

解释：

- 降低 sparse token budget 没有明显突破。
- 当前主要瓶颈不是 attention FLOPs 本身，而是 offload 路径固定开销。

## 7. 运行命令

语法检查：

```bash
.venv/bin/python -m py_compile \
  src/sparsevllm/config.py \
  src/sparsevllm/engine/cache_manager/base.py \
  src/sparsevllm/engine/sparse_controller.py \
  src/sparsevllm/engine/cache_manager/attnpredict.py \
  src/sparsevllm/engine/cache_manager/attnpredict_offload.py \
  src/sparsevllm/layers/attention.py \
  src/sparsevllm/triton_kernel/context_flashattention_nopad.py
```

128k benchmark：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
ATTNPREDICT_MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth \
LENGTHS=128000 \
BATCH_SIZES=2 \
OUTPUT_LEN=64 \
GPU_MEMORY_UTILIZATION=0.7 \
ATTNPREDICT_REUSE_STEPS=4 \
ATTNPREDICT_MAX_STALE_STEPS=6 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

小 smoke：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
.venv/bin/python scripts/bench_sparse_vllm.py \
  --model_path /root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
  --methods attnpredict-offload \
  --lengths 4096 \
  --batch_sizes 2 \
  --output_len 8 \
  --temperature 0.0 \
  --hyper_params '{"gpu_memory_utilization":0.7,"chunk_prefill_size":4096,"tensor_parallel_size":1,"attnpredict_model_path":"/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth","attnpredict_reuse_steps":4,"attnpredict_max_stale_steps":6,"attnpredict_offload_cpu_memory_utilization":0.7}'
```

## 8. 当前判断

当前 `attnpredict-offload` 更像是省显存方案，而不是吞吐必胜方案。

原因：

- vanilla 全量 KV 在 GPU 上，路径简单，kernel 连续读，没有 CPU/GPU 搬运、predictor、lease、residency bookkeeping。
- offload 虽然减少 GPU KV 常驻量，但新增了 CPU full backing、H2D prefetch、packed view 构造、predictor refresh、异步 future 等固定开销。
- 在 `128k / bs=2 / output=64` 下，decode token 数较短，固定开销很难被摊平。
- 4 层跨层复用已经明显改善，但仍只有 vanilla 的约 30% decode throughput。

## 9. 后续可能方向

如果继续追求超过 vanilla，建议不要再堆 Python 策略，优先考虑更底层的实现变化：

1. **kernel/layout 级重做 packed view**
   - 减少每层每步 Python/Numpy/Torch 打包。
   - 尽量让 predictor 输出直接变成 GPU-side slot/block 表。

2. **真正高效的 fused kernel**
   - 当前试过的简单单阶段 fused kernel 没赢。
   - 如果继续做，需要专业地重写 GQA decode kernel，保证并行度和访存模式。

3. **减少 CPU/GPU 搬运**
   - 改进 active pool 布局。
   - 避免频繁驱逐再搬回。
   - 探索更连续的 GPU residency 结构。

4. **质量评估后扩大复用**
   - 更大的 `reuse_steps` 或更大的 `layer_reuse_stride` 可能提速。
   - 但需要 LongBench/固定样例质量评估，不能只看速度。

5. **换实验场景**
   - 更长输出长度、更大 batch、更紧显存条件下，offload 的价值可能更明显。
   - 当前 `bs=2/output=64` 对 vanilla 很友好，对 offload 不友好。

## 10. 给新会话的建议起点

新会话开始时建议先读：

```text
README.md
attnpredict-offload资料/推理流程与attention调用时序.md
attnpredict-offload资料/2026-05-28_AttnPredictOffload会话上下文.md
src/sparsevllm/engine/cache_manager/attnpredict_offload.py
```

然后确认：

- 是否继续保留当前固定 `layer_reuse_stride=4`。
- 是否需要做质量评估。
- 是否要转向 kernel/layout 优化，而不是继续加 Python 调度策略。
- 写完任何代码后，是否已执行 Claude Code 只读审查，并基于证据判断是否采纳。
