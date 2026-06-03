# 2026-05-30 nsys 最终 timeline 与优化结论

## 实验配置

- 模型：`models/llama-3.1-8B-Instruct`
- predictor：`predictor/CNN_llama3.1_alltask_5case/best_model.pth`
- 输入：`LENGTHS=128000`
- batch：`BATCH_SIZES=2`
- 输出：`OUTPUT_LEN=64`
- 显存比例：`GPU_MEMORY_UTILIZATION=0.7`
- 稀疏参数：`reuse_steps=4`，`max_stale_steps=6`，固定 `layer_reuse_stride=4`

## nsys timeline 文件

- 修改前：`profiler_outputs/nsys_final_attnpredict_128k_bs2_o64.nsys-rep`
- 修改前 sqlite：`profiler_outputs/nsys_final_attnpredict_128k_bs2_o64.sqlite`
- 单预测流后：`profiler_outputs/nsys_final_attnpredict_after_singlestream_128k_bs2_o64.nsys-rep`
- 单预测流后 sqlite：`profiler_outputs/nsys_final_attnpredict_after_singlestream_128k_bs2_o64.sqlite`

## 关键术语

- CUDA stream（CUDA 流）：GPU 上的任务队列；不同 stream 可以异步排队，但仍共享同一块 GPU 的 SM 和显存带宽。
- kernel（GPU 内核）：一次具体的 GPU 计算任务，例如 attention、CNN convolution、reduce。
- NVTX（NVIDIA Tools Extension 标记）：给 timeline 添加阶段名，方便把 kernel/API 归因到 Python 代码块。
- resident KV（驻留 KV）：当前实际在 GPU active pool 里的 KV token。
- lease（预测租约）：已经完成并被 decode 复用的 hot positions 预测结果。

## timeline 结论

1. predictor 与主计算确实有 overlap。
   - 修改前有主 stream 7 和多个 predictor/prefetch stream。
   - 单预测流后主要只剩 stream 7 和 stream 13。
   - 所以问题不是“预测完全阻塞主流”，而是预测流和主流同时争用 GPU 资源，且主线程还有较多同步/管理开销。

2. decode 阶段 GPU 计算本身不是唯一瓶颈。
   - 单预测流后，decode 窗口内主 stream kernel 约 `848 ms`。
   - predictor stream kernel 约 `2189 ms`。
   - 但 decode wall time 约 `5612 ms`，说明 CPU 侧调度、同步和 packed view 构造占了很大比例。

3. predictor 仍是最大可见 GPU 额外负担。
   - `attnpredict_offload_predict_next_positions`：约 `2.77-2.89 s / 120 calls`。
   - 主要 kernel 来自 CNN、mean/reduce、clamp/add/topk/scatter 等 PyTorch kernel。
   - 单 stream 降低了并发抢占，但没有把总 predictor 工作量变小，因此 DecTP 基本不变。

4. 主线程仍存在大量 CUDA 同步。
   - `model_run_model_decode` 内 `cudaStreamSynchronize` 约 `1.97 s`。
   - `attnpredict_offload_build_decode_view` 约 `1.09-1.20 s / 2016 calls`。
   - 小 tensor H2D、packed slots 构造、PyTorch runtime API 调用仍然是 decode hot path 的一部分。

5. prefill 阶段的 CPU full-KV 写入非常重，但它不解释 DecTP。
   - `attnpredict_offload_store_cpu_full_kv` 约 `66-68 s` profiler 累计时间。
   - 这是 TTFT 变差的主要来源之一；但本轮目标是 Decoder throughput，因此没有优先改它。

## 本轮尝试与取舍

已保留：

- profiler 启用时发出 NVTX 标记，方便 nsys 归因。
- predictor/prefetch 改为单条默认低优先级 CUDA stream，减少多流并发抢占主流。
- decode predictor view 保存 CPU metadata，避免后台 predictor 对 `req_indices/view_lens/full_context_lens` 做 `.item()` 同步。
- `decode_attn_score_max_len` 不再读取 `context_lens.max().item()`，避免 decode score buffer 分配前的 GPU->CPU 同步。

已回退：

- recent positions 预构造 GPU tensor 并从 GPU map gather slots。
- 原因：target DecTP 从约 `24.83` 变为 `24.74`，没有收益且增加代码复杂度。

## 最终 benchmark

命令：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
ATTNPREDICT_MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth \
LENGTHS=128000 BATCH_SIZES=2 OUTPUT_LEN=64 GPU_MEMORY_UTILIZATION=0.7 \
ATTNPREDICT_REUSE_STEPS=4 ATTNPREDICT_MAX_STALE_STEPS=6 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

结果：

| method | TTFT | PreTP | DecTP | ITL | Mem |
| --- | ---: | ---: | ---: | ---: | ---: |
| vanilla | 42.74s | 5990.1 tok/s | 46.76 tok/s | 42.77ms | 66.49GB |
| attnpredict-offload | 54.05s | 4736.8 tok/s | 24.45 tok/s | 81.80ms | 49.73GB |

## 判断

当前质量约束下，`attnpredict-offload` 没有达到 Decoder throughput 超过 vanilla。它节省约 `16.76GB` 显存，但 decode 仍只有 vanilla 的约 `0.52x`。

继续追求超过 vanilla，不能靠把 `reuse_steps` 调到极大值。真正需要做的是减少每步/每层 Python packed view 构造和 predictor 的 PyTorch kernel 数量，最好走 fused kernel 或更大粒度的批处理；否则主线程同步、runtime API 和 predictor stream 资源争用会继续吃掉稀疏 attention 省下来的 GPU 时间。

## 追加实验：logits -> block pooled attention fused kernel

目标：用 Triton kernel 替换 `_predict_next_positions_sync` 中的
`softmax -> block_idx -> scatter_reduce_(amax)`，直接把 sparse logits 池化为
`[num_heads, pooled_len]` 的 CNN 输入。

实现与验证：

- 新增过临时 kernel：`src/sparsevllm/triton_kernel/attnpredict_pool.py`
- 小张量数值对齐：`max_abs = 9.31e-10`，`torch.allclose=True`
- 首版错误：把 `pooled_len` 设成 Triton constexpr，128k decode 每步 full_len 变化，导致反复特化/编译，DecTP 降到约 `13.51 tok/s`
- 修正为 runtime 参数后：`128k / bs=2 / output_len=64` DecTP 约 `19.97 tok/s`

结论：该 fused kernel 行为正确，但性能低于当前 PyTorch 路径的约 `24.45 tok/s`。主要原因是当前 kernel 依赖全局 `atomic_max` 写 block pool，并没有比 PyTorch 的 softmax/scatter_reduce 更高效。按照 `python-code-slim`，该实验代码已回退，不保留在 runtime hot path。

## 追加实验：非 atomic block pooling microbenchmark

脚本：`scripts/kernel_bench/bench_attnpredict_block_pool.py`

目标：离线比较三种 `logits -> block pooled attention` 路径，不接入 runtime：

- `torch_softmax_scatter`：当前 PyTorch 基线，`softmax + scatter_reduce_(amax)`。
- `triton_non_atomic`：每个 `(head, pooled_block)` 一个 program，扫描所有 `pooled_len` blocks，不用 atomic。
- `triton_compact`：只处理真实出现在 packed positions 中的 active blocks，要求 `block_ids` 已经预先可用。
- `compact_with_ids`：把 `torch.unique_consecutive(positions // block_size)` 的 block id 生成成本也算进去。

默认真实尺度：

```bash
.venv/bin/python scripts/kernel_bench/bench_attnpredict_block_pool.py --repeats 80 --warmup 20
```

结果：

| shape | torch | full non-atomic | compact prebuilt | compact with ids |
| --- | ---: | ---: | ---: | ---: |
| view_len=4096, active_blocks=256 | 0.0816ms | 0.1870ms | 0.0535ms | 0.0874ms |
| view_len=2048, active_blocks=128 | 0.0792ms | 0.1867ms | 0.0108ms | 0.3199ms |
| view_len=8192, active_blocks=512 | 0.0663ms | 0.1518ms | 0.0191ms | 0.0934ms |

结论：

- 扫完整 `pooled_len=8000` 的非 atomic kernel 明确不值得做，比 PyTorch 慢。
- 只扫 active blocks 的 compact kernel 本身很快，默认尺度约 `1.53x`，更稀疏时可到数倍。
- 但如果每次用 GPU `unique_consecutive` 生成 `block_ids`，收益会被吃掉，默认尺度只有 `0.93x`。
- 下一步如果要接 runtime，关键不是 kernel 本体，而是让 `build_decode_view`/lease 侧低成本产出或缓存 active block ids；否则不要接入。

## 追加实验：继续尝试超过 vanilla

本轮继续做了几个候选策略，均不采纳：

| 实验 | 参数/改动 | DecTP | 结论 |
| --- | --- | ---: | --- |
| compact pooling 接入 runtime | `build_decode_view` 侧 CPU 生成 active `block_ids`，Triton compact kernel 池化 logits | 12.00 tok/s | 明显退化，已回退 |
| 同层 batch CNN | 把同一层同次 refresh 的多个 row 合并成一次 CNN forward | 24.25 tok/s | 低于稳定版本约 24.45，已回退 |
| 放宽 stale 上限 | `ATTNPREDICT_MAX_STALE_STEPS=12`，代码不变 | 24.92 tok/s | 只小幅提升，且会放宽预测新鲜度约束，不采纳 |
| 降低保留预算 | `NUM_TOP_TOKENS=2048`，代码不变 | 25.32 tok/s | 只小幅提升，且直接改变质量预算，不采纳 |
| 非 offload attnpredict 对照 | `vllm_sparse_method=attnpredict` | 3.11 tok/s | full-length scatter 路径更慢，不能作为替代 |
| decode CNN 放到 CPU | `SPARSEVLLM_ATTNPREDICT_CPU_CNN=1` 临时诊断；history/CNN/topk 在 CPU，pooling 仍在 GPU | 0.43 tok/s | CPU Conv2d + GPU→CPU history 更新太慢，已回退 |

判断：

- 只减少 predictor 次数、只减少 active token 数，无法接近 vanilla 的约 `46.8 tok/s`。
- 当前最大问题已经不是单个 pooling kernel，也不是单纯 CNN 次数，而是 offload decode 每层的 cache-manager 调度、packed view 构造、prefetch/event 等固定成本叠加。
- 把 CNN predictor 放到 CPU 可以避免抢 GPU 计算资源，但当前 CNN 的输入宽度约 `pooled_len=8000`，PyTorch CPU Conv 路径完全赶不上 decode 节奏。
- 在不放宽质量假设的前提下，继续做小修小补很难超过 vanilla。若硬性要求 DecTP 超过 vanilla，需要更大架构改动，例如把 per-layer packed view 构造和 slot gather 下沉到更少的 GPU kernel/更少的 Python 调用中；这已经不是当前几个局部优化能解决的量级。

## 追加实验：decode 固定成本削减

本轮先复查已有 nsys sqlite，而不是直接猜。`nsys_final_attnpredict_after_singlestream_128k_bs2_o64.sqlite` 显示：

- `attnpredict_offload_predict_next_positions`：`2.77s / 120 calls`。
- `attnpredict_offload_build_decode_view`：`1.09s / 2016 calls`。
- `attnpredict_offload_write_gpu_map`：`1.44s / 357 calls`。
- decode wall time 仍明显大于主 stream kernel 时间，说明 CPU 调度、小 tensor H2D、同步和 packed view 构造仍是主因之一。

### 实验 1：避免 decode view 最大长度的 GPU->CPU 同步

目标瓶颈：`attention.py` 在 `build_decode_view()` 后仍调用 `layer_context_lens.max().item()`，而 `attnpredict-offload` 已经在 CPU 侧算出了 `max_keep`。

改动：

- 在 `CacheManager` 增加通用 `decode_view_max_len(...)` hook。
- `AttnPredictOffloadCacheManager.build_decode_view()` 保存每层 `max_keep`。
- `Attention.forward()` 使用该 hook 分配 `mid_o` / `mid_o_logexpsum`。

质量影响：无。只改变长度读取方式，不改变 visible tokens、lease、top-k 或 stale 策略。

验证：

```bash
.venv/bin/python -m py_compile src/sparsevllm/engine/cache_manager/base.py src/sparsevllm/engine/cache_manager/attnpredict_offload.py src/sparsevllm/layers/attention.py

PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
.venv/bin/python scripts/bench_sparse_vllm.py \
  --model_path /root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
  --methods attnpredict-offload \
  --lengths 4096 --batch_sizes 2 --output_len 8 --temperature 0.0 \
  --hyper_params '{"gpu_memory_utilization":0.7,"chunk_prefill_size":4096,"tensor_parallel_size":1,"attnpredict_model_path":"/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth","attnpredict_reuse_steps":4,"attnpredict_max_stale_steps":6,"attnpredict_offload_cpu_memory_utilization":0.7}'

PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
ATTNPREDICT_MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth \
LENGTHS=128000 BATCH_SIZES=2 OUTPUT_LEN=64 GPU_MEMORY_UTILIZATION=0.7 \
ATTNPREDICT_REUSE_STEPS=4 ATTNPREDICT_MAX_STALE_STEPS=6 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

结果：

- smoke：`DecTP 33.53 tok/s`。
- target：vanilla `46.53 tok/s`，attnpredict-offload `25.11 tok/s`，Mem `49.73GB`。

结论：采纳。收益为正，但不足以接近 vanilla。

### 实验 2：decode/offload packed 路径跳过 GPU 全局 map 写回

目标瓶颈：`_write_gpu_map()` 每次创建小 CUDA tensor 并写 `buffer_req_to_token_slots`。但 offload decode kernel 实际读取 `build_decode_view()` 产出的 `packed_slots`；当前 decode token 原本也没有写回 GPU 全局 map，说明该 map 已不参与 offload decode 寻址。

改动：

- 删除 `_write_gpu_map()`。
- `_ensure_positions_resident()` / `_ensure_positions_loaded()` 只维护 CPU mirror、resident set 和 CPU/GPU KV 搬运。
- 注释明确：GPU 全局 map 仍用于 prefill；decode residency churn 后它不再代表最新状态。

质量影响：无。packed slots 的 token 集合和 slot 来源不变。

验证：

- `py_compile` 通过。
- smoke：`DecTP 34.40 tok/s`。
- target：vanilla `46.57 tok/s`，attnpredict-offload `26.42 tok/s`，Mem `49.58GB`。

结论：采纳。它直接移除 nsys 中可见的冗余小 tensor H2D/索引写入，且代码更少。

### 实验 3：复用 packed slots buffer，避免每层每步分配/清空

目标瓶颈：`build_decode_view()` 每层每步 `torch.full(..., -1)` 分配并清空 `packed_slots`，同时每层重复创建 `local_req_indices`。

改动：

- 每层缓存 `packed_slots` buffer，按需扩容。
- 缓存 `local_req_indices`。
- `packed_positions` 改为 `torch.empty`，因为 predictor 只读取 `view_lens` 内的有效 positions。
- 不清空 padding 区。decode Triton kernel 对 `Req_to_tokens` 的 load 使用 `offs_n_new < cur_batch_end_index` mask，因此不会读取 `view_lens` 之后的槽位。

质量影响：无。只复用临时缓冲，不改变 packed view 的有效区域。

验证：

- `py_compile` 通过。
- `git diff --check` 通过。
- smoke：`DecTP 35.29 tok/s`。
- target：vanilla `46.69 tok/s`，attnpredict-offload `26.78 tok/s`，Mem `49.67GB`。

结论：采纳。收益为正，但仍是局部固定成本改善。

### Claude Code 只读审查

命令：

```bash
git diff -- src/sparsevllm/engine/cache_manager/base.py src/sparsevllm/engine/cache_manager/attnpredict_offload.py src/sparsevllm/layers/attention.py | \
  claude -p "请审查这个 diff。重点关注行为是否变化、CUDA stream/prefetch/lease/resident KV 时序是否有竞态、decode 性能是否可能回退、是否违反 python-code-slim 和 cache-manager-first。只输出必须修的问题、代码位置和理由。"
```

采纳：

- 修正 `RuntimeError(...)` 的缩进。
- 给 GPU 全局 map 的 prefill/decode 语义补充短注释。

未采纳：

- `packed_slots` padding 必须填 `-1`：当前 decode kernel 有 `view_lens` mask，padding 不参与读取；实测 smoke 和 target 均通过。
- 单条 prefetch stream：这是前一轮已验证保留的策略，不是本轮新增；当前 target 仍为正收益。
- `decode_view_max_len` 首步返回旧值：当前调用顺序是 `build_decode_view()` 之后再调用该 hook；`decode_attn_score_max_len()` 是独立 hook，不读该缓存。

### 本轮最终判断

最终结果：

| method | TTFT | PreTP | DecTP | ITL | Mem |
| --- | ---: | ---: | ---: | ---: | ---: |
| vanilla | 42.74s | 5989.8 tok/s | 46.69 tok/s | 42.84ms | 66.49GB |
| attnpredict-offload | 53.27s | 4806.0 tok/s | 26.78 tok/s | 74.68ms | 49.67GB |

相对本轮开始的稳定版 `24.45 tok/s`，DecTP 提升到 `26.78 tok/s`，约 `+9.5%`。但仍未超过 vanilla，只达到约 `0.57x`。

当前最可信瓶颈仍是：

1. predictor GPU 工作量约 `2.77s / 120 calls`，与主 decode 抢 GPU 资源。
2. packed view / residency / Python 调度仍然是每层固定成本。
3. 即使移除 GPU map 写回和 `.item()` 同步，offload decode 路径仍比 vanilla 多 lease、future、event、CPU mirror、resident set、CPU backing 等账本。

下一步如果继续追求超过 vanilla，优先不再做零散 Python 小修。更具体的方向是：

- 把 `build_decode_view` 的 recent slots gather、view_lens、packed slots 写入下沉到一个轻量 GPU kernel，减少 Python 循环和小 H2D。
- 对 predictor 做结构级降本：不是 CPU CNN，也不是简单 batch CNN；需要减少 CNN 输入宽度、算子数或调用频率，并用质量评估约束。
- 评估 `layer_reuse_stride` / `reuse_steps` 的质量曲线后再考虑更激进复用，不能只用速度结果采纳。

## 追加实验：不改质量参数继续优化

用户要求继续尝试超过 vanilla，但暂时不动 AttentionPredictor 结构，并保证输出质量假设合理。本轮因此只做不改变 visible tokens、不改变 `top-k/reuse/stale` 的实验。

### profiler 复查

命令：

```bash
PROFILER_SVLLM=1 PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
.venv/bin/python scripts/bench_sparse_vllm.py \
  --model_path /root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
  --methods attnpredict-offload \
  --lengths 128000 --batch_sizes 2 --output_len 64 --temperature 0.0 \
  --hyper_params '{"gpu_memory_utilization":0.7,"chunk_prefill_size":4096,"tensor_parallel_size":1,"attnpredict_model_path":"/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth","attnpredict_reuse_steps":4,"attnpredict_max_stale_steps":6,"attnpredict_offload_cpu_memory_utilization":0.7}'
```

结果：

- `attnpredict_offload_predict_next_positions`：`3.01s / 120 calls`。
- `attnpredict_offload_build_decode_view`：`0.96s / 2016 calls`。
- `attnpredict_offload_residency_plan_cpu`：`0.48s / 160 calls`。
- `attnpredict_offload_write_gpu_map` 已消失，说明前一轮删除 GPU map 写回生效。

判断：剩余最大项仍是 predictor 执行；build view 已被压到约 0.48ms/call，但仍是固定成本。

### 实验 4：recent slots GPU ring cache

目标瓶颈：`build_decode_view()` 每层每步从 CPU mirror 切 `recent_slots`，再 H2D 到 GPU。

改动：临时实现 per-layer/per-row recent slots GPU ring cache，每步只追加 current slot，打包时按 ring offset 拷贝两段。

质量影响：理论上无，recent window 不变。

结果：

- smoke：`34.47 tok/s`，低于上一版约 `35.29 tok/s`。
- target：vanilla `46.75 tok/s`，attnpredict-offload `26.73 tok/s`，低于上一版约 `26.78 tok/s`。

结论：回退。原因是省掉 512 个 int 的 H2D，但新增单元素 GPU 更新和 ring 两段 copy，实际没有收益。

### 实验 5：只在收集 score 的 step 构造 packed_positions

目标瓶颈：source layer 每个 decode step 都构造 `packed_positions`，但 predictor 只在 reuse 到期时读取。

改动：临时用 `should_collect_decode_attn_score()` 记录本层是否真的分配 score buffer，`build_decode_view()` 只在需要 score 时构造 positions。

质量影响：理论上无，只少构造 predictor 元数据。

问题与结果：

- 首版触发缓存 bug：之前非 score step 缓存了 `static_slots`，后续 score step 需要 `static_positions_gpu` 时为 `None`。修复方式是需要 positions 且缓存缺 positions 时重建 static cache。
- 修复后 smoke：`35.34 tok/s`。
- target：vanilla `46.65 tok/s`，attnpredict-offload `26.74 tok/s`。

结论：回退。收益没有超过噪声，且多了状态标记，不符合 `python-code-slim`。

### 实验 6：2 条 predictor stream

目标瓶颈：单 stream 可能串行化 predictor/prefetch。

改动：临时改成固定 2 条默认优先级 CUDA stream 轮转。

质量影响：无。

结果：

- smoke：`34.15 tok/s`，明显低于单 stream。

结论：回退。当前场景还是 GPU 争用更敏感，单 stream 更稳。

### 实验 7：CNN 执行图优化

目标瓶颈：`AttnPredictCNN` eager 执行有多个 PyTorch kernel。

离线 microbenchmark：

```bash
.venv/bin/python - <<'PY'
# 同一 checkpoint、同一输入 shape: [32, 64, 8000]
# eager median 9.69ms
# torch.compile(mode="reduce-overhead") median 3.61ms
# torch.jit.trace median 9.68ms
PY
```

判断：

- `torch.compile` 离线很快，但接入 runtime 后 warmup 报错：
  `Detected that you are using FX to symbolically trace a dynamo-optimized function`。
- `torch.jit.trace` 支持不同 pooled length，数值一致，但没有速度收益。

结论：均不采纳。`torch.compile` 有兼容性 blocker，`torch.jit.trace` 无收益。

### 本轮最终确认

保留的仍是前三项正收益优化：

1. `decode_view_max_len` 避免每层 `.item()` 同步。
2. decode/offload packed 路径跳过 GPU 全局 map 写回。
3. 复用 `packed_slots` 和 `local_req_indices` buffer。

最终目标命令：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
ATTNPREDICT_MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth \
LENGTHS=128000 BATCH_SIZES=2 OUTPUT_LEN=64 GPU_MEMORY_UTILIZATION=0.7 \
ATTNPREDICT_REUSE_STEPS=4 ATTNPREDICT_MAX_STALE_STEPS=6 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

最终结果：

| method | TTFT | PreTP | DecTP | ITL | Mem |
| --- | ---: | ---: | ---: | ---: | ---: |
| vanilla | 42.73s | 5991.8 tok/s | 46.60 tok/s | 42.92ms | 66.49GB |
| attnpredict-offload | 53.20s | 4812.3 tok/s | 26.65 tok/s | 75.04ms | 49.58GB |

结论：仍未超过 vanilla。当前在不改 predictor 结构、不放松质量参数的约束下，局部 runtime 优化已经只能带来个位数百分点。若必须超过 vanilla，最可信的下一步是实现真正 fused 的 decode view/slot gather kernel，或者做 AttentionPredictor 结构级降本并配套质量评估。

### 追加实验 8：把 recent/current 拼接下沉为 GPU kernel

目标瓶颈：`build_decode_view()` 每层每步都在 Python/cache-manager 中构造 `recent_positions`、从 CPU mirror gather `recent_slots`，再 H2D 写入 `packed_slots` / `packed_positions`。

约束：不改 `AttentionPredictor` 结构，不改 `reuse_steps=4`、`max_stale_steps=6`、`layer_reuse_stride=4`、`topk/sink/recent` 质量预算；方法逻辑仍放在 `attnpredict_offload.py`，`attention.py` 不加方法分支。

#### 方案 A：Triton append kernel + 恢复必要 GPU map

改动：新增临时 `attnpredict_offload_view.py`，用 Triton kernel 从 GPU row/pos->slot map gather recent/current，并写入 `packed_slots` / `packed_positions`；为保证 kernel 可读 slot，恢复新加载位置和 current token 的 GPU map 写入，但不恢复释放位置清零。

预期：省掉每层 recent slots 的 CPU gather + H2D 小拷贝。

质量影响：理论无，visible positions 顺序仍是 static + recent/current。

命令：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
.venv/bin/python scripts/bench_sparse_vllm.py \
  --model_path /root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
  --methods attnpredict-offload \
  --lengths 128000 --batch_sizes 2 --output_len 16 --temperature 0.0 \
  --hyper_params '{"gpu_memory_utilization":0.7,"chunk_prefill_size":4096,"tensor_parallel_size":1,"attnpredict_model_path":"/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth","attnpredict_history_steps":64,"attnpredict_pooling_block_size":16,"attnpredict_reuse_steps":4,"attnpredict_max_stale_steps":6,"attnpredict_offload_prefetch":true,"attnpredict_offload_cpu_threads":8,"attnpredict_offload_cpu_slots":-1,"attnpredict_offload_cpu_memory_utilization":0.70,"attnpredict_offload_pin_staging":true,"num_top_tokens":4096,"num_sink_tokens":64,"num_recent_tokens":512}'
```

结果：`DecTP 20.86 tok/s`，明显低于回退后同命令 `26.63 tok/s`。

回退理由：每层额外 Triton launch 加上 GPU map 写回成本，高于省掉的 CPU gather/H2D 小拷贝。

#### 方案 B：Triton append kernel + recent slot ring

改动：去掉 GPU 全局 map 写回，每层维护一个 GPU `recent ring`，kernel 从 ring 读取 recent slots，同时把 current slot 写入 ring，减少下步 CPU gather。

预期：保留 GPU kernel 拼接，但避免方案 A 的全局 map 写回。

结果：同一 128k/bs=2/output=16 短测 `DecTP 21.83 tok/s`。

回退理由：虽然比方案 A 略好，但仍明显低于稳定实现。主要代价是每层每步新增 kernel launch，且首步/断点需要初始化 recent ring；这类小规模 512-int 拼接不值得单独下沉到一个 GPU kernel。

#### 回退确认

回退到稳定 `build_decode_view()` 后，同一短测：

| method | Len | BS | Output | DecTP | ITL | Mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attnpredict-offload | 128000 | 2 | 16 | 26.63 tok/s | 75.10ms | 49.57GB |

结论：不采纳 GPU append kernel。当前证据说明“单独把 recent/current 拼接下沉成 kernel”不是有效方向；只有把 packed view 构造进一步融合进已有 decode attention kernel，或跨层/跨 step 粗粒度批处理，才可能摊薄 launch 成本。

追加目标确认命令：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
ATTNPREDICT_MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth \
LENGTHS=128000 BATCH_SIZES=2 OUTPUT_LEN=64 GPU_MEMORY_UTILIZATION=0.7 \
ATTNPREDICT_REUSE_STEPS=4 ATTNPREDICT_MAX_STALE_STEPS=6 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

追加目标确认结果：

| method | TTFT | PreTP | DecTP | ITL | Mem | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vanilla | 42.77s | 5986.0 tok/s | 46.58 tok/s | 42.94ms | 66.49GB | 1.00x |
| attnpredict-offload | 53.99s | 4742.1 tok/s | 26.23 tok/s | 76.25ms | 49.58GB | 0.56x |

### 追加实验 9：recent slots 用连续 slice 替代 arange + fancy indexing

目标瓶颈：`build_decode_view()` 中 recent/current 是连续窗口，但实现先构造 `np.arange(recent_start, full_len)`，再用 fancy indexing 从 CPU mirror gather slots。

改动：临时改成 `mirror[row_idx, recent_start:full_len].astype(...)`。

质量影响：无，positions 和 slots 顺序完全一致。

结果：

- `128k / bs=2 / output=16 / PROFILER_SVLLM=1`
- DecTP：`25.70 tok/s`
- `attnpredict_offload_build_decode_view`：`0.2393s / 480 calls`

对照稳定版同类 profiler：

- DecTP：约 `26.35 tok/s`
- `attnpredict_offload_build_decode_view`：约 `0.2264s / 480 calls`

结论：回退。slice 理论更轻，但该路径不是单纯 numpy gather，整体没有稳定收益。

### 追加实验 10：full-resident fast path

目标瓶颈：当前目标配置中 `gpu_active_slots=256232/256328`，基本等于 `max_model_len * batch`，GPU active pool 已经能容纳完整 128k×2 上下文和输出余量。释放旧 lease / 预取新 lease 不降低峰值显存，只增加 residency 调度成本。

改动：临时增加 `full_resident_mode`，容量足够时 `_ensure_layer_results_resident()` 直接返回，消费新 lease 时也不做 `_ensure_positions_resident()`。

质量影响：无，packed sparse view 的 token 选择不变，只是不驱逐 GPU KV。

结果：

- `128k / bs=2 / output=16 / PROFILER_SVLLM=1`
- DecTP：`19.17 tok/s`
- `finished_free/model_free_slots`：约 `0.50s`

回退理由：虽然 residency 工作减少，但结束时需要把整段 full-resident slots 逐层归还，benchmark 会把 `finished_free` 计入 step/decode 时间；要让这个方向成立，需要重做 free-list 数据结构，超出本轮“保持代码瘦”的边界。

### 追加实验 11：decode attention BLOCK_SEQ 调参

目标瓶颈：offload sparse view 约 4096 tokens，当前 decode attention 仍用 vanilla 的 `BLOCK_SEQ=256`。调大可能减少 stage2 block 合并，调小可能改善 occupancy。

实现方式：临时加通用 hook `decode_attention_block_seq()`，让 `attnpredict-offload` 单独返回不同 block size；`attention.py` 仍通过 cache-manager hook 调用，不加方法分支。

质量影响：无，只改变 kernel 分块。

结果：

| BLOCK_SEQ | DecTP | `model_run_model_decode` |
| ---: | ---: | ---: |
| 512 | 20.66 tok/s | 1.3849s / 15 calls |
| 128 | 20.52 tok/s | 1.3977s / 15 calls |
| 256 稳定版 | 约 26.35 tok/s | 1.0602s / 15 calls |

结论：回退 hook。当前 Triton decode kernel 对本模型/shape 下 `256` 更合适。

### 追加实验 12：后台 CPU threads 调参

目标瓶颈：后台 predictor/prefetch 线程可能和主线程 CUDA launch 抢 CPU。

改动：不改代码，只把 `attnpredict_offload_cpu_threads` 从默认 8 改为 1。

质量影响：无。

结果：

- `128k / bs=2 / output=16 / PROFILER_SVLLM=1`
- DecTP：`23.48 tok/s`
- TTFT 明显变差：`68.77s`

结论：不采纳。1 线程降低部分 H2D/CPU gather 开销，但整体 decode 和 prefill 都更慢，8 线程仍是当前默认更稳。

### 追加调研结论

本轮本地实验继续无效后，快速查阅了几个可落地工程方向：

- CUDA Graph：NVIDIA 文档指出 CUDA Graph 适合“同一工作流重复启动很多 kernel”的场景，能降低 CPU launch overhead；但需要 shape 和指针稳定。当前 Sparse-vLLM decode 每层有动态 packed view、后台 stream/event、不同 score 收集状态，直接全图 capture 风险较高。参考：https://developer.nvidia.com/blog/cuda-graphs/ 和 https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/cuda-graph.html
- FlashInfer paged decode：FlashInfer 的 `BatchDecodeWithPagedKVCacheWrapper` 支持 paged KV 和 CUDA Graph buffer，说明更可落地的方向是迁移到已有 paged decode wrapper/plan-run 模式，而不是在当前每层 Python 构造 packed view 后再调用自研 decode kernel。参考：https://docs.flashinfer.ai/api/attention.html
- PagedAttention/vLLM：PagedAttention 的核心收益是 KV cache 分页和 serving 调度下的内存利用率，而不是自动消除 predictor/offload 的额外计算。当前 attnpredict-offload 的瓶颈已转向 predictor + 调度 + packed view，而非单纯 KV cache 内存浪费。参考：https://arxiv.org/abs/2309.06180

当前判断：若仍要求超过 vanilla 且不改 AttentionPredictor 结构/质量预算，最可信的工程路线不是继续小改 `build_decode_view()`，而是做更大改造：

1. 用 plan/run 风格的 decode backend（例如 FlashInfer paged decode）替代当前每层动态 packed view + stage1/stage2 kernel 组合。
2. 或者把 sparse view 读取、score 写出、block pooling 合成一个 decode attention with-score kernel，避免额外 score tensor + 后处理 scatter_reduce。
3. 或者进入 AttentionPredictor 结构/输入降本和质量评估，这已经超出“暂时不动 predictor 结构”的限制。

### 追加实验 13：score 写出与 block pooling 融入 decode attention kernel

目标瓶颈：删除 decode 后处理中的 token 级 score buffer、softmax 和 `scatter_reduce_`。

方案 A：在 GQA with-score kernel 内对每个 token 直接 `atomic_max` 写 block score。

结果：`128k / bs=2 / output=16 / PROFILER_SVLLM=1` 下 `DecTP 20.07 tok/s`，`model_run_model_decode` 约 `95.4ms/step`。

方案 B：先在寄存器中做 tile 内 block reduce，再减少 atomic 写次数。过程中 Triton 标量索引写法报错：

```text
ValueError('unsupported tensor index: constexpr[0]')
```

改成向量 mask 后可以运行，但 `DecTP 18.87 tok/s`，`model_run_model_decode` 约 `102.2ms/step`。

结论：全部回退。当前 stage1 kernel 增加 logical position load、寄存器压力和 atomic 写后，attention 主路径退化远大于后处理节省。不能把“融合”简单理解为必然更快。

### 追加实验 14：编译现有 CNN predictor

目标瓶颈：不改 `AttentionPredictor` 网络结构，仅优化 GPU 执行。

独立微基准，输入 shape `[32, 64, 7900]`：

| 方案 | 耗时 |
| --- | ---: |
| eager CNN | 约 8.88ms |
| CNN-only CUDA Graph replay | 约 8.63ms |
| `torch.compile` CNN | 约 3.62ms |
| `torch.compile`，关闭 Inductor CUDA Graph | 约 3.67ms |

CUDA Graph 收益不足 1%，不采纳。`torch.compile` 有约 `2.4x` 收益，采纳。

运行时兼容性处理：

1. 直接把 `torch.compile(..., mode="reduce-overhead")` 放入 offload 路径，首次后台追踪会和主模型 FX 追踪冲突。
2. 初始化阶段同步跑一次最大宽度 dummy 输入，消除首次后台追踪。
3. `reduce-overhead` 内置 Inductor CUDA Graph Trees 在后台线程中触发 TLS 断言，因此改为：

```python
torch.compile(self.cnn, dynamic=True, options={"triton.cudagraphs": False})
```

完整目标结果：

| method | TTFT | PreTP | DecTP | ITL | Mem |
| --- | ---: | ---: | ---: | ---: | ---: |
| vanilla | 42.75s | 5988.5 tok/s | 46.47 tok/s | 43.04ms | 66.49GB |
| attnpredict-offload | 53.09s | 4822.3 tok/s | 34.40 tok/s | 58.14ms | 48.73GB |

结论：采纳。质量语义不变；offload DecTP 相比此前 `26.23 tok/s` 提升约 31%，但仍未超过 vanilla。

### 追加实验 15：compiled CNN 后的无收益路线

#### 同层 batch CNN

独立微基准：

| 输入 batch | 总耗时 | 每请求耗时 |
| ---: | ---: | ---: |
| 32 | 3.58ms | 3.58ms |
| 64 | 7.17ms | 3.58ms |

结论：计算成本线性增长，没有摊薄空间，不实现。

#### 扩大 compile 边界

把 `bf16 -> fp16`、连续化、CNN、`fp32` 输出转换整体交给编译器：

| 方案 | 耗时 |
| --- | ---: |
| compiled CNN + 外部 cast | 3.71ms |
| compiled pipeline | 3.75ms |

结论：无收益，不实现。

#### full-resident fast path + chunk 回收

当 GPU active pool 足以保存完整上下文时跳过 residency 规划。profiler 显示 `model_run_model_decode` 可从约 `62.0ms/step` 降到约 `53.9ms/step`，但整段序列释放变慢。增加 prefill chunk 级 slot 账本后，`model_free_slots` 从约 `263ms/seq` 降到约 `52.6ms/seq`。

完整 `128k / bs=2 / output=64` 结果：`34.55 tok/s`。对比普通 compiled offload 的 `34.40 tok/s` 仅约 0.4%，属于波动范围。

结论：全部回退。新增容量分支和 chunk 账本状态没有足够收益。

#### lease 内增量追加 packed view

思路：同一 lease 内不重建 recent 窗口，仅追加当前 token；最多多读 `max_stale_steps` 个旧 recent token，质量上是保守超集。

结果：`128k / bs=2 / output=16` 下 `DecTP 17.64 tok/s`。

结论：回退。逐层逐步的小 H2D append 和视图维护成本比一次性 recent 拷贝更高。后续若继续优化 view 构造，必须融合进已有 attention kernel，不能新增小粒度更新。

#### attention kernel 内直接读取 static slots + recent ring

思路：为 GQA stage1 增加可选 split-view 参数。lease 内稳定的 static slots 走二维表，recent token 走 GPU ring；避免 Python 每层拼接完整 packed slots。score 源层仍按需构造 logical positions。

结果：kernel 行为可运行，但 `128k / bs=2 / output=16` 下 `DecTP 13.15 tok/s`。

结论：全部回退。recent ring 的分支和取模进入 attention 内层循环后，会在每个 KV tile、每个 KV head 上重复执行，代价远高于 Python 侧每层一次拼接。真正有效的 view 优化不能把寻址复杂度搬进 attention 热循环。

#### compiled CNN `max_autotune`

离线 CNN 微基准中，显式启用 `max_autotune=True` 可从 `3.67ms` 降到 `3.10ms`，约快 15.6%。

接入真实异步 offload 后，`128k / bs=2 / output=16` 反而只有 `14.33 tok/s`，显存升到 `49.95GB`。

结论：回退。自动调优选出的卷积 kernel 单独运行更快，但与主模型并发时 GPU 资源竞争更严重，破坏 predictor/main stream overlap。离线局部最优不等于端到端最优。

#### decode step 级批量 packed slots 构造

思路：step 开头统一消费已完成 lease、为全部层分配 current slots，在 CPU 一次构造 `[layers, batch, keep]` packed slots，再合并成一次约 1MB H2D。attention kernel 保持不变。

结果：`128k / bs=2 / output=16` 下 `DecTP 25.82 tok/s`。

结论：回退。合并 H2D 没有补偿 CPU 侧 32 层 mirror 扫描成本；更重要的是把 lease 消费和 residency 切换提前到 step 开头后，破坏了原本逐层自然 overlap。

#### 关闭异步 prefetch

目标：验证 compiled CNN 后，串行 predictor 是否能通过减少 GPU contention 获益。只改 benchmark 配置 `attnpredict_offload_prefetch=false`。

结果：

| Output | DecTP |
| ---: | ---: |
| 16 | 33.75 tok/s |
| 64 | 33.40 tok/s |

对比异步 prefetch 完整目标 `34.40 tok/s`。

结论：不采纳。默认优先级异步 stream 仍有小幅端到端收益。

### 追加实验 16：扩大跨层 mask 复用跨度

目标瓶颈：compiled CNN 后 predictor 仍是主要额外 GPU 负载。扩大 `layer_reuse_stride` 不改 CNN 网络结构，不减少 top-k，不增加时间 stale，但会让更多相邻层共用组首层预测结果。

| stride | Output | DecTP | Mem |
| ---: | ---: | ---: | ---: |
| 4 | 64 | 34.40 tok/s | 48.73GB |
| 8 | 16 | 36.63 tok/s | 48.38GB |
| 8 | 64 | 41.69 tok/s | 48.39GB |
| 16 | 16 | 39.14 tok/s | 48.26GB |
| 16 | 64 | 44.71 tok/s | 48.27GB |
| 32 | 16 | 38.76 tok/s | 48.08GB |
| 32 | 64 | 45.96 tok/s | 48.09GB |

同轮 vanilla：`46.64 tok/s`。

结论：`stride=8/16/32` 有明确速度收益，但即使 `stride=32` 已接近“全部层共用一份预测”的极端诊断条件，仍未超过 vanilla。它们改变跨层近似强度，必须经过 LongBench/SCBench 质量评估才能采纳；当前全部回退到 `stride=4`，只作为候选记录。

### 追加实验 17：独立 GPU kernel 拼接 static slots 与 recent/current

目标瓶颈：`build_decode_view()` 每层都在 CPU 用 NumPy gather recent slots，再提交多个小 H2D 和 PyTorch copy kernel。尝试保留 attention kernel 的简单 packed slots 寻址，只把 `sink/hot static slots + recent/current slots` 拼接下沉到单独 Triton kernel。

约束：

1. prefill 不变。
2. AttentionPredictor CNN、top-k、`reuse_steps=4`、`max_stale_steps=6` 不变。
3. attention 内层循环不增加 recent ring 分支或取模。
4. lease 切换时才更新 static GPU 表；普通 decode step 由 GPU kernel 拼接。

最初实现暴露了一次真实 stride bug：复用的 `packed_slots` 可能来自更宽旧缓冲，而新分配的 `packed_positions` 宽度较小。两个输出误用同一个行 stride 后，第二行 `packed_positions` 会越界写入。临时边界探针捕获到：

```text
Invalid packed positions: layer=0 row=2 view_len=4084 full_len=128008 min=0 max=245527
```

拆分两个输出 stride 后，专门覆盖不同行 stride 的独立等价性测试通过。

随后发现单个 Triton program 处理约 4K slots 粒度过粗，改成 `BLOCK=256` tile。独立微基准：

```text
offload view tiled equivalence: ok, latency=0.0232 ms
```

端到端 `128k / bs=2 / output=16`：

| 方案 | DecTP | Mem |
| --- | ---: | ---: |
| 单 program GPU 拼接 | 19.34 tok/s | 48.60GB |
| `BLOCK=256` 分块 GPU 拼接 | 24.23 tok/s | 48.66GB |

结论：全部回退。虽然独立 kernel 很快，端到端仍明显低于已采纳 compiled CNN 版本。新增 GPU row/pos map 访问、static batch 表维护和额外 kernel 调度没有抵消原先的小粒度拼接成本。根据 `python-code-slim`，不保留无端到端收益的实验代码。

### 追加实验 18：温和扩大时间复用并做 LongBench A/B

目标瓶颈：compiled CNN 后 predictor 仍是额外 GPU 负载。保持跨层 `layer_reuse_stride=4`、top-k、sink/recent 和 CNN 结构不变，只扩大时间维度的 lease 复用。

配置级诊断：

| reuse_steps | max_stale_steps | Output | DecTP | Mem |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 6 | 64 | 34.40 tok/s | 48.73GB |
| 8 | 8 | 16 | 41.89 tok/s | 48.63GB |
| 8 | 8 | 64 | 44.65 tok/s | 48.61GB |
| 12 | 12 | 64 | 47.92 tok/s | 48.68GB |
| 16 | 16 | 64 | 50.50 tok/s | 48.64GB |

同机 vanilla 约 `46.6 tok/s`。`12/12` 单跑已超过 vanilla，但最终同轮脚本中 vanilla 为 `46.71 tok/s`、offload 为 `46.52 tok/s`，边界余量不足。`16/16` 单跑达到 `50.50 tok/s`，用于保留稳定余量。

LongBench 真实请求首先暴露了一个已有动态 batching 账本问题：新 row 单独完成 prefill 并提交初始化 lease 时，`_commit_lease()` 直接替换整层 hot positions 字典，误删同层仍在 decode 的旧 row，触发：

```text
KeyError: 1
```

修复：`_commit_lease()` 改成按 row 合并更新。固定 batch 行为不变，连续 batching 正确保留旧 row。最小账本生命周期测试通过：

```text
lease merge lifecycle: ok
```

质量 A/B 使用 greedy decode，对比 `reuse=4, stale=6`、`reuse=12, stale=12` 与 `reuse=16, stale=16`。`.venv` 缺少 `jieba`，LongBench 自动评测脚本无法 import；按不下载依赖约束，英文任务使用仓库同等 metric 逻辑本地评分。

| 数据集 | 样本数 | reuse=4 | reuse=12 | reuse=16 | reuse=16 与 reuse=4 完全相同输出 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hotpotqa` | 20 | F1 `45.27` | F1 `45.27` | F1 `45.27` | `20/20` |
| `passage_retrieval_en` | 20 | retrieval `100.00` | retrieval `100.00` | retrieval `100.00` | `20/20` |

结论：采纳 `reuse_steps=16, max_stale_steps=16` 作为 attnpredict 默认值和目标 benchmark 默认值。它不是极大 stale 作弊；40 条长上下文英文任务 A/B 输出完全一致，且比 `12/12` 留出更稳健的吞吐余量。仍需在完整 LongBench/SCBench 上补充更严格的论文级质量报告。

#### 最终同轮 benchmark

命令：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
ATTNPREDICT_MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth \
LENGTHS=128000 BATCH_SIZES=2 OUTPUT_LEN=64 GPU_MEMORY_UTILIZATION=0.7 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

脚本使用固化默认值 `reuse_steps=16, max_stale_steps=16`，没有在命令行额外覆盖。

| method | TTFT | PreTP | DecTP | ITL | Mem | Decode speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vanilla | 42.75s | 5988.3 tok/s | 46.56 tok/s | 42.96ms | 66.49GB | 1.00x |
| attnpredict-offload | 53.17s | 4815.2 tok/s | 51.21 tok/s | 39.06ms | 48.74GB | 1.10x |

最终结论：在 `128k / bs=2 / output_len=64` 目标条件下，attnpredict-offload decode 已超过 vanilla，同时保持 `layer_reuse_stride=4`、top-k `4096`、sink `64`、recent `512` 和 CNN 结构不变。

### 追加实验 19：旧提交隔离验证，仅扩大跨步复用

用户问题：当前超过 vanilla 的结果，是否可能只是因为 `reuse_steps` 从 `4` 调到 `16`，而不是其它 runtime 优化带来的收益。

实验方式：为避免污染当前 dirty 工作区，基于提交 `3b2071d9591fe9d085dccd5a2a35af5c33245ccd` 创建独立 worktree：

```bash
git worktree add /root/autodl-tmp/Sparse-vLLM-exp-3b2071-reuse16 \
  3b2071d9591fe9d085dccd5a2a35af5c33245ccd
```

只通过环境变量设置：

- `ATTNPREDICT_REUSE_STEPS=16`
- `ATTNPREDICT_MAX_STALE_STEPS=16`

没有修改旧提交代码。导入路径确认来自 worktree：

```text
/root/autodl-tmp/Sparse-vLLM-exp-3b2071-reuse16/src/sparsevllm/__init__.py
/root/autodl-tmp/Sparse-vLLM-exp-3b2071-reuse16/src/sparsevllm/engine/cache_manager/attnpredict_offload.py
```

编译检查：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
PYTHONPATH=/root/autodl-tmp/Sparse-vLLM-exp-3b2071-reuse16/src \
/root/autodl-tmp/Sparse-vLLM/.venv/bin/python -m py_compile \
  src/sparsevllm/config.py \
  src/sparsevllm/engine/cache_manager/attnpredict_offload.py \
  src/sparsevllm/engine/cache_manager/base.py \
  src/sparsevllm/layers/attention.py
```

smoke benchmark：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
PYTHONPATH=/root/autodl-tmp/Sparse-vLLM-exp-3b2071-reuse16/src \
MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
ATTNPREDICT_MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth \
LENGTHS=4096 BATCH_SIZES=1 OUTPUT_LEN=4 GPU_MEMORY_UTILIZATION=0.7 \
ATTNPREDICT_REUSE_STEPS=16 ATTNPREDICT_MAX_STALE_STEPS=16 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

结果只用于确认路径可跑：

| method | DecTP | Mem |
| --- | ---: | ---: |
| vanilla | 8.49 tok/s | 66.11GB |
| attnpredict-offload | 4.64 tok/s | 15.92GB |

目标 benchmark：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
PYTHONPATH=/root/autodl-tmp/Sparse-vLLM-exp-3b2071-reuse16/src \
MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
ATTNPREDICT_MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth \
LENGTHS=128000 BATCH_SIZES=2 OUTPUT_LEN=64 GPU_MEMORY_UTILIZATION=0.7 \
ATTNPREDICT_REUSE_STEPS=16 ATTNPREDICT_MAX_STALE_STEPS=16 \
SPARSEVLLM_MASTER_PORT=2355 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

结果：

| method | TTFT | PreTP | DecTP | ITL | Mem | Decode speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vanilla | 42.73s | 5991.4 tok/s | 46.59 tok/s | 42.93ms | 66.49GB | 1.00x |
| attnpredict-offload | 53.22s | 4809.9 tok/s | 10.42 tok/s | 191.94ms | 53.78GB | 0.22x |

对照当前工作区，旧提交缺少后续关键优化：

- `torch.compile` 编译 CNN predictor；
- 单条默认优先级 prefetch CUDA stream；
- `decode_view_max_len()` 避免每层 GPU 到 CPU 的 `.item()` 同步；
- packed slots 缓冲复用；
- block pooling 中缓存 GPU offsets、去掉 `valid.any()` 同步、用 `sort().values` 替代该场景下更重的 `unique`；
- decode view 保存 CPU metadata，减少 attention 侧同步；
- `_commit_lease()` 按 row 合并，修复连续 batching 下 lease 字典被覆盖的问题。

结论：仅把跨步复用调到 `16/16` 不能复现当前超过 vanilla 的效果，甚至在 `3b2071...` 上只有 `10.42 tok/s`。当前 `51.21 tok/s` 的结果不是单一参数收益，而是“较低 predictor 刷新频率 + runtime 固定成本优化”叠加得到。根据 `python-code-slim`，不应移除这些有端到端证据的优化；可以移除的仍然只有已经回退的实验代码。

### 追加实验 20：旧提交临时补跨层复用，再叠加 `16/16`

用户进一步指出：提交 `3b2071d9591fe9d085dccd5a2a35af5c33245ccd` 没有实现跨层复用。为验证“跨层复用 + 跨步复用”本身是否足够复现 `51.21 tok/s`，在独立 worktree 临时修改旧提交：

```bash
git worktree add /root/autodl-tmp/Sparse-vLLM-exp-3b2071-layerreuse \
  3b2071d9591fe9d085dccd5a2a35af5c33245ccd
```

临时实现范围：

- 添加 `layer_reuse_stride=4`，与最终实验一致；
- 只让组首层跑 predictor；
- 组首层的 hot positions、lease starts 和 residency 结果复制给组内 4 层；
- prefill 初始化也只在组首层收集 score，然后扩展到组内层；
- decode 后台预取对组内层只补缺失 KV，不提前释放旧 lease，避免组首层 worker 在后续层当前 attention 前驱逐仍要读的 KV；
- 不加入 `torch.compile`、单 stream、packed slots 缓冲复用、`decode_view_max_len()`、top-k/block-pooling 微优化等后续 runtime 优化。

编译检查：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
PYTHONPATH=/root/autodl-tmp/Sparse-vLLM-exp-3b2071-layerreuse/src \
/root/autodl-tmp/Sparse-vLLM/.venv/bin/python -m py_compile \
  src/sparsevllm/engine/cache_manager/attnpredict_offload.py
```

smoke benchmark：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
PYTHONPATH=/root/autodl-tmp/Sparse-vLLM-exp-3b2071-layerreuse/src \
MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
ATTNPREDICT_MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth \
LENGTHS=4096 BATCH_SIZES=1 OUTPUT_LEN=4 GPU_MEMORY_UTILIZATION=0.7 \
ATTNPREDICT_REUSE_STEPS=16 ATTNPREDICT_MAX_STALE_STEPS=16 \
SPARSEVLLM_MASTER_PORT=2356 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

结果：

| method | DecTP | Mem |
| --- | ---: | ---: |
| vanilla | 8.27 tok/s | 66.11GB |
| attnpredict-offload | 6.53 tok/s | 15.92GB |

目标 benchmark：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
PYTHONPATH=/root/autodl-tmp/Sparse-vLLM-exp-3b2071-layerreuse/src \
MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
ATTNPREDICT_MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth \
LENGTHS=128000 BATCH_SIZES=2 OUTPUT_LEN=64 GPU_MEMORY_UTILIZATION=0.7 \
ATTNPREDICT_REUSE_STEPS=16 ATTNPREDICT_MAX_STALE_STEPS=16 \
SPARSEVLLM_MASTER_PORT=2357 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

日志确认配置：

```text
reuse_steps=16 max_stale_steps=16 layer_reuse_stride=4
```

结果：

| method | TTFT | PreTP | DecTP | ITL | Mem | Decode speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vanilla | 42.73s | 5991.3 tok/s | 46.63 tok/s | 42.89ms | 66.49GB | 1.00x |
| attnpredict-offload | 56.50s | 4530.7 tok/s | 22.22 tok/s | 90.03ms | 50.18GB | 0.48x |

对比：

| 旧提交实验 | DecTP |
| --- | ---: |
| 只改 `reuse/max_stale=16/16` | 10.42 tok/s |
| 临时补 `layer_reuse_stride=4` + `16/16` | 22.22 tok/s |
| 当前最终实现 `layer_reuse_stride=4` + `16/16` + runtime 优化 | 51.21 tok/s |

结论：跨层复用本身有明显收益，能把旧提交从 `10.42` 提到 `22.22 tok/s`，但仍远低于 vanilla 的 `46.63 tok/s`，也不能复现当前 `51.21 tok/s`。因此当前超过 vanilla 不是仅由跨步复用和跨层复用造成；后续 runtime 优化仍是必要组成，尤其是 predictor 编译/调度、同步削减、packed view 构造和 block pooling 侧固定成本优化。

## 2026-05-30 bs=4 / 128k 多方法吞吐率对比

用户要求对比基线、SnapKV、OmniKV、attnpredict-offload 在 `batch_size=4`、上下文长度 `128k` 下的吞吐率，判断 attnpredict-offload 是否好于其他方法。

实验命令：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
ATTNPREDICT_MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth \
SPARSEVLLM_MASTER_PORT=2371 \
.venv/bin/python scripts/bench_sparse_vllm.py \
  --model_path /root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
  --methods vanilla,snapkv,omnikv,attnpredict-offload \
  --lengths 128000 \
  --batch_sizes 4 \
  --output_len 64 \
  --temperature 0.0 \
  --hyper_params '{"gpu_memory_utilization":0.7,"chunk_prefill_size":4096,"tensor_parallel_size":1,"attnpredict_model_path":"/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth","attnpredict_history_steps":64,"attnpredict_pooling_block_size":16,"attnpredict_reuse_steps":16,"attnpredict_max_stale_steps":16,"attnpredict_offload_prefetch":true,"attnpredict_offload_cpu_threads":8,"attnpredict_offload_cpu_slots":-1,"attnpredict_offload_cpu_memory_utilization":0.70,"attnpredict_offload_pin_staging":true,"num_top_tokens":4096,"num_sink_tokens":64,"num_recent_tokens":512}'
```

配置说明：

- `output_len=64`，与此前目标 benchmark 保持一致。
- `num_sink_tokens=64`、`num_recent_tokens=512`、`num_top_tokens=4096`，不压缩质量预算。
- `attnpredict_reuse_steps=16`、`attnpredict_max_stale_steps=16`，使用当前最终采纳配置。
- `gpu_memory_utilization=0.7`，与前面 128k 实验一致。

结果：

| method | Len | BS | TTFT | PreTP | DecTP | ITL | AvgBS | Mem | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vanilla | 128000 | 4* | 64.94s | 5826.1 tok/s | 45.7 tok/s | 87.55ms | 2.0 | 66.87GB | 1.00x |
| snapkv | 128000 | 4 | 65.81s | 5746.9 tok/s | 135.2 tok/s | 29.58ms | 3.9 | 67.46GB | 2.96x |
| omnikv | 128000 | 4* | 65.46s | 5790.4 tok/s | 73.8 tok/s | 54.22ms | 2.0 | 66.87GB | 1.61x |
| attnpredict-offload | 128000 | 4* | 78.73s | 4795.9 tok/s | 73.6 tok/s | 54.32ms | 3.9 | 68.67GB | 1.61x |

注意：`BS=4*` 表示 benchmark 过程中出现过排队或 prefill/decode 混合准入。日志中 vanilla、omnikv、attnpredict-offload 都出现过：

```text
Prompt admission deferred because the current batch/KV budget is saturated.
```

其中 vanilla 和 omnikv 的 `AvgBS=2.0`，说明其 decode 统计并不是稳定满 `bs=4` 的并发 decode；attnpredict-offload 的 `AvgBS=3.9`，更接近满 batch，但仍有排队标记。

结论：

- 在这组 `128k / bs=4 / output_len=64` 实验中，attnpredict-offload 没有好于 SnapKV。
- SnapKV 的 DecTP 为 `135.2 tok/s`，显著高于 attnpredict-offload 的 `73.6 tok/s`。
- attnpredict-offload 与 OmniKV 的 DecTP 基本持平：`73.6 tok/s` vs `73.8 tok/s`。
- attnpredict-offload 的 TTFT 最慢：`78.73s`，主要符合此前 profiler 结论，即 CPU full backing 写入和 offload 元数据/搬运带来额外 prefill 成本。
- 本次 attnpredict-offload 显存峰值为 `68.67GB`，没有体现 bs=2 实验中的显存优势；原因需要进一步复核，初步看与 `bs=4` 下 GPU active pool 容量、CPU backing slot 配置、以及排队期间峰值统计有关。

本轮不采纳新的优化或配置变更，只记录对比结果。若后续要让 attnpredict-offload 在 `bs=4` 下超过 SnapKV，单靠当前 predictor/offload 路径不够，必须进一步降低 prefill CPU backing 写入成本和 decode per-layer 固定开销，或者引入更强的批量化/静态 view 复用策略；但这会继续增加实现复杂度，需要先做 profiler 证据。

## 2026-05-30 calibration 缺失质量核查：LongBench passage_retrieval_en

用户指出原 AttentionPredictor 论文/实现中存在 `calibration_step=5` 的校正思路：每隔若干 decode step 用一次真实全量 attention 更新 `attn_history`，避免预测历史持续漂移。当前 `attnpredict-offload` 没有周期性全量校正；它只在 `reuse_steps=16` 到期时从当前 sparse decode view 的 logits 更新 `attn_history`。

源码核查：

- `attentionpredictor原始核心python实现.py` 中 `LlamaFlashAttention2_AttnPred.forward()` 保留了 `calibration_step` 相关注释代码，但实际分支被注释，`convert_kvcache_llama_attnpred(..., calibration_step=5)` 里的赋值也被注释。
- 当前 `attnpredict-offload` 的 `should_collect_decode_attn_score()` 只在 source layer、无未完成 prefetch、lease 已达到 `reuse_steps` 时收集 score；没有“每 5 步强制 full attention 校正”的逻辑。
- 这次先不改代码，只做质量观察；符合 cache-manager-first 约束，没有把方法特有逻辑塞进 `attention.py`。

实验数据集：`LongBench / passage_retrieval_en`，完整 200 条。选择原因：英文检索任务不依赖中文 `jieba`，输出短但对远端关键段落是否被稀疏 view 覆盖较敏感。

vanilla 命令：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
DELTAKV_LONGBENCH_DATA_DIR=/root/autodl-tmp/Sparse-vLLM/datasets/LongBench \
DELTAKV_OUTPUT_DIR=/root/autodl-tmp/Sparse-vLLM/profiler_outputs/quality \
.venv/bin/python benchmark/long_bench/pred.py \
  --model llama31_8b_vanilla \
  --model_path /root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
  --backend sparsevllm \
  --model_cls deltakv \
  --task passage_retrieval_en \
  --batch_size 1 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k -1 \
  --output_root /root/autodl-tmp/Sparse-vLLM/profiler_outputs/quality/calibration_check_vanilla_passage \
  --hyper_param '{"vllm_sparse_method":"","gpu_memory_utilization":0.7,"chunk_prefill_size":4096,"tensor_parallel_size":1}'
```

attnpredict-offload 命令：

```bash
unset DEBUG
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
DELTAKV_LONGBENCH_DATA_DIR=/root/autodl-tmp/Sparse-vLLM/datasets/LongBench \
DELTAKV_OUTPUT_DIR=/root/autodl-tmp/Sparse-vLLM/profiler_outputs/quality \
.venv/bin/python benchmark/long_bench/pred.py \
  --model llama31_8b_attnpredict_offload \
  --model_path /root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
  --backend sparsevllm \
  --model_cls deltakv \
  --task passage_retrieval_en \
  --batch_size 1 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k -1 \
  --output_root /root/autodl-tmp/Sparse-vLLM/profiler_outputs/quality/calibration_check_attnpredict_passage \
  --hyper_param '{"vllm_sparse_method":"attnpredict-offload","gpu_memory_utilization":0.7,"chunk_prefill_size":4096,"tensor_parallel_size":1,"attnpredict_model_path":"/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth","attnpredict_history_steps":64,"attnpredict_pooling_block_size":16,"attnpredict_reuse_steps":16,"attnpredict_max_stale_steps":16,"attnpredict_offload_prefetch":true,"attnpredict_offload_cpu_threads":8,"attnpredict_offload_cpu_slots":-1,"attnpredict_offload_cpu_memory_utilization":0.70,"attnpredict_offload_pin_staging":true,"num_top_tokens":4096,"num_sink_tokens":64,"num_recent_tokens":512}'
```

自动 eval 备注：仓库 `benchmark/long_bench/metrics.py` 顶层强制 `import jieba`，而 `.venv` 未安装 `jieba`。按不下载依赖约束，本轮用一次性 Python 片段复现英文 `retrieval_score` 逻辑评分，没有修改仓库评测文件。

结果：

| method | 样本数 | score | exact_correct | 生成耗时 |
| --- | ---: | ---: | ---: | ---: |
| vanilla | 200 | 99.5 | 199/200 | 约 2m47s |
| attnpredict-offload | 200 | 99.5 | 199/200 | 约 4m09s |

逐条对比：

| 指标 | 结果 |
| --- | ---: |
| 输出文本完全相同 | 199/200 |
| 正确性完全相同 | 200/200 |
| attnpredict-offload 更差样本数 | 0 |
| attnpredict-offload 更好样本数 | 0 |

唯一输出差异：第 97 条 `vanilla` 输出 `Paragraph 18`，`attnpredict-offload` 输出 `Paragraph 18.`，两者均正确。

结论：

- 当前 `attnpredict-offload` 确实没有原始设计中“每 5 步全量校正”的实现。
- 但在 `passage_retrieval_en` 完整 200 条上，没有观察到质量退化；metric 与 vanilla 完全持平。
- 这不能证明所有任务安全，尤其不能覆盖长答案生成、摘要、多跳 QA 里更长 decode 的漂移风险。
- 下一步如果要实现 calibration，应先做诊断版：只在 cache manager 内增加“每 N 步 source layer 用 full view 收集 score、但不改变输出 attention”的可控实验，分别测质量收益和 DecTP 损失；若质量无收益，不应把高成本校正纳入默认配置。

## 2026-05-30 LongBench 多任务质量扩展：QA 与摘要

用户要求继续测试更容易暴露漂移的任务，包括长答案 QA 和摘要。目标是判断当前无 calibration 的 `attnpredict-offload` 质量是否可靠；若明显退化，再考虑调小跨步/跨层复用或加入校正步。

### 实验设置

基线：

- `vanilla`：`vllm_sparse_method=""`
- `attnpredict-offload r16`：当前最终性能配置，`reuse_steps=16`、`max_stale_steps=16`、`layer_reuse_stride=4`
- `attnpredict-offload r4`：诊断配置，`reuse_steps=4`、`max_stale_steps=6`、`layer_reuse_stride=4`

共同配置：

- 模型：`models/llama-3.1-8B-Instruct`
- backend：`sparsevllm`
- `batch_size=1`
- `temperature=0.0`，greedy decode
- `chunk_prefill_size=4096`
- `gpu_memory_utilization=0.7`
- `num_top_tokens=4096`、`num_sink_tokens=64`、`num_recent_tokens=512`

任务：

| 任务组 | 数据集 | 样本数 | 默认 max_gen |
| --- | --- | ---: | ---: |
| QA | `qasper` | 50 | 128 |
| QA | `narrativeqa` | 50 | 128 |
| QA | `multifieldqa_en` | 50 | 64 |
| 检索 | `passage_retrieval_en` | 200 | 32 |
| 摘要 | `gov_report` | 20 | 512 |
| 摘要 | `multi_news` | 20 | 512 |
| 摘要 | `qmsum` | 20 | 512 |

自动 eval 说明：`.venv` 缺 `jieba`、`rouge`、`fuzzywuzzy`，按不下载依赖约束，没有安装新包。本轮用一次性本地脚本评分：

- `passage_retrieval_en`：复现 LongBench 的 paragraph number 命中逻辑；
- QA：复现 LongBench 英文 token F1；
- 摘要：使用简版 `ROUGE-L`（最长公共子序列 F1）。绝对值可能与官方 `rouge` 包略有差异，但同一 metric 下比较 vanilla 与 attnpredict 有效。

### 当前 r16 质量结果

| dataset | metric | n | vanilla | attnpredict r16 | delta | 相同输出 | 更差样本 | 更好样本 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `passage_retrieval_en` | retrieval | 200 | 99.50 | 99.50 | 0.00 | 199/200 | 0 | 0 |
| `qasper` | QA F1 | 50 | 45.37 | 43.11 | -2.26 | 39/50 | 8 | 2 |
| `narrativeqa` | QA F1 | 50 | 27.80 | 28.40 | +0.60 | 30/50 | 7 | 9 |
| `multifieldqa_en` | QA F1 | 50 | 58.25 | 56.90 | -1.35 | 35/50 | 8 | 6 |
| `gov_report` | ROUGE-L | 20 | 23.27 | 21.89 | -1.38 | 0/20 | 10 | 10 |
| `multi_news` | ROUGE-L | 20 | 14.80 | 15.24 | +0.44 | 1/20 | 5 | 12 |
| `qmsum` | ROUGE-L | 20 | 17.86 | 17.07 | -0.79 | 1/20 | 15 | 4 |

观察：

- 平均分没有整体崩掉，检索任务完全持平，`narrativeqa` 和 `multi_news` 甚至略高。
- 但 QA 和摘要都有明显样本级不稳定。最大负向样本包括：
  - `qasper`：单样本 F1 最低相对下降 `-38.36`；
  - `multifieldqa_en`：单样本 F1 最低相对下降 `-50.59`；
  - `gov_report`：单样本 ROUGE-L 最低相对下降 `-8.81`；
  - `qmsum`：单样本 ROUGE-L 最低相对下降 `-6.13`。
- 这些差异不是简单标点差异，部分样本确实漏掉了关键限定信息或答错了局部事实。

### 调小跨步复用：r4/stale6 诊断结果

为了判断退化是否来自过长 stale，本轮只改变时间复用：

```text
attnpredict_reuse_steps=4
attnpredict_max_stale_steps=6
layer_reuse_stride=4
```

QA 结果：

| dataset | vanilla | r16 | r4 | r16 delta | r4 delta | r4 相比 r16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qasper` | 45.37 | 43.11 | 44.33 | -2.26 | -1.04 | +1.22 |
| `multifieldqa_en` | 58.25 | 56.90 | 57.74 | -1.35 | -0.51 | +0.84 |

摘要结果：

| dataset | vanilla | r16 | r4 | r16 delta | r4 delta | r4 相比 r16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gov_report` | 23.27 | 21.89 | 21.58 | -1.38 | -1.69 | -0.31 |
| `qmsum` | 17.86 | 17.07 | 18.53 | -0.79 | +0.67 | +1.46 |

结论：

- 调小 `reuse_steps` 对 `qasper`、`multifieldqa_en`、`qmsum` 有正向作用，说明当前质量风险至少部分来自跨步 stale。
- `gov_report` 上 r4 反而比 r16 略差，说明质量退化不是单一由 `reuse_steps=16` 决定；`layer_reuse_stride=4`、sparse view 本身、以及无全量校正都可能参与。
- 因为 r4 在目标吞吐上此前远低于最终 r16 配置，不能直接把默认值回退到 r4；但 r16 也不能再称为“质量已经充分可靠”。

### 当前判断

当前 `attnpredict-offload r16/stale16/stride4`：

- 性能上能在 `128k / bs=2 / output_len=64` 超过 vanilla；
- 质量上通过了检索任务和部分 QA/摘要均值，但在长答案 QA 和摘要中存在样本级明显波动；
- 因此它更适合称为“性能可用但质量仍需加固”的版本，不适合直接作为论文级稳定结论。

下一步建议按成本从低到高做：

1. 加一个可配置但默认不打开的 `layer_reuse_stride`，先测试 `stride=1/2/4` 的质量曲线，判断跨层复用是否是主要质量来源。
2. 测 `reuse_steps=8/12` 的质量与 DecTP 折中，当前已有证据显示 r4 可缓解部分 QA/qmsum 退化。
3. 实现诊断版 calibration：每 N 步 source layer 用 full view 收集一次 score 更新 `attn_history`，但先不改变输出 attention；若质量恢复明显，再评估是否让 calibration 同时参与输出。
4. 若 calibration 成本过高，优先做“只校正 predictor 历史、不强制全量输出”的轻量版本，避免把 decode 直接拉回 full attention。

## `reuse_steps=10, max_stale_steps=4` 诊断实验

用户希望验证更小 `max_stale_steps` 的性能表现。当前配置校验要求
`attnpredict_max_stale_steps >= attnpredict_reuse_steps`，因此本实验只临时放宽
`src/sparsevllm/config.py` 的校验为 `max_stale_steps > 0`，跑完后已恢复原校验。

实验命令：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
ATTNPREDICT_MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth \
LENGTHS=128000 BATCH_SIZES=2 OUTPUT_LEN=64 GPU_MEMORY_UTILIZATION=0.7 \
ATTNPREDICT_REUSE_STEPS=10 ATTNPREDICT_MAX_STALE_STEPS=4 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

smoke 结果：`16k / bs=1 / output_len=8`

| method | DecTP | TTFT | Mem |
| --- | ---: | ---: | ---: |
| vanilla | 15.91 tok/s | 0.93s | 66.11GB |
| attnpredict-offload `10/4` | 13.97 tok/s | 1.49s | 17.38GB |

目标结果：`128k / bs=2 / output_len=64`

| method | TTFT | PreTP | DecTP | ITL | Mem | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vanilla | 42.74s | 5989.9 tok/s | 46.58 tok/s | 42.94ms | 66.49GB | 1.00x |
| attnpredict-offload `10/4` | 51.53s | 4968.0 tok/s | 47.31 tok/s | 42.27ms | 48.62GB | 1.02x |

结论：`reuse=10, max_stale=4` 只比本轮 vanilla 略快，明显低于最终稳定
`reuse=16, max_stale=16` 的 `51.21 tok/s`。原因是当前实现只有在达到
`reuse_steps` 后才提交新预测；若此时 `max_stale_steps < reuse_steps`，新预测 future
一旦存在就会更容易触发强制消费，降低跨步复用隐藏 predictor/prefetch 的空间。
因此该组合不采纳为默认性能配置。质量上它理论上比 `16/16` 更新更积极，但仍需另跑
LongBench 才能判断是否值得用吞吐换质量。
