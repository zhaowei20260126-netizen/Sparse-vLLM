# AttnPredict-Offload 最终技术报告

日期：2026-05-30

## 1. 结论摘要

Sparse-vLLM（稀疏版 vLLM 推理引擎，即在 vLLM 风格推理框架里加入稀疏注意力和缓存管理）当前的 `attnpredict-offload` 已在目标条件下超过 `vanilla`（全量 attention 基线，即每步 decode 读取完整 KV cache 的普通实现）。

目标条件：

- 模型：`llama-3.1-8B-Instruct`
- 上下文长度：`128000`
- batch size：`2`
- output length：`64`
- GPU memory utilization：`0.7`
- top-k/sink/recent：`4096 / 64 / 512`

最终同轮 benchmark：

| method | TTFT | PreTP | DecTP | ITL | Mem | Decode speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vanilla | 42.75s | 5988.3 tok/s | 46.56 tok/s | 42.96ms | 66.49GB | 1.00x |
| attnpredict-offload | 53.17s | 4815.2 tok/s | 51.21 tok/s | 39.06ms | 48.74GB | 1.10x |

DecTP/decode throughput（解码吞吐，即 decode 阶段每秒生成 token 数）从约 `24.45 tok/s` 提升到 `51.21 tok/s`，并超过 vanilla 的 `46.56 tok/s`。显存从 vanilla 的 `66.49GB` 降到 `48.74GB`。

## 2. 核心概念

KV cache（Key/Value 缓存，即 Transformer attention 中历史 token 的 K/V 张量缓存）：decode 每生成一个 token 都要读取历史 KV。长上下文下，完整 KV cache 很大，是显存和带宽压力来源。

AttentionPredictor（注意力预测器，即用 CNN 根据历史 attention 模式预测下一步重要 token 的模块）：本项目用它预测下一步哪些历史 token 更可能被关注。

offload（卸载，即把完整 KV 放在 CPU backing 里，只把当前需要读的一小部分 KV 放到 GPU）：`attnpredict-offload` 用 CPU 保存完整历史 KV，用 GPU active pool 保存当前稀疏可见 token。

CUDA stream（CUDA 流，即 GPU 上异步排队执行 kernel/copy 的队列）：本项目用独立 stream 做 predictor 和 H2D copy，尽量与主 attention 计算重叠。

kernel（GPU 内核，即在 GPU 上执行的一段并行程序）：decode attention、block pooling、slot gather 等热点操作最终都体现为 GPU kernel 或张量算子。

prefetch（预取，即在下一步真正需要 KV 前提前把预测出的 KV 从 CPU 搬到 GPU）：可以隐藏 CPU gather 和 H2D copy 的一部分延迟。

hot tokens（热点 token，即 AttentionPredictor 预测出的重要历史 token）：这些 token 会和 sink/recent/current 一起组成下一步 decode 可见集合。

packed view（打包读视图，即把每个请求实际要读的 GPU slot 压成二维表）：decode kernel 读 `packed_slots[b, j]`，避免扫完整 row/position 映射。

lease（租约，即一份 predictor 结果被复用的有效区间）：同一份 hot tokens 可以跨多个 decode step 使用，直到刷新或超过 stale 限制。

residency（驻留状态，即某个逻辑 token 的 KV 当前是否在 GPU active pool）：cache manager 维护 CPU/GPU 映射和 GPU slot 生命周期。

score buffer（分数缓冲，即 attention kernel 写出的注意力 logits/score）：AttentionPredictor 用这些分数更新历史并预测下一步 hot tokens。

## 3. 最终配置

代码默认值和 benchmark 默认值：

| 参数 | 最终值 | 含义 |
| --- | ---: | --- |
| `attnpredict_reuse_steps` | 16 | 跨步复用步数；每 16 个 decode step 刷新一次 predictor 结果 |
| `attnpredict_max_stale_steps` | 16 | 最大陈旧步数；后台预测未完成时最多复用旧 lease 到 16 步 |
| `layer_reuse_stride` | 4 | 跨层复用跨度；每 4 层只让组首层跑 predictor，组内层复用结果 |
| `num_top_tokens` | 4096 | 预测保留 token 预算 |
| `num_sink_tokens` | 64 | sink token 数；序列开头固定保留 |
| `num_recent_tokens` | 512 | recent token 数；最近 token 固定保留 |
| `attnpredict_history_steps` | 64 | predictor 使用的历史步数 |
| `attnpredict_pooling_block_size` | 16 | block pooling 的 token 块大小 |
| `attnpredict_offload_prefetch` | true | 开启异步预取 |
| `attnpredict_offload_cpu_threads` | 8 | CPU worker 线程数 |
| `attnpredict_offload_cpu_slots` | -1 | 自动决定 CPU backing slot 数 |
| `attnpredict_offload_cpu_memory_utilization` | 0.70 | CPU backing 可用内存比例 |
| `attnpredict_offload_pin_staging` | true | H2D copy 前使用 pinned memory staging |

最终目标 benchmark 命令：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
ATTNPREDICT_MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth \
LENGTHS=128000 BATCH_SIZES=2 OUTPUT_LEN=64 GPU_MEMORY_UTILIZATION=0.7 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

## 4. 采纳的优化点

### 4.1 跨步复用：`reuse_steps=16, max_stale_steps=16`

目标瓶颈：每层每步都跑 predictor 会带来额外 GPU 负载、softmax、block pooling、top-k 和 prefetch 调度成本。

做法：把 `attnpredict_reuse_steps` 从 `4` 调到 `16`，同时把 `attnpredict_max_stale_steps` 从 `6` 调到 `16`。

为什么会快：predictor 刷新频率降低，score 写出、CNN forward、top-k、lease 切换和预取调度次数下降。

质量影响：理论上预测新鲜度降低，但 LongBench 英文 A/B 中 40 条样本输出完全一致。它不是无限 stale 或极端参数作弊。

### 4.2 跨层复用：`layer_reuse_stride=4`

目标瓶颈：相邻层每层独立跑 predictor，重复开销高。

做法：每 4 层为一组，只让组首层收集 score 并跑 predictor，组内其它层复用组首层 hot tokens 和 lease。

为什么会快：理论 predictor 触发层数从 32 层降到 8 组，减少 score buffer 消费、CNN forward 和 prefetch 任务数量。

质量影响：不同层 attention 模式并不完全相同，因此这是有质量假设的优化；当前与 `top-k=4096/sink=64/recent=512` 配合，LongBench 小样本未发现输出变化。

### 4.3 `torch.compile` 编译 CNN predictor

`torch.compile`（PyTorch 图编译，即把 PyTorch eager 执行转换为更少调度开销的编译图）：用于 CNN predictor。

目标瓶颈：CNN predictor 在 decode 中频繁执行，eager 模式下 kernel launch 和 Python 调度成本高。

做法：对 `self.cnn` 使用 `torch.compile(dynamic=True, options={"triton.cudagraphs": False})`，并在初始化时做 max-width dummy prewarm。

为什么会快：减少 predictor forward 的 Python 调度和小 kernel 开销；prewarm 避免首次真实请求时承担编译延迟。

质量影响：不改变模型权重和数学逻辑，只改变执行方式。

### 4.4 单条默认优先级 predictor/prefetch stream

目标瓶颈：旧版每层一个 CUDA stream，多个后台 predictor/prefetch 可能抢占主 stream，造成调度和资源竞争。

做法：改成所有层共享一条默认优先级 `torch.cuda.Stream(priority=0)`。

为什么会快：减少 stream 数量和并发竞争，使 predictor/prefetch 更像有序后台流水，而不是多层同时抢 GPU。

质量影响：不改变 token 选择，只改变任务排队方式。

### 4.5 `decode_view_max_len()` 避免每层 `.item()` 同步

`.item()` 同步（GPU 到 CPU 同步，即 CPU 读取 GPU 标量时会等待 GPU 完成相关工作）：在 decode hot path 中非常贵。

目标瓶颈：attention decode 原来每层用 `context_lens.max().item()` 计算最大长度，会引入 GPU/CPU 同步。

做法：增加 cache-manager hook：`decode_view_max_len()`。`attnpredict-offload` 在 `build_decode_view()` 的 CPU 侧已经知道 `max_keep`，直接缓存并返回。

为什么会快：减少每层 decode 的 CPU 等 GPU，同步点更少。

质量影响：不改变可见 token 集合，只改变长度获取方式。

### 4.6 packed slots 缓冲复用和 local req indices 缓存

目标瓶颈：每层 `build_decode_view()` 重复分配 `packed_slots` 和 `torch.arange`。

做法：为每层缓存 `packed_slots` buffer，并缓存 local req indices。

为什么会快：减少 decode hot path 的 GPU allocation 和小张量创建。

质量影响：只复用存储空间，`view_lens` 限定有效区域，不改变行为。

### 4.7 decode view 保存 CPU metadata

目标瓶颈：`predict_next_positions` 中反复从 GPU tensor 取 `req_indices/view_lens/full_context_lens`，会触发 `.item()` 或 D2H 同步。

做法：`build_decode_view()` 同时保存 CPU 侧 metadata：`req_indices_cpu`、`view_lens_cpu`、`full_context_lens_cpu`。

为什么会快：predictor 后处理直接读 CPU 列表，减少 GPU 标量同步。

质量影响：metadata 与 GPU view 同源，不改变选择逻辑。

### 4.8 block pooling 和 top-k 小优化

block pooling（块级池化，即把 token attention 分数按固定 token block 聚合）：用于把 token 级 attention 压成 predictor 使用的 block 分数。

目标瓶颈：`logits -> attention -> block score -> top-k tokens` 里存在小张量创建和同步。

做法：

- 缓存 `_pooling_offsets_gpu`，避免每次创建 `torch.arange(block_size)`；
- 去掉 `valid.any()` 这类会同步的判断；
- 在 top-k block 已互异的场景，用 `.sort().values` 替代更重的 `torch.unique`。

为什么会快：减少小 kernel、小张量和潜在同步。

质量影响：top-k block 本身互异，排序不改变 token 集合。

### 4.9 `_commit_lease()` 按 row 合并

目标瓶颈和正确性问题：LongBench 连续 batching 中，新 row 单独完成 prefill 后提交 lease，会覆盖整层 `_lease_hot_positions[layer]`，误删仍在 decode 的旧 row，触发 `KeyError: 1`。

做法：`_commit_lease()` 改为按 row 更新 dict，而不是替换整层 dict。

为什么重要：保证混合 prefill/decode、连续 batching 下 lease 生命周期正确。

质量影响：这是正确性修复，不改变固定 batch 的预测策略。

## 5. 已验证但未采纳/已回退的方向

| 方向 | 结果 | 结论 |
| --- | ---: | --- |
| fused `logits -> block pooled attention` runtime kernel | 正确但更慢 | 回退 |
| 同层 batch CNN | 约 24.25 tok/s | 低于稳定版，回退 |
| CPU CNN predictor | 约 0.43 tok/s | 极慢，回退 |
| `max_stale_steps=12` 且 reuse 仍小 | 约 24.92 tok/s | 收益太小，质量预算不采纳 |
| `num_top_tokens=2048` | 约 25.32 tok/s | 小幅提速但压缩质量预算，不采纳 |
| 非 offload `attnpredict` | 约 3.11 tok/s | 不可替代 |
| `layer_reuse_stride=8` | 41.69 tok/s | 未超过 vanilla |
| `layer_reuse_stride=16` | 44.71 tok/s | 未超过 vanilla，质量风险更高 |
| `layer_reuse_stride=32` | 45.96 tok/s | 仍未超过 vanilla，质量风险过高 |
| GPU kernel 下沉 `build_decode_view` slot gather + recent/current 拼接 | output16 最高 24.23 tok/s | 端到端负收益，回退 |
| split static slots + recent ring 进 attention kernel | 更慢 | 回退 |
| full-resident fast path / chunk reclamation | 收益很小 | 不保留 |
| incremental packed view append | 更慢 | 回退 |
| `torch.compile` max-autotune predictor | 微基准好，端到端 overlap 差 | 不采纳 |

## 6. 关键实验结果

### 6.1 从稳定版到最终版的主要吞吐变化

| 阶段 | DecTP | 说明 |
| --- | ---: | --- |
| vanilla 基线 | 约 46.8 tok/s | 目标要超过它 |
| 初始稳定 attnpredict-offload | 约 24.45 tok/s | 显存约 49.7GB |
| compiled CNN + stride4 + reuse4 | 34.40 tok/s | runtime 优化后明显提升 |
| reuse8/stale8 output64 | 44.65 tok/s | 接近 vanilla |
| reuse12/stale12 单跑 | 47.92 tok/s | 单跑超过，但同轮余量不足 |
| reuse16/stale16 单跑 | 50.50 tok/s | 有稳定余量 |
| 最终同轮 benchmark | 51.21 tok/s | 超过 vanilla 46.56 tok/s |

### 6.2 跨步/跨层复用隔离实验

旧提交 `3b2071d9591fe9d085dccd5a2a35af5c33245ccd` 隔离验证：

| 实验 | DecTP | 结论 |
| --- | ---: | --- |
| 旧提交只改 `reuse/max_stale=16/16` | 10.42 tok/s | 不能复现当前效果 |
| 旧提交临时补 `layer_reuse_stride=4` + `16/16` | 22.22 tok/s | 跨层复用有效，但远不够 |
| 当前最终实现 `layer_reuse_stride=4` + `16/16` + runtime 优化 | 51.21 tok/s | 说明 runtime 优化是必要组成 |

结论：最终收益不是单纯靠增大复用步数，也不是只靠跨层复用；必须叠加 predictor 编译、stream 调度、同步削减、packed view 构造和 block pooling 固定成本优化。

### 6.3 质量 A/B

LongBench（长上下文评测集，即用于测试长文本理解/检索/问答的 benchmark）小样本 greedy decode 对比：

| 数据集 | 样本数 | reuse=4 | reuse=12 | reuse=16 | reuse=16 与 reuse=4 完全相同输出 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hotpotqa` | 20 | F1 45.27 | F1 45.27 | F1 45.27 | 20/20 |
| `passage_retrieval_en` | 20 | retrieval 100.00 | retrieval 100.00 | retrieval 100.00 | 20/20 |

说明：`.venv` 缺少 `jieba`，按不下载依赖约束，没有跑完整中文 LongBench 官方 eval；英文任务使用仓库同等 metric 逻辑本地评分。论文级结论仍应补完整 LongBench/SCBench。

## 7. 架构约束落实情况

cache-manager-first（缓存管理器优先架构，即方法特有状态和调度放在 cache manager，而不是塞进 attention 层）：已遵守。

具体落点：

- 方法核心状态仍在 `src/sparsevllm/engine/cache_manager/attnpredict_offload.py`；
- `attention.py` 只调用通用 hook，例如 `build_decode_view()` 和 `decode_view_max_len()`；
- predictor、lease、residency、prefetch、packed view 构造都由 cache manager 管理；
- 未保留无端到端收益的实验 kernel 或复杂兜底；
- 没有用极端 top-k、无限 stale、极大 reuse 作弊换速度。

## 8. 验证命令

编译检查：

```bash
.venv/bin/python -m py_compile \
  src/sparsevllm/config.py \
  src/sparsevllm/engine/cache_manager/attnpredict_offload.py \
  src/sparsevllm/engine/cache_manager/base.py \
  src/sparsevllm/layers/attention.py
```

benchmark 脚本语法检查：

```bash
bash -n scripts/bench_attnpredict_vs_vanilla_128k.sh
```

最终目标 benchmark：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
ATTNPREDICT_MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth \
LENGTHS=128000 BATCH_SIZES=2 OUTPUT_LEN=64 GPU_MEMORY_UTILIZATION=0.7 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

## 9. 最可信瓶颈与后续方向

当前 DecTP 已超过 vanilla，但 TTFT 更差，prefill 仍慢。最可信剩余瓶颈：

- prefill 阶段 CPU full backing 写入导致 TTFT 高；
- decode 仍有 per-layer Python/cache-manager 调度和 packed view 构造成本；
- predictor 虽降低频率，仍在后台消耗 GPU；
- H2D prefetch 和 event 消费仍会在某些层形成等待。

下一步更值得做的方向：

- 在不改变质量预算的前提下，进一步减少 `build_decode_view()` 的 per-layer Python 成本；
- 对 CPU backing 写入和 dirty KV 写回做更粗粒度批处理；
- 用更可靠的端到端 profiler 定位剩余同步点，而不是继续盲目融合 kernel；
- 补完整 LongBench/SCBench 质量报告，验证 `16/16 + stride4` 的质量稳定性。

## 10. 追加优化实验：view 构造、CPU backing 和端到端 profiler

本轮目标是继续验证三个方向：

- `build_decode_view()`（构造 decode 稀疏读视图，即把每层要读的 token slot 打包给 attention kernel） 的 per-layer Python 成本；
- CPU backing（CPU 完整 KV 后备存储，即 GPU 驱逐后可从 CPU 恢复的 KV）写入是否能做更粗粒度批处理；
- Nsight Systems/NVTX（NVIDIA 系统级时间线分析器/代码区间标记，即查看 CUDA API、kernel、memcpy 和自定义 range 的端到端 profiler）定位剩余同步点。

### 10.1 profile 基线

命令形态：`128k / bs=2 / output_len=16`，只跑 `attnpredict-offload`，开启 `PROFILER_SVLLM=1`。

结果：

| 指标 | 数值 |
| --- | ---: |
| TTFT | 53.20s |
| DecTP | 50.49 tok/s |
| Mem | 48.60GB |
| `attnpredict_offload_store_cpu_full_kv` | 50.34s，总 1056 次，47.67ms/次 |
| `attnpredict_offload_build_decode_view` | 0.0830s，总 480 次，0.1729ms/次 |

结论：decode view 有优化空间，但最大瓶颈仍是 prefill 阶段 GPU 到 CPU 的 KV backing 写入。

### 10.2 失败实验：prefill KV 完全延迟写回

尝试策略：prefill 阶段不立即把 K/V 写入 CPU backing，只在首次 shrink 驱逐 dirty KV 时批量写回。

预期收益：减少 prefill 每层每 chunk 的 D2H copy 和 CPU `index_copy_`。

实际结果：

| 指标 | 数值 |
| --- | ---: |
| TTFT | 56.57s |
| DecTP | 5.08 tok/s |
| `attnpredict_offload_dirty_d2h_writeback` | 36.09s |
| `attnpredict_offload_prefetch_wait` | 2.47s |

结论：回退。原因是 D2H 写回从 prefill 分散写入变成首轮 shrink/lease 消费时集中爆发，直接阻塞 decode。这个方向不能保留。

### 10.3 采纳实验：更轻的 `build_decode_view()`

改动：

- recent slots 从 `np.arange + advanced indexing` 改成连续切片 `mirror[row, recent_start:full_len]`；
- 为 source layer 复用 `packed_positions` buffer，减少每次分配；
- 继续复用 `packed_slots` 和 local req indices。

结果：

| 指标 | 修改前 | 修改后 |
| --- | ---: | ---: |
| `build_decode_view` 平均耗时 | 0.1729ms/layer | 0.1614ms/layer |
| output16 DecTP | 50.49 tok/s | 52.53 tok/s |

结论：采纳。该改动不改变可见 token 集合，只减少 view 构造开销。

### 10.4 采纳实验：CPU backing 连续段写入

观察：初始 prefill 中，CPU slots 通常按每个序列 chunk 连续分配。原实现即使 slots 连续，也走 `index_copy_`。

改动：在 `_prepare_prefill()` 记录连续 CPU slot segment；`on_kv_stored()` 中如果本批 slots 可表示为连续段，就对 `cpu_kv_cache` 做切片 `copy_`，不连续时才回退 `index_copy_`。

结果：

| 指标 | 修改前 | 修改后 |
| --- | ---: | ---: |
| TTFT | 53.20s | 52.56s |
| PreTP | 4812.18 tok/s | 4870.42 tok/s |
| DecTP | 50.49 tok/s | 52.61 tok/s |
| `store_cpu_full_kv` 平均耗时 | 47.67ms/次 | 47.05ms/次 |

结论：采纳。收益不大，但端到端为正；不改变 KV 内容和稀疏选择质量。

### 10.5 Nsight Systems 结论

可用工具路径：

```bash
/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
```

另一个系统路径 `/usr/lib/x86_64-linux-gnu/nsight-systems/target-linux-x64/nsys` 因 GLIBC 符号不兼容失败，未采用。

Nsight Systems 生成文件：

- `profiler_outputs/nsys_attnpredict_after_view_128k_bs2_o16.nsys-rep`
- `profiler_outputs/nsys_attnpredict_after_view_128k_bs2_o16.sqlite`
- `profiler_outputs/nsys_attnpredict_after_view_128k_bs2_o16_stats_*.csv`

Nsight tracing overhead 较大，吞吐数字不可直接和普通 benchmark 比；但定位结论清楚：

| Nsight report | 关键结果 |
| --- | --- |
| CUDA API summary | `cudaMemcpyAsync` 占 CUDA API 时间 98.3% |
| GPU mem time summary | D2H 占 GPU memcpy 时间 84.3%，H2D 占 15.7% |
| NVTX summary | `attnpredict_offload_store_cpu_full_kv` 仍是最大自定义区间 |

结论：剩余最大瓶颈不是盲目融合 attention kernel 可以解决的，而是 prefill 阶段 CPU backing 的大规模 D2H 写入。延迟写回已验证负收益，更可信的后续方向是改变 CPU backing 布局或减少必须写回的 KV 量，而不是把写回推迟到 decode 临界路径。

### 10.6 追加 full benchmark

命令：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
ATTNPREDICT_MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth \
LENGTHS=128000 BATCH_SIZES=2 OUTPUT_LEN=64 GPU_MEMORY_UTILIZATION=0.7 \
SPARSEVLLM_MASTER_PORT=2361 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

结果：

| method | TTFT | PreTP | DecTP | ITL | Mem | Decode speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vanilla | 42.85s | 5974.0 tok/s | 45.41 tok/s | 44.04ms | 66.49GB | 1.00x |
| attnpredict-offload | 51.81s | 4941.4 tok/s | 50.99 tok/s | 39.22ms | 48.64GB | 1.12x |

结论：追加优化后仍稳定超过 vanilla。相比之前 `51.21 tok/s`，full benchmark DecTP 基本同档；新增改动主要改善小 profile 下的 view 构造和 prefill CPU 写入细节，不是数量级变化。
