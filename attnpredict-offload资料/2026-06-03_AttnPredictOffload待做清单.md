# AttnPredict-Offload 待做清单

## 1. 复用 `tail_score` GPU buffer，减少分配开销和峰值碎片

> 核心思路：`tail_score` 仍然保留在 GPU 上，但不在每个 source layer 的 prefill predictor 初始化时反复新建，而是按最大需要尺寸复用已有 buffer。

### 当前开销

`tail_score` 是 prefill 最后一个 chunk 中用于初始化 predictor 的临时 score buffer，shape 为：

```python
(batch_size, num_heads, history_step, pooled_len)
```

在当前 `128k / bs=2` 配置下，近似为：

```text
batch_size = 2
num_heads = 32
history_step = 64
pooled_len = 128000 / 16 = 8000
dtype = float32
```

单个 `tail_score` 约：

```text
2 * 32 * 64 * 8000 * 4 bytes ≈ 131 MB
```

当前开启跨层复用，`layer_reuse_stride = 4`，32 层中只有 `0, 4, 8, ..., 28` 这些 source layer 会创建 `tail_score`，理论上最多约 8 个 source layer：

```text
131 MB * 8 ≈ 1.0 GB
```

实际运行中不一定 8 个长期同时存在，但在 prefill 最后阶段和异步 prefetch 任务持有 pending view 时，多个 `tail_score` 可能短时间重叠，占用显存并增加 CUDA allocator 的分配/释放压力。

### 为什么不能简单放 CPU

`tail_score` 是给 GPU attention kernel 写入的结果 buffer，后续 predictor 也在 GPU 上读取并运行 CNN。如果放到 CPU，会引入额外的 GPU -> CPU 和 CPU -> GPU 拷贝，反而更可能拖慢 TTFT。

因此更合理的方向是：

```text
保持 tail_score 在 GPU；
减少重复分配；
尽量复用已有 GPU buffer。
```

### 合理性

- 不改变 sparse token 选择，不改变 predictor 输入语义，理论上不影响输出质量；
- 目标是降低 prefill 阶段临时大 Tensor 的反复分配开销；
- 可能减少显存峰值碎片，降低 CUDA allocator 抖动；
- 对 `128k / bs=2` 这种长上下文场景更有意义，因为单个 `tail_score` 已经约 131 MB。

### 可能风险

- buffer 复用必须保证前一个异步 prefetch worker 已经消费完对应 `tail_score`，否则会出现写读覆盖；
- 不同 batch size、context length、`pooled_len` 可能变化，复用 buffer 需要正确处理“当前需要尺寸 <= 已分配尺寸”的切片视图；
- 如果为了安全加入过多同步，可能抵消复用收益；
- 该优化主要减少分配和显存碎片，未必显著减少 GPU attention kernel 本身或 GPU->CPU KV backing 拷贝开销，需要单独 profiler 验证。

### 验证方式

- 先做最小实现：每个 source layer 或全局维护可复用 `tail_score` buffer，只返回当前 shape 的切片 view；
- 跑 `.venv/bin/python -m py_compile src/sparsevllm/engine/cache_manager/attnpredict_offload.py`；
- 跑小规模 smoke benchmark，确认 prefill predictor 初始化和首个 decode 正常；
- 再用 `128k / bs=2 / output_len=16` 对比 TTFT、显存峰值、CPU backing 写入外的 profiler 区间；
- 如果出现异步读取覆盖、额外同步、TTFT 无收益或显存峰值无改善，则回退。

## 2. Source layer 组级独立异步 stream 实验

> 核心思路：当前所有 source layer 的 predictor / prefetch 共用一条异步 stream，后台任务会串行排队；可以尝试只为 source layer 复用组创建独立 stream，减少 source layer 之间的后台排队。

### 当前开销

当前代码中：

```python
self._prefetch_stream = torch.cuda.Stream(priority=0)
self._prefetch_streams = [self._prefetch_stream for _ in range(self.num_layers)]
```

这意味着所有 source layer 的后台任务都进入同一个 GPU stream。

例如 `layer_reuse_stride = 4` 时，真正提交 predictor 的 source layer 是：

```text
0, 4, 8, 12, ...
```

但它们的后台任务会排成一条队列：

```text
layer0 predictor/prefetch -> layer4 predictor/prefetch -> layer8 predictor/prefetch -> ...
```

如果 layer0 的后台任务较慢，layer4 即使已经到达可提交时机，也只能排在后面等待。

### 优化方法

只给 source layer 复用组创建独立异步 stream，而不是每层一个 stream：

```text
layer 0/1/2/3   -> stream0
layer 4/5/6/7   -> stream4
layer 8/9/10/11 -> stream8
```

实际只有 source layer 会提交 predictor 任务，复用层只是共享同组 lease 和 residency 结果。

### 合理性

- 符合跨层复用结构：只有 source layer 需要真正跑 predictor；
- 比“每层一条 stream”更克制，避免创建过多并发 stream；
- 可能减少 source layer 之间的后台任务排队；
- 不改变 hot token 选择逻辑，不改变 `reuse_steps` / `max_stale_steps` 质量预算。

### 可能风险

- 多条 stream 并发 predictor 可能抢占主计算流的 SM、显存带宽或 PCIe 带宽；
- H2D copy 并发可能导致单次 copy 更慢；
- 如果后台并发增加导致 main stream attention 变慢，decode 吞吐可能下降；
- 需要用 Nsight Systems 检查 overlap 和主流 kernel 是否被挤占，不能只看单次 benchmark。

### 验证方式

- 先做最小实现：按 `layer_reuse_stride` 给 source layer 组创建 stream；
- 跑 `.venv/bin/python -m py_compile src/sparsevllm/engine/cache_manager/attnpredict_offload.py`；
- 跑小规模 smoke benchmark，确认 future/event/lease cleanup 顺序正确；
- 对比 `128k / bs=2 / output_len=64` 的 DecTP、ITL 和 stale 等待情况；
- 用 Nsight Systems 对比单 stream 与组级 stream：
  - predictor/prefetch 是否减少排队；
  - main stream attention 是否被明显挤占；
  - H2D copy 是否出现带宽争用。
- 如果主流 kernel 被挤占或 DecTP 下降，则回退。

## 3. 跨机器复现与 CPU-GPU 链路敏感性验证

> 核心思路：当前 22 vCPU 机器上 `attnpredict-offload` 已超过 vanilla，但另一台同型号 GPU、25 vCPU 机器上 offload decode 明显低于 vanilla。后续需要确认差异来自机器的 CPU-GPU 数据通路、后台线程调度还是随机波动。

### 当前现象

相同代码、相同命令、相同主要配置：

```text
128k / bs=2 / output_len=64
reuse_steps=16
max_stale_steps=16
layer_reuse_stride=4
topk=4096 / sink=64 / recent=512
```

两台机器结果：

| 机器 | vanilla DecTP | attnpredict-offload DecTP | speedup |
| --- | ---: | ---: | ---: |
| 22 vCPU 机器 | 46.70 tok/s | 52.33 tok/s | 1.12x |
| 25 vCPU 机器 | 46.17 tok/s | 37.28 tok/s | 0.81x |

vanilla 基本一致，但 offload decode 单独下降，说明差异大概率不在 GPU attention 主计算，而在 offload decode 额外引入的 CPU backing、prefetch/event、H2D 搬运或线程调度链路。

### 合理性

- 本机同轮 benchmark 仍然有效：当前机器上 offload 和 vanilla 是同代码、同命令、同机对比；
- 但 offload 方法依赖 CPU backing 到 GPU active pool 的数据搬运，比 vanilla 更敏感；
- 如果后续要写论文或对外报告，需要补充多机复现或解释机器敏感性；
- 该项不改变算法和代码，只用于确认系统层瓶颈。

### 可能风险

- 云机器虽然显示同型号 GPU，但宿主机负载、CPU 频率、NUMA 拓扑、PCIe 链路状态可能不同；
- vCPU 数量更多不代表单线程延迟、内存带宽或 CPU-GPU 拷贝更好；
- 如果只凭一次 benchmark 下结论，可能把随机调度波动误认为方法问题。

### 验证方式

在两台机器分别记录硬件和链路信息：

```bash
lscpu
numactl -H
nvidia-smi topo -m
nvidia-smi -q -d CLOCK,PERFORMANCE,POWER,TEMPERATURE,PCI
```

每台机器重复跑 2-3 次目标 benchmark，确认结果是否稳定：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
ATTNPREDICT_MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth \
LENGTHS=128000 \
BATCH_SIZES=2 \
OUTPUT_LEN=64 \
GPU_MEMORY_UTILIZATION=0.7 \
ATTNPREDICT_REUSE_STEPS=16 \
ATTNPREDICT_MAX_STALE_STEPS=16 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

坏机器优先扫 CPU 线程数：

```bash
ATTNPREDICT_OFFLOAD_CPU_THREADS=1/2/4/8
```

如果低线程数改善 DecTP，优先怀疑 CPU 调度和线程争用；如果所有线程数都慢，优先怀疑 H2D/PCIe/NUMA 链路。进一步用 profiler 对比：

- `attnpredict_offload_prefetch_wait`
- `attnpredict_offload_prefetch_event_pending`
- `attnpredict_offload_ensure_positions_loaded`
- `attnpredict_offload_ensure_positions_resident`

若坏机器这些区间明显变长，则记录为 offload 系统链路敏感性，而不是当前机器性能结果失效。
