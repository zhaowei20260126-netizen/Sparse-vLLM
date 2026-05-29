# 2026-05-29 `_predict_and_prefetch_worker` 拆解报告

## 结论

`_predict_and_prefetch_worker` 的主要 decode 性能问题来自 predictor 计算，而不是 worker 调度，也不是 H2D prefetch 本身。

在 `128k / bs=2 / output_len=64` 下：

- 正常 attnpredict-offload：`21.45 tok/s`
- `empty`：`45.67 tok/s`
- `predict_only`：`17.90 tok/s`
- `residency_only`：`35.97 tok/s`

这说明：

1. 空 worker、CUDA event 和 with_score kernel 不是主因。
2. 只跑 predictor、不做 residency/H2D 时，速度已经降到 `17.90 tok/s`。
3. 只做 residency/H2D 也会降速，但降到 `35.97 tok/s`，影响明显小于 predictor。
4. Nsight Systems 显示 decode 窗口里 8 条非主 CUDA stream 上有大量 predictor kernel，最大单项是 `AdaptiveAvgPool2d` 对应的 `adaptive_average_pool` kernel。

## 术语说明

- `worker`：后台任务。这里指 `_predict_and_prefetch_worker`，负责等 attention score 写完、预测下一步 hot tokens、准备下一步 GPU resident KV。
- `predictor`：AttentionPredictor 的 CNN 预测器。它根据 attention 历史预测下一步要保留的 block。
- `residency/H2D`：GPU 驻留计划和 Host-to-Device 拷贝。即决定哪些 KV 留在 GPU，哪些 KV 从 CPU 拷回 GPU。
- `CUDA stream`：CUDA 流。GPU 上的任务队列；不同 stream 可以并发或交错执行，但会共享 GPU 计算资源和显存带宽。
- `with_score kernel`：带分数输出的 attention kernel。除了输出 attention result，还写出 attention logits 给 predictor。
- `Nsight Systems / nsys`：NVIDIA 系统级 GPU timeline 工具。用于观察不同 CUDA stream 上的 kernel 是否重叠、是否抢资源。

## 临时实验模式

本次临时修改了 `_predict_and_prefetch_worker`，实验后已撤销。

| 模式 | 含义 | 目的 |
| --- | --- | --- |
| normal | 原始 worker | 基线 |
| empty | 等 attention event，然后直接返回旧 lease | 测 worker 调度和 event 开销 |
| predict_only | 跑 `_predict_next_positions_sync`，丢弃预测结果，不做 residency/H2D | 测 predictor 计算影响 |
| residency_only | 不跑 predictor，复用旧 hot positions，只做 residency/H2D | 测 KV 驻留和预取影响 |

注意：后三个模式都会破坏正常质量假设，只用于性能定位。

## Python profiler 结果

配置：`128k / bs=2 / output_len=64 / PROFILER_SVLLM=1`

| 模式 | DecTP | ITL | model_run_decode | model_run_model_decode | 关键 worker 计时 |
| --- | ---: | ---: | ---: | ---: | --- |
| normal | 21.45 tok/s | 93.23 ms | 92.79 ms/step | 89.46 ms/step | `worker_predict_total=6.89s`，`worker_residency_total=0.74s` |
| empty | 45.67 tok/s | 43.80 ms | 43.32 ms/step | 41.89 ms/step | 只有 event/旧 lease，worker 计时很小 |
| predict_only | 17.90 tok/s | 111.72 ms | 111.20 ms/step | 108.24 ms/step | `worker_predict_total=8.75s`，无 residency/H2D |
| residency_only | 35.97 tok/s | 55.60 ms | 55.11 ms/step | 53.46 ms/step | `worker_residency_total=1.50s`，无 predictor |

解读：

- `predict_only` 比 normal 更慢，是因为它跑 predictor 但不真正更新 active set，触发次数也更多；但这反而证明 predictor 计算本身足以压垮主 decode。
- `residency_only` 的 H2D/dirty writeback 有成本，但不是最大项。
- `empty` 接近 vanilla 速度，说明 packed decode view 和 with_score 路径不是当前 21 tok/s 的根因。

## Nsight Systems 结果

可用的 nsys 路径：

```bash
/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
```

Ubuntu 源安装的 `/usr/lib/nsight-systems/bin/nsys` 是 2021.3，运行目标程序时触发 `libToolsInjectionProxy64.so` 的 `GLIBC_PRIVATE` 符号错误，未使用。

生成文件：

```text
profiler_outputs/nsys_attnpredict_offload_normal_128k_o64.nsys-rep
profiler_outputs/nsys_attnpredict_offload_normal_128k_o64.sqlite
profiler_outputs/nsys_attnpredict_offload_normal_128k_o64_decode_window.nsys-rep
profiler_outputs/nsys_attnpredict_offload_normal_128k_o64_decode_window.sqlite
```

decode 窗口命令使用 `--delay=63 --duration=8`，覆盖正常 worker 的 128k decode 区间。

### CUDA stream 摘要

decode 窗口内：

| stream | kernel time |
| ---: | ---: |
| main stream 7 | 2659 ms |
| predictor stream 13 | 452 ms |
| predictor stream 29 | 450 ms |
| predictor stream 93 | 443 ms |
| predictor stream 61 | 439 ms |
| predictor stream 77 | 436 ms |
| predictor stream 45 | 420 ms |
| predictor stream 109 | 412 ms |
| predictor stream 125 | 388 ms |

8 条非主 stream 的 kernel time 总计约 `3.44s`。这不是主 stream 显式 wait，而是后台 predictor kernel 和主 decode kernel 共享 GPU 资源。

### 非主 stream 主要 kernel

非主 stream 最大开销：

- `adaptive_average_pool`：每条 predictor stream 大约 `174-241 ms`，总计约 `1.70s`
- `CUDAFunctor_add<Half>`：每条约 `53-57 ms`
- `launch_clamp_scalar`：每条约 `46-47 ms`
- cuDNN layout/conv kernels：`nhwcToNchw`、`nchwToNhwc`、`implicit_convolve_sgemm`、`xmma_fprop`

这和模型结构吻合：`AttnPredictCNN` 里有 `Conv2d -> ReLU -> Conv2d -> ReLU -> AdaptiveAvgPool2d((1, None)) -> Conv1d`。  
其中 `AdaptiveAvgPool2d((1, None))` 实际上是在 history 维做平均，但当前 PyTorch/CUDA 实现非常重。

### stream overlap

非主 stream 和 main stream 有部分重叠。例如：

| stream | non-main kernel time | overlap with main stream |
| ---: | ---: | ---: |
| 13 | 452 ms | 62 ms |
| 29 | 450 ms | 49 ms |
| 93 | 443 ms | 39 ms |
| 61 | 439 ms | 38 ms |
| 77 | 436 ms | 36 ms |
| 45 | 420 ms | 37 ms |
| 109 | 412 ms | 32 ms |
| 125 | 388 ms | 54 ms |

这说明 predictor stream 确实异步执行，但不是“免费 overlap”。它们在 decode 窗口持续占用 GPU，部分与主 stream 重叠，部分插在主 stream kernel 间隙里，都会拉长主 decode 的 wall time。

## 关键判断

当前问题不是“异步流没有异步”。  
更准确地说：预测确实在异步 stream 上跑，但它的 GPU kernel 太重，且有多个 layer-reuse source stream 同时排队，导致 GPU 资源被 predictor 抢走。

尤其注意：`_cnn_lock` 只串行化 Python 侧的 CNN forward 调用，不等于串行化 GPU 执行。  
因为 CUDA kernel launch 是异步的，线程拿到锁后把 kernel 发到自己的 stream，很快释放锁；多个 stream 上的 CNN kernel 仍然会在 GPU 上重叠或交错执行。

## 下一步最值得做

1. 优先优化 `AttnPredictCNN.forward()` 里的 `AdaptiveAvgPool2d((1, None))`。  
   这个操作等价于对 history 维求平均，可以实验替换为更轻的 `mean(dim=2, keepdim=True)` 或专门的 reduce kernel。它没有参数，不影响 checkpoint 结构。

2. 给 predictor 使用单独低优先级 CUDA stream。  
   当前每个层复用组都有自己的 prefetch stream，predictor kernel 会分散到 8 条 stream 上。可以实验让 CNN forward 走一个 shared low-priority predictor stream，H2D 仍留在 per-layer prefetch stream。

3. 串行化 GPU predictor 执行，而不是只串行化 Python launch。  
   如果低优先级 stream 不够，可以让 predictor stream 上的 kernel 严格排队，牺牲预测新鲜度，换主 decode 稳定性。

4. 如果第 1 点收益明显，再考虑 residency/H2D。  
   `residency_only` 仍有 35.97 tok/s，说明这块也有优化空间，但优先级低于 predictor。

