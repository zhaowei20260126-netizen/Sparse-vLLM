# 2026-05-29 decode predictor launch 诊断实验报告

## 结论

这次实验说明：attnpredict-offload 的 decode 慢，主要不是 packed view 或 sparse attention 本身导致的。

在保留 offload、resident KV 和 packed decode view 的情况下，只要不启动 decode 阶段的后台 predictor worker，`128k / bs=2 / output_len=64` 的 DecTP 从正常 offload 的 `21.05 tok/s` 回升到 `47.51 tok/s`，接近 vanilla 的 `46.66 tok/s`。

因此当前最大嫌疑是 decode predictor worker 的后台 GPU/CPU 工作影响了主 decode。它虽然没有明显让主 stream 显式等待，但很可能在 GPU 计算资源、显存带宽、CUDA 调度或 CPU 调度上与主计算竞争。

## 术语说明

- `DecTP`：decode throughput，解码吞吐。表示每秒生成多少 token。
- `ITL`：inter-token latency，token 间延迟。表示平均生成一个 token 的耗时。
- `with_score kernel`：带分数输出的 attention kernel。除了算 attention output，还会把 attention logits 写出来，作为 AttentionPredictor 输入。
- `decode predictor launch`：decode 阶段启动后台预测任务。代码里对应提交 `_predict_and_prefetch_worker`，它会做 sparse logits 后处理、CNN predictor、residency 计划和可能的 H2D prefetch。
- `resident KV`：驻留在 GPU active pool 的 K/V。offload 模式只把当前可见 token 的 KV 放在 GPU，其余保存在 CPU full backing。
- `packed view`：压缩后的 attention 读取视图。把当前层可见 token 的 GPU slots 打包成 `[batch, keep]`，交给 sparse decode kernel。

## 实验设计

本次用了两个临时代码开关，跑完后都已移除，没有保留到正式代码。

1. 关闭整条 decode 预测路径
   - 做法：让 `should_collect_decode_attn_score()` 返回 `False`。
   - 影响：不运行 with_score kernel，也不启动 decode predictor worker。
   - 目的：测 offload + packed view 的理论上限。

2. 只禁用 decode predictor launch
   - 做法：保留 with_score kernel，但在 `predict_next_mask()` 里不提交 `_predict_and_prefetch_worker`。
   - 影响：attention logits 仍会写出，但不跑后台 softmax/CNN/residency/H2D。
   - 目的：隔离后台 predictor worker 的影响。

注意：两个实验都会让 decode 长时间复用旧 lease，不满足质量假设，只用于性能诊断，不能作为正式 benchmark 策略。

## 目标配置结果

配置：`128k / bs=2 / output_len=64 / reuse_steps=4 / max_stale_steps=6 / gpu_memory_utilization=0.7`

| 模式 | DecTP | ITL | Mem | 说明 |
| --- | ---: | ---: | ---: | --- |
| vanilla | 46.66 tok/s | 42.86 ms | 66.49 GB | 无 offload |
| 正常 attnpredict-offload | 21.05 tok/s | 95.00 ms | 50.18 GB | 当前正式路径 |
| 关闭整条 decode 预测路径 | 49.89 tok/s | 40.09 ms | 50.19 GB | 不写 score，不启动 worker |
| 只禁用 decode predictor launch | 47.51 tok/s | 42.10 ms | 50.32 GB | 写 score，但不启动 worker |

## Profiler 对比

配置：`128k / bs=2 / output_len=32 / PROFILER_SVLLM=1`

| 模式 | DecTP | model_run_decode | model_run_model_decode | build_decode_view | predictor 相关 |
| --- | ---: | ---: | ---: | ---: | --- |
| 正常 attnpredict-offload | 19.85 tok/s | 99.60 ms/step | 94.44 ms/step | 0.60 s total | `predict_next_positions` 3.13 s total |
| 关闭整条 decode 预测路径 | 44.07 tok/s | 44.22 ms/step | 41.97 ms/step | 0.24 s total | decode predictor 计时消失 |
| 只禁用 decode predictor launch | 41.88 tok/s | 46.58 ms/step | 43.46 ms/step | 0.27 s total | decode predictor 计时消失，`sparse_prepare_attn_score` 仍存在 |

## 判断

1. `packed view` 和 sparse decode kernel 不是当前 21 tok/s 的根因。  
   关闭后台 predictor 后，offload decode 能达到 41 到 50 tok/s。

2. `with_score kernel` 有开销，但不是主因。  
   关闭整条 decode 预测路径是 49.89 tok/s，只禁用 launch 是 47.51 tok/s，差距约 2.38 tok/s。

3. 最大嫌疑是 `_predict_and_prefetch_worker` 的后台工作。  
   它不一定表现为 `prefetch_wait`，但可能通过 GPU stream 资源竞争、CNN forward、scatter/pooling kernel、H2D prefetch 或 CPU 调度拖慢主 decode。

## 下一步建议

下一步应该继续拆 `_predict_and_prefetch_worker`：

1. 保留 worker，但跳过 CNN forward，只做 event 和空任务，测 CUDA stream 调度/线程池开销。
2. 只跑 `_predict_next_positions_sync`，但不做 `_ensure_layer_results_resident`，区分 predictor 计算和 residency/H2D。
3. 只做 residency/H2D，复用旧 hot positions，区分 prefetch 搬运和 CNN 计算。
4. 如果能安装 `nsys`，用 Nsight Systems 看主 stream 和 prefetch stream 的 GPU timeline。

