# 2026-05-29 GPU predictor 后续优化与 nsys 报告

## 结论

本轮继续优化 GPU predictor，而不是把 predictor 放到 CPU。

保留了两个行为等价的 CNN 优化：

1. `AdaptiveAvgPool2d((1, None))` 改为 `mean(dim=2, keepdim=True)`。
2. Conv2d 路径使用 `channels_last` 布局，减少 cuDNN layout 转换。

目标配置 `128k / bs=2 / output_len=64 / reuse=4 / max_stale=6` 下：

| 版本 | DecTP | predictor 计时 |
| --- | ---: | ---: |
| mean 之前正常路径 | 约 21.45 tok/s | `predict_next_positions` 约 6.88s |
| mean-only | 24.04 tok/s | `predict_next_positions` 约 3.35s |
| mean + channels_last | 24.53 tok/s | `PROFILER_SVLLM=1` 下 `predict_next_positions` 约 2.79s |

`channels_last` 端到端收益不大，但 profiler 中 predictor 总时长继续下降，因此保留。

## 等价性验证

用同一个 checkpoint 同时构造旧 pool 版 CNN 和当前 mean + channels_last 版 CNN。

输入覆盖：

- `pooled_len=256`
- `pooled_len=2000`
- `pooled_len=8000`，对应 128k 长度按 block size 16 池化

输入分布覆盖：

- uniform
- spiky
- decay

结果：

```text
max_abs_diff = 0
mean_abs_diff = 0
min_topk_overlap = 1.000000
avg_topk_overlap = 1.000000
```

说明当前优化没有改变 CNN logits，也没有改变 top-k hot block 选择。

## nsys 观察

生成文件：

```text
profiler_outputs/nsys_attnpredict_offload_mean_128k_o64_decode_window.nsys-rep
profiler_outputs/nsys_attnpredict_offload_mean_128k_o64_decode_window.sqlite
profiler_outputs/nsys_attnpredict_offload_mean_cl_128k_o64_decode_window.nsys-rep
profiler_outputs/nsys_attnpredict_offload_mean_cl_128k_o64_decode_window.sqlite
```

mean-only 之后，原先最大的 `adaptive_average_pool` kernel 消失。

新的非主 stream 主要耗时变为：

- `CUDAFunctor_add<Half>`：Conv bias add
- `launch_clamp_scalar`：ReLU
- cuDNN Conv2d kernel
- cuDNN layout transform
- `MeanOps<Half>`：`mean(dim=2)` reduce

加入 `channels_last` 后，非主 stream 的 CNN kernel 总量下降，但 `mean(dim=2)` 的单次 reduce 变慢。端到端 benchmark 和 profiler 仍显示 predictor 总时长下降。

## Claude Code 审查

Claude Code 只读审查结论：没有必须修的问题。

它确认：

- pool 到 mean 语义等价。
- 删除 `AdaptiveAvgPool2d` 不影响 checkpoint，因为该层没有参数。
- `channels_last` 只改变内存布局，不改变数值。
- `squeeze(2).contiguous()` 对 Conv1d 输入是合理的。
- 本次 diff 不改 prefetch、lease、CUDA stream 时序。
- 不违反 `python-code-slim`。

## 下一步建议

当前最大问题仍是 GPU predictor 和主 decode 抢 GPU 资源。

下一步优先级：

1. 分析是否能减少 CNN 的 ReLU/bias add/layout kernel 数量。
2. 评估是否能把 `mean(dim=2)` 放在 Conv2d 之前或改写网络结构；这会改变模型语义，需要重新训练或质量验证，因此不能直接改。
3. 若继续不改模型结构，下一步更可能需要减少 predictor 调用次数或调度方式，而不是继续做小算子替换。
