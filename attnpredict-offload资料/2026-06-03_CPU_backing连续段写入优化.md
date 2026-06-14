# 优化1

## CPU backing 连续段写入优化

> 核心原理：当 prefill 分配到的 CPU slots 是连续递增区间时，用连续切片 `copy_` 替代 `index_copy_` 索引写入，减少 CPU backing 落入阶段的索引处理开销。

### 背景

在 `attnpredict-offload` 中，完整历史 KV 不全部常驻 GPU，而是保存在 CPU backing 中。
prefill 阶段每一层生成新的 K/V 后，都需要把这批 K/V 从 GPU 写入 CPU backing，保证后续如果 GPU active pool 中的 KV 被驱逐，仍然可以从 CPU 恢复。

原实现统一使用 `index_copy_` 按 `cpu_slots` 写入：

```python
cpu_kv_cache[0, layer_idx].index_copy_(0, cpu_slots, host_k)
cpu_kv_cache[1, layer_idx].index_copy_(0, cpu_slots, host_v)
```

这种方式适合 CPU slot 不连续的情况，但如果当前 chunk 的 `cpu_slots` 实际是连续递增的，仍然要额外处理索引数组，存在不必要的索引开销。

> 说明：即使 `cpu_slots = [5000, 5001, ...]`，`index_copy_` 也会按“读取索引 -> 定位目标 slot -> 写入”的散写流程执行，不能直接退化成简单的连续 slice copy。

### 优化方法

在 `_prepare_prefill()` 中检测当前 chunk 分配到的 CPU slots 是否连续：

```text
cpu_slots = [5000, 5001, 5002, ..., 5999]
```

如果连续，就记录为一个连续写入片段：

```python
(token_start, token_end, slot_start, slot_end)
```

随后在 `on_kv_stored()` 中改用连续切片写入：

```python
cpu_kv_cache[0, layer_idx, slot_start:slot_end].copy_(host_k[token_start:token_end])
cpu_kv_cache[1, layer_idx, slot_start:slot_end].copy_(host_v[token_start:token_end])
```

这样可以把原来的“按索引散写”变成“连续内存拷贝”。

> 适用范围：该优化只作用于 prefill 阶段的 CPU backing 写入。decode 阶段新 token 的 KV 先标记为 dirty，后续驱逐前写回 CPU 时仍走普通写回路径。

> 触发条件：当前 prefill batch 中每个 seq 的 `cpu_slots` 都必须是连续递增片段；如果某个 seq 的 slots 不连续，或者 CPU slot 分配已经碎片化，就回退到 `index_copy_`。

### 为什么有收益

连续切片写入的收益主要来自：

- 避免每层重复构造和消费 `cpu_slots` 索引 Tensor；
> 例子：如果当前 chunk 有 4096 个 token，原路径每一层都要把这 4096 个 `cpu_slots` 作为索引传给 `index_copy_`；连续段路径只需要 `slot_start:slot_end` 两个边界。

- 减少 `index_copy_` 的索引调度和散写处理；
- `_cpu_store_segments` 只在 prefill 准备阶段生成一次，后续所有 layer 都可以复用。
> 例子：一次 prefill 准备阶段得到 segment `(0, 4096, 5000, 9096)`，第 0 层到第 31 层写 CPU backing 时都可以复用这个 segment，只是写入的 `layer_idx` 不同。

该优化只改变 CPU backing 的写入方式，不改变保留 token 集合，也不改变 AttentionPredictor 的预测结果，因此不影响输出质量。

### 实验结果

实验配置为 `128k / bs=2 / output_len=16`，只测试 `attnpredict-offload`。
这个优化发生在 prefill 阶段，不直接减少每个 decode step 的计算时间。

Profiler 结果显示，CPU backing 写入路径在一次完整推理中共调用 `1056` 次：

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| CPU backing 写入平均耗时 | 47.67 ms/次 | 47.05 ms/次 |
| 单次调用平均减少 | - | 0.62 ms/次 |
| TTFT | 53.20 s | 52.56 s |
| PreTP | 4812.18 tok/s | 4870.42 tok/s |

> 观测结果：TTFT 从 `53.20 s` 降到 `52.56 s`，说明该优化对 128k / bs=2 的 prefill 阶段有小幅正收益。

整体看，该优化在 `128k / bs=2` prefill 中带来小幅端到端收益。
收益有限的原因是该路径的主要瓶颈仍然是 GPU 到 CPU 的 K/V 拷贝，连续段写入只优化了 CPU 侧落入 backing 的写入部分。
