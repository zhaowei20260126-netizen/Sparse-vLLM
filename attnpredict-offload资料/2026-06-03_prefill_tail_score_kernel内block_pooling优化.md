# 优化2

## Prefill tail score 的 kernel 内 block pooling 写出

> 核心原理：prefill attention kernel 在正常计算 attention 输出的同时，只把最后 `history_step` 个 query 的 attention probability 按 block 做 max pooling 后写入 `tail_score`，避免先写 token 级 score 再额外做 pooling。

### 背景

`attnpredict-offload` 需要在 prefill 最后一个 chunk 初始化 predictor history。原始语义需要拿到最后 `history_step` 个 query 对历史 KV 的注意力分数，然后按 `pooling_block_size` 压成 block 级输入。

如果先保存 token 级 attention score，再在 PyTorch 里做 pooling，路径会变成：

```text
prefill attention -> 写 token 级 score -> 额外 pooling -> 更新 predictor history
```

这会带来两个问题：

- token 级 score buffer 很大；
- prefill attention 后还要额外启动 pooling 相关张量操作。

### 优化方法

在 `prepare_prefill_predictor_inputs()` 中创建 4D `tail_score`：

```python
(batch_size, num_heads, history_step, pooled_len)
```

其中：

```text
pooled_len = ceil(max_context_len / pooling_block_size)
```

随后 `context_attention_fwd()` 根据 4D `attn_score` 走 `_fwd_kernel_with_tail_score` 分支。该 kernel 在 attention 内部完成：

```text
1. 计算正常 prefill attention 输出；
2. 只针对最后 history_step 个 query；
3. 对 KV 维度按 block 做 max pooling；
4. 直接写入 block 级 tail_score。
```

> 说明：这里融合的是“score 写出 + block pooling”，不是把 CNN predictor、top-k、lease 更新或 prefetch 全部融合进 prefill attention kernel。

### 为什么有收益

- score buffer 从 token 级长度变成 block 级长度；
> 例子：`128k` 上下文、`pooling_block_size = 16` 时，KV 维度从 `128000` 个 token 压到约 `8000` 个 block。

- 只收集最后 `history_step` 个 query，不保存整个 prefill chunk 的全部 query 分数；
- pooling 在 attention kernel 内完成，避免 prefill attention 后再额外做一次 token score 到 block score 的转换；
- 该优化只改变 predictor history 的生成方式，不改变 prefill attention 的可见 KV，也不改变模型输出 attention 结果。

### 适用范围

该优化只作用于 prefill 最后一个 chunk 的 predictor 初始化。

触发条件：

```text
attn_score 是 4D tail_score
attn_score_block_size = pooling_block_size
```

非 source layer 会因为跨层复用跳过 predictor 初始化，因此不会创建这份 `tail_score`。

### 实验结果

该优化已经在当前 `attnpredict-offload` prefill predictor 初始化路径中采用，但目前没有单独关闭该分支做独立 ablation。

| 指标 | 当前状态 |
|---|---:|
| 是否已接入当前实现 | 是 |
| 作用阶段 | prefill 最后一个 chunk |
| 是否直接减少单步 decode 时间 | 否 |
| 独立优化前后对比 | 暂未单独测量 |

> 观测结果：当前无法单独给出该优化点的 TTFT 或吞吐收益；可以确定的是，它减少了 prefill predictor 初始化所需的 score buffer 尺寸，并把 token score 到 block score 的转换下沉到了 attention kernel 内。
