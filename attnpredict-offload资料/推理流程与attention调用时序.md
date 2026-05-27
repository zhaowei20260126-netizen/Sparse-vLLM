# attnpredict-offload 推理流程与 attention 调用时序

本文档是给我自己理解 `attnpredict-offload` 用的，不是对外汇报稿。

重点回答三个问题：

1. Sparse-vLLM 推理时，attention 计算到底怎么走。
2. `attnpredict-offload` 在这些调用点里分别做什么。
3. 用一个 `batch=2, seq_len=10000, topk=4096` 的例子，把数据流和方法调用串起来。

## 0. 先解释本文常用术语

下面这些词后面会反复出现，先给短解释。

| 术语 | 中文解释 |
|---|---|
| KV cache | 键值缓存。每个历史 token 在 attention 里会产生 Key 和 Value，后续生成 token 时可以复用。 |
| prefill | 预填充阶段。一次性处理输入 prompt，并建立 prompt 的 KV cache。 |
| decode | 解码阶段。每次生成一个新 token，并追加这个 token 的 KV cache。 |
| offload | 卸载。这里指完整 KV 放到 CPU 内存，GPU 只放当前要参与 attention 的一小部分 KV。 |
| GPU active pool | GPU 活跃缓存池。GPU 上每层一组可复用 slot，只放当前驻留 GPU 的 KV。 |
| CPU full backing | CPU 完整后备缓存。CPU 上保存所有历史 token 的完整 KV，是 GPU 缺 KV 时的来源。 |
| slot | 槽位。KV cache 里的物理存储位置。逻辑 token 位置和物理 slot 不一定相同。 |
| sparse view | 稀疏视图。逻辑上决定这一步 attention 要看哪些 token。 |
| packed view | 打包视图。把 sparse view 选中的 token 转成 GPU slot 表，喂给 attention 内核。 |
| hot token | 热点 token。AttentionPredictor 预测下一步会重点关注的历史 token。 |
| hot lease | 热点租约。一次预测出的 hot token 集合复用若干 decode step。 |
| attn_score | 注意力分数缓冲区。注意力内核额外写出的分数，供 predictor 更新历史。 |
| with_score kernel | 带分数输出的注意力计算内核。除了算 attention 输出，还会把注意力分数写到 `attn_score`。 |
| logits | softmax 前的原始注意力打分。代码里 decode kernel 写出的 score 是 logits，不是归一化概率。 |
| CUDA stream | CUDA 流。GPU 任务队列。主计算流和预取流可以并行排队执行。 |
| CUDA event | CUDA 事件标记。表示某个 GPU 任务点完成，另一个流可以等待它。 |
| prefetch | 预取。提前把下一步可能需要的 KV 从 CPU 搬到 GPU。 |
| pinned memory | 固定页内存。CPU 上不会被操作系统换出的内存，更适合异步 CPU 到 GPU 拷贝。 |
| H2D | CPU 到 GPU 的数据传输。Host to Device。 |
| D2H | GPU 到 CPU 的数据传输。Device to Host。 |
| residency | 驻留状态。某个 token 的 KV 当前是否在 GPU active pool 里。 |
| residency diff | 驻留差分。比较旧 GPU 驻留集合和新目标集合，只释放/加载变化的 token。 |
| full-resident fast path | 全量驻留快速路径。如果 GPU active pool 已经能放完整上下文，就跳过复杂 offload 搬运，只做稀疏读取。 |

## 1. 总体结构：attention.py 保持通用，cache manager 管方法细节

`attnpredict-offload` 没有把逻辑塞进 `attention.py`。主流程仍然是：

```text
模型层 forward
  -> Attention.forward()
     -> cache_manager.get_layer_store_view()
     -> store_kvcache()
     -> cache_manager.on_kv_stored()
     -> sparse_controller.get_read_view()
     -> cache_manager.build_decode_view()
     -> attention kernel
     -> sparse_controller.on_attention_end()
```

关键文件：

- `src/sparsevllm/layers/attention.py`
  - 负责通用 attention 计算。
  - 不直接判断 `attnpredict-offload` 的细节。
- `src/sparsevllm/engine/sparse_controller.py`
  - 决定是否需要收集 attention 分数。
  - attention 结束后回调 cache manager。
- `src/sparsevllm/engine/cache_manager/attnpredict_offload.py`
  - `attnpredict-offload` 的核心状态和逻辑。
  - 管 CPU/GPU KV、预测、预取、稀疏视图、驻留集合。

`Attention.forward()` 的核心入口在：

```text
src/sparsevllm/layers/attention.py
```

其中最关键几段是：

```text
get_layer_store_view()       取得本层写 KV 的 GPU cache 和 slot_mapping
store_kvcache()              把当前 token/chunk 的 K/V 写入 GPU cache
on_kv_stored()               offload 方法额外处理 CPU backing 或 dirty 标记
get_read_view()              从 SparseController 拿逻辑读取视图
build_decode_view()          offload 方法把逻辑 token 打包成 GPU slot 表
attention kernel             真正做 attention 计算
on_attention_end()           attention 完成后触发 predictor 或 prefill 初始化
```

## 2. prefill 阶段：完整 attention + 初始化 predictor

prefill 是处理 prompt 的阶段。

对 `attnpredict-offload` 来说，prefill 的原则是：

```text
模型本身仍然做完整 attention。
predictor 额外收集最后 history_step 行的 block 级 attention 历史。
```

也就是说，prefill 不会因为 AttentionPredictor 而让模型少看 token。稀疏主要发生在 decode。

### 2.1 `prepare_step(..., is_prefill=True)`

Cache manager 会进入：

```text
AttnPredictOffloadCacheManager._prepare_prefill()
```

它做几件事：

1. 为本轮 prefill chunk 的 token 分配 CPU full backing slot。
2. 为每一层分配 GPU active slot。
3. 设置每层的 `LayerBatchStates`：
   - `slot_mapping`：当前 chunk token 要写到哪些 GPU slot。
   - `context_lens`：每个序列当前完整上下文长度。
   - `req_indices`：每个序列对应 cache manager 内部哪一行。

为什么 prefill 要每层都分配 GPU slot？

因为 prefill 阶段要求完整 attention。当前 chunk 写入后，attention kernel 需要能读到完整历史和当前 chunk。

### 2.2 Attention.forward 写入 KV

每一层进入 `Attention.forward()` 后，先调用：

```text
cache_manager.get_layer_store_view(layer_idx)
```

prefill 时它基本直接返回：

```text
本层 GPU k_cache
本层 GPU v_cache
本层 slot_mapping
```

然后：

```text
store_kvcache(k, v, store_k_cache, store_v_cache, slot_mapping)
```

把当前 chunk 的 K/V 写入 GPU active pool。

接着调用：

```text
cache_manager.on_kv_stored(layer_idx, k, slot_mapping, v=v)
```

prefill 时，这里会把 K/V 同步写入 CPU full backing。因为 CPU full backing 要保存完整 KV，后续 GPU 驱逐后才能恢复。

注意：decode 阶段这里不立刻写 CPU，decode 会走“懒写回”，后面单独讲。

### 2.3 prefill 最后一块：准备 block 级 tail-score

AttentionPredictor 需要历史 attention 作为输入。代码在 prefill 分支里调用：

```text
cache_manager.prepare_prefill_predictor_inputs(...)
```

`attnpredict-offload` 的实现会在最后一个 prefill chunk 创建一个 4D 缓冲区：

```text
tail_score shape =
[batch_size, num_heads, history_step, ceil(max_context_len / block_size)]
```

这里 `tail_score` 可以理解成：

```text
prompt 最后 history_step 个 query，对历史 KV 的 block 级注意力概率。
```

为什么是 block 级？

原始 token 级可能是：

```text
[batch, heads, history, seq_len]
```

如果 `seq_len=10000`，最后一维是 10000。

block size 为 16 时，变成：

```text
ceil(10000 / 16) = 625
```

最后一维从 10000 缩成 625，少 16 倍。

这一步只是 predictor 的输入压缩，不影响模型 prefill 完整 attention。

### 2.4 context_attention_fwd 做完整 prefill attention

接下来调用：

```text
context_attention_fwd(...)
```

它仍然做完整 prefill attention。

如果传入了 `tail_score`，Triton kernel 会额外把最后 `history_step` 个 query 的注意力概率按 block 做最大池化，写入 `tail_score`。

中文理解：

```text
模型正常算完整 attention。
顺便在旁路记录一份 predictor 需要的 block 级历史分数。
```

### 2.5 attention 结束后：on_prefill_layer_end

attention kernel 完成后，`Attention.forward()` 调用：

```text
sparse_controller.on_attention_end(layer_idx, context)
```

对 `attnpredict-offload`，prefill 分支会调用：

```text
cache_manager.on_prefill_layer_end(layer_idx)
```

这个方法会：

1. 拿到刚才 kernel 写好的 `tail_score`。
2. 记录 CUDA event，保证后台任务等 attention 写完。
3. 提交后台任务 `_prefill_predict_prefetch_worker()`。

后台任务会：

```text
tail_score
  -> 更新 attn_history
  -> CNN predictor 预测 hot tokens
  -> 生成首个 decode step 的 hot lease
  -> 预取首个 decode 需要的 KV
```

最后，prefill step 结束前：

```text
SparseController.post_forward()
  -> cache_manager.on_prefill_step_end()
```

这里会等待所有层首个 decode 所需的预测和预取完成。原因是第一个 decode step 必须已有可用 lease，否则 decode throughput 会被第一次等待污染。

## 3. decode 阶段：稀疏 attention + 预测下一步

decode 是每次生成一个新 token 的阶段。`attnpredict-offload` 的核心逻辑都在这里。

### 3.1 `prepare_step(..., is_prefill=False)`

Cache manager 进入：

```text
AttnPredictOffloadCacheManager._prepare_decode()
```

它做这些事：

1. 为每个序列当前新 token 分配 CPU full backing slot。
2. 暂时不为所有层一次性分配 GPU slot。
3. 记录当前 batch 的：
   - `_decode_rows`：每个序列对应哪一行。
   - `_decode_current_positions`：当前新 token 的逻辑位置。
4. 检查每层是否是 full-resident fast path。
5. 为 full-resident 层提前准备 current token 的 GPU slot。
6. 计算每个 row 的 recent window 位置，供后续快速打包。

为什么 decode 当前 token 的 GPU slot 不在 `_prepare_decode()` 里一次性分配？

因为每一层的 K/V 不同。当前 token 的第 0 层 KV、第 1 层 KV、... 都要分别写入各层 cache。代码选择在每层真正进入 `get_layer_store_view(layer)` 时再处理该层，这样能和该层的预取结果自然衔接。

### 3.2 到达某一层：get_layer_store_view 是“按层消费点”

每层 attention 开始时，先调用：

```text
cache_manager.get_layer_store_view(layer_idx)
```

decode 时这是非常关键的函数。

它做这些事：

1. 如果上一轮后台 refresh 已完成，就消费它，切换到新 hot lease。
2. 如果后台 refresh 未完成，但旧 lease 还没太旧，就继续用旧 lease，不阻塞。
3. 如果超过最大过期步数，才等待后台任务。
4. 组合本层本 batch 的目标 token positions：

```text
sink tokens
+ leased hot tokens
+ recent tokens
+ current token
```

5. 更新 GPU active pool 的驻留状态：
   - 目标里需要但 GPU 没有的 token，从 CPU 加载到 GPU。
   - GPU 里有但目标不需要的 token，释放 GPU slot。
6. 给当前 decode token 分配本层 GPU slot。
7. 返回本层写入 K/V 用的 `slot_mapping`。

这里的目标集合是 sparse view，也就是：

```text
逻辑上这一层这一批序列要看哪些 token。
```

### 3.3 residency diff：只处理变化的 KV

驻留差分在：

```text
_ensure_positions_resident(layer_idx, row_positions, release_old=True/False)
```

核心逻辑是：

```text
old    = 当前 GPU 里已经驻留的 token positions
target = 下一步 attention 需要的 token positions

to_free = old - target
missing = target - old
```

中文解释：

```text
to_free：GPU 里有，但下一步不看了 -> 释放 GPU slot
missing：下一步要看，但 GPU 没有 -> 从 CPU full backing 加载
```

如果后台 refresh 还没正式被主线程消费，代码会用：

```text
release_old=False
```

这时只加载新 token，不释放旧 token。这样避免主线程还在用旧 view 时，后台刚加载的新 token 或旧 token 被错误释放。

### 3.4 写入当前 token 的 K/V

`get_layer_store_view()` 返回后，`Attention.forward()` 调用：

```text
store_kvcache(k, v, store_k_cache, store_v_cache, slot_mapping)
```

这把当前 token 的本层 K/V 写到刚分配的 GPU slot。

然后：

```text
cache_manager.on_kv_stored(layer_idx, k, slot_mapping, v=v)
```

decode 时，`attnpredict-offload` 不立刻把 K/V 拷到 CPU。

它只做：

```text
把当前 token 标记为 dirty
```

dirty 的中文意思是“脏数据”：GPU 上有最新 KV，但 CPU full backing 还没有。

为什么这样做？

因为当前 token 通常会在 recent window 中保留一段时间。如果每层每步都马上 GPU 到 CPU 拷贝，会造成同步开销。现在改为：

```text
只有该 token 将要从 GPU 驱逐时，才补写 CPU。
```

这叫懒写回。

### 3.5 get_read_view：拿到通用读取视图

接着 `Attention.forward()` 调用：

```text
sparse_controller.get_read_view(layer_idx)
```

对 `attnpredict-offload`，它会返回一个较通用的读取视图。真正的稀疏打包还没有完成，后面由 cache manager 的：

```text
build_decode_view()
```

来完成。

### 3.6 build_decode_view：从 sparse view 到 packed view

`build_decode_view()` 做的事情是：

```text
逻辑 token positions
  -> 查 GPU slot
  -> 打包成 packed_slots
  -> 返回给 decode attention kernel
```

举例：

```text
sparse view 选择的 token 位置:
[0, 1, 63, 2000, 5000, 9999, 10000]

这些 token 在 GPU active pool 的 slot:
[88, 91, 155, 3001, 42, 777, 900]

packed view:
packed_slots = [88, 91, 155, 3001, 42, 777, 900]
```

所以：

```text
sparse view 回答：看哪些 token。
packed view 回答：这些 token 在 GPU 哪些 slot，怎么交给 kernel。
```

`build_decode_view()` 还会复用临时 buffer，避免每层每步重复分配：

```text
packed_slots
packed_positions
view_lens
local_req_indices
```

如果当前层需要收集 attention 分数，它还会保存 `_last_decode_view`，给 predictor 后续使用。

### 3.7 decode attention kernel

decode attention 计算在 `Attention.forward()` 中走两阶段：

1. stage1：分块计算 attention 的局部结果。
2. stage2：合并所有块，得到最终 attention 输出。

如果当前 step 需要刷新 predictor，会用带分数输出的 attention 内核：

```text
with_score kernel
```

中文理解：

```text
普通 attention 内核：
  只输出 attention 结果。

带分数输出的 attention 内核：
  输出 attention 结果
  额外把当前 query 对 visible tokens 的 logits 写入 attn_score。
```

注意：这里写入的是 softmax 前的 logits。后面 predictor 会执行：

```text
softmax(logits * attn_scale)
```

把它变成注意力概率。

如果当前 step 不需要刷新 predictor，则用普通 attention kernel，不分配/写入 `attn_score`。

### 3.8 attention 结束后：predict_next_mask

attention kernel 完成后，`Attention.forward()` 调用：

```text
sparse_controller.on_attention_end(layer_idx, context)
```

decode 时，controller 检查：

```text
state.attn_score is not None
```

如果有 `attn_score`，说明本层本 step 需要刷新 predictor，于是调用：

```text
cache_manager.predict_next_mask(layer_idx, state.attn_score)
```

如果开启异步预取，这里不会马上跑 CNN。它只做三件事：

1. 记录 CUDA event，表示当前 attention kernel 完成点。
2. 保存当前层的 `attn_logits` 和 decode view 信息。
3. 把任务放入 `_pending_decode_predict_inputs`。

为什么要 CUDA event？

因为后台线程不能在 attention kernel 写完 `attn_score` 之前读它。CUDA event 相当于：

```text
在主计算流里插一个“attention 到这里已经写完”的标记。
后台预取流必须等这个标记完成后，才能读 attn_score。
```

### 3.9 decode step 结束：批量提交后台预测和预取

一个 decode step 的模型 forward 结束后，controller 的 post_forward 会调用：

```text
cache_manager.on_decode_step_end()
```

它把这一轮 pending 的多个层一起提交到后台线程：

```text
_predict_decode_batch_and_prefetch_worker(items)
```

后台 worker 做：

1. 让预取 CUDA stream 等每个 layer 的 event。
2. 对 `attn_logits` 做 softmax，得到 attention 概率。
3. 按 block 聚合 attention 历史。
4. 批量跑 CNN predictor。
5. 得到新的 hot positions。
6. 只加载新 lease 需要但 GPU 当前没有的 KV。
7. 记录 done event，等待下一次同层消费。

这里有一个重要点：

```text
后台 worker 只加载新 hot set，不释放旧 view。
```

释放旧 view 是主线程在后续 `get_layer_store_view()` 消费新 lease 时做的。这样可以避免主线程和后台线程同时改 GPU active pool 引发竞态。

## 4. Cross-step reuse：不是每步都刷新 predictor

`should_collect_decode_attn_score(layer_idx)` 决定当前 decode step 是否需要收集 attention score。

逻辑是：

```text
如果没有 lease -> 需要收集
如果有未完成 prefetch future -> 不收集，继续用旧 lease
如果 lease 使用步数达到 reuse_steps -> 需要收集
否则 -> 不收集
```

因此很多 decode step 不会走带分数输出的注意力内核，也不会跑 CNN。

当前代码里 `config.py` 的默认值是：

```text
attnpredict_reuse_steps = 100000
attnpredict_max_stale_steps = 100000
```

这几乎等价于：

```text
首轮预测后，长时间复用 hot lease。
```

这会显著提升速度，但也意味着 predictor 不会频繁根据最新 attention 漂移更新。做实验解释时要说明实际配置。

## 5. Decode Attn-Score 缩宽放在流程里的位置

如果当前 step 需要刷新 predictor，SparseController 会为每层分配 `attn_score`。

普通完整方式会按完整上下文长度分配：

```text
[batch_size, num_heads, full_context_len]
```

`attnpredict-offload` 会改成按稀疏可见长度上界分配：

```text
[batch_size, num_heads, keep_bound]
```

其中：

```text
middle_budget = topk - sink - recent
block_budget  = middle_budget // block_size
keep_bound    = sink + recent + block_budget * block_size + 1
```

`+1` 是给当前 token 留的保守空间。

这样做的原因：

```text
decode attention 实际只看 packed view 里的 token。
with_score kernel 只需要写这些 visible tokens 的分数。
没有必要给完整 10000 或 128000 长度分配 score buffer。
```

## 6. full-resident fast path：为什么有时 offload 也不搬运

如果某个 batch 的完整上下文能够全部放进 GPU active pool，代码会标记该层：

```text
_decode_full_resident_layers[layer_id] = True
```

这时：

```text
完整 KV 已经在 GPU
不需要从 CPU 加载
不需要驱逐
只需要构造 sparse packed view
```

这就是 full-resident fast path。

它的本质是：

```text
物理 KV 全在 GPU，但计算仍然只读 sparse view。
```

这能解释一些 benchmark 里 `attnpredict-offload` 比 vanilla decode 更快的情况：  
如果 active pool 足够大，它实际没有承受 CPU 到 GPU 搬运压力，而是主要收益来自 sparse attention 计算量变小。

## 7. 例子：batch=2, seq_len=10000, topk=4096

下面用具体数字走一遍。

假设配置：

```text
batch_size = 2
prefill 后每个序列长度 = 10000
topk = 4096
num_sink_tokens = 64
num_recent_tokens = 512
pooling_block_size = 16
history_steps = 64
num_heads = 32
```

注意：这里 `topk=4096` 是总预算，包含 sink 和 recent，不是额外再加。

### 7.1 prefill 阶段的数据形状

两个序列：

```text
seq0: token positions 0..9999
seq1: token positions 0..9999
```

`_prepare_prefill()` 分配：

```text
CPU full backing:
  seq0 的 10000 个 token -> CPU slots
  seq1 的 10000 个 token -> CPU slots

GPU active pool:
  每层都为 seq0 的 10000 个 token 分配 GPU slots
  每层都为 seq1 的 10000 个 token 分配 GPU slots
```

每层 prefill attention 仍然完整看：

```text
seq0 的 0..9999
seq1 的 0..9999
```

最后一个 prefill chunk 创建 predictor 的 `tail_score`：

```text
pooled_len = ceil(10000 / 16) = 625

tail_score shape =
[2, 32, 64, 625]
```

如果是 token 级则是：

```text
[2, 32, 64, 10000]
```

block 级节省 16 倍最后一维。

prefill attention kernel 做完整 attention，同时把最后 64 个 query 的 block 级 attention 概率写进 `tail_score`。

然后每层：

```text
on_prefill_layer_end()
  -> 后台 _prefill_predict_prefetch_worker()
     -> _predict_prefill_positions_sync()
        -> 更新 attn_history
        -> CNN predictor 预测 hot tokens
        -> 生成首个 hot lease
        -> 预取首个 decode 需要的 KV
```

prefill step 结束：

```text
on_prefill_step_end()
  -> 等所有层首个 decode lease 准备好
```

### 7.2 topk=4096 如何拆成 sink/hot/recent

预算：

```text
topk = 4096
sink = 64
recent = 512
middle_budget = 4096 - 64 - 512 = 3520
```

block size 为 16：

```text
block_budget = 3520 / 16 = 220 个 block
```

CNN predictor 在中间区域选择 220 个 block：

```text
220 blocks * 16 tokens/block = 3520 tokens
```

所以每个序列首个 decode 大约看：

```text
64 个 sink token
+ 3520 个 predicted hot token
+ 512 个 recent token
= 4096 个 token
```

### 7.3 第一个 decode step：逻辑长度变成 10001

prefill 后每个序列长度是 10000。

第一个 decode step 会生成当前位置：

```text
seq0 current_pos = 10000
seq1 current_pos = 10000
```

`_prepare_decode()` 先为每个序列当前 token 分配 CPU backing slot：

```text
seq0 pos 10000 -> CPU slot
seq1 pos 10000 -> CPU slot
```

此时完整逻辑长度变成：

```text
context_len = 10001
```

但当前 token 的每层 K/V 还没有写入 GPU，因为模型还没算到每一层。

### 7.4 到达第 L 层 attention：先准备写入位置

进入第 L 层 `Attention.forward()`：

```text
get_layer_store_view(layer_idx=L)
```

它为每个序列组合目标 sparse view：

```text
seq0:
  sink:   0..63
  hot:    predictor 选出的 3520 个中间 token
  recent: 9489..10000   因为 10001 - 512 = 9489

seq1:
  sink:   0..63
  hot:    predictor 选出的 3520 个中间 token
  recent: 9489..10000
```

注意：当前 token `10000` 已经包含在 recent window 里，所以不会重复计数。代码最后会 `np.unique` 去重。

接着做 residency diff。

假设第 L 层 GPU 里当前已驻留：

```text
old(seq0) = prefill 后还保留在 GPU 的某些 token
target(seq0) = sink + hot + recent/current
```

如果不是 full-resident fast path，则：

```text
to_free = old - target
missing = target - old
```

对于 `missing`：

```text
从 CPU full backing 找到对应 cpu_slot
分配 GPU active slot
CPU gather 出 K/V
通过 H2D 拷贝到 GPU active pool
更新 row/pos -> gpu_slot 映射
```

对于 `to_free`：

```text
如果 token 是 dirty，先写回 CPU backing
然后释放 GPU active slot
把 row/pos -> gpu_slot 写成 -1
```

最后，为当前 token `10000` 分配本层 GPU slot，返回 `slot_mapping`。

### 7.5 写入当前 token 的 K/V

`Attention.forward()` 接着：

```text
store_kvcache(k, v, store_k_cache, store_v_cache, slot_mapping)
```

把 seq0、seq1 当前 token 在第 L 层的 K/V 写入 GPU active pool。

然后：

```text
on_kv_stored(layer_idx=L, ...)
```

decode 阶段这里不马上 D2H 写回 CPU，只标记：

```text
seq0 pos 10000 dirty
seq1 pos 10000 dirty
```

### 7.6 构造 packed view

然后：

```text
sparse_controller.get_read_view(L)
cache_manager.build_decode_view(L, ...)
```

对每个序列，`build_decode_view()` 把目标 token positions 转成 GPU slot。

例如 seq0：

```text
sparse view positions:
[0, 1, ..., 63, 120, 500, ..., 9489, ..., 10000]

查表得到 GPU slots:
[88, 91, ..., 155, 3001, 42, ..., 777, ..., 900]

packed_slots[0, :] =
[88, 91, ..., 155, 3001, 42, ..., 777, ..., 900]
```

seq1 类似：

```text
packed_slots[1, :] = [...]
```

最终传给 attention kernel 的关键张量：

```text
packed_slots shape 约 [2, 4096]
view_lens shape = [2]
view_lens = [4096, 4096]  # 近似，边界去重时可能略有差异
```

### 7.7 attention 计算实际读什么

decode attention kernel 看到的是：

```text
batch=2
每个序列 visible_len 约 4096
```

而不是完整 10001。

它会对每个序列、每个 head：

```text
当前 q
  x packed_slots 中对应的 K
  -> 注意力分数
  -> softmax
  -> 加权 V
  -> 输出 o
```

如果当前 step 需要刷新 predictor，则用带分数输出的 attention 内核：

```text
输出 o
同时把 visible tokens 的 logits 写入 attn_score
```

此时 `attn_score` 的长度不是 10001，而是大约 4097：

```text
middle_budget = 3520
block_budget = 220
keep_bound = 64 + 512 + 220 * 16 + 1 = 4097

attn_score shape = [2, 32, 4097]
```

如果没有缩宽，则可能是：

```text
[2, 32, 10001]
```

所以这一步减少了 score buffer 的分配和写入。

### 7.8 attention 结束后预测下一步

第 L 层 attention 结束后：

```text
sparse_controller.on_attention_end(L, context)
```

如果这一层这一 step 收集了 `attn_score`，则调用：

```text
cache_manager.predict_next_mask(L, attn_score)
```

异步开启时它不立刻跑 CNN，只做：

```text
event = torch.cuda.Event()
event.record(current_stream)
保存 attn_logits + 当前 decode view + event
```

当前层继续后面的计算，模型继续跑后续层。

一个 decode step 全部结束后：

```text
cache_manager.on_decode_step_end()
```

把所有 pending 层提交给后台 worker：

```text
_predict_decode_batch_and_prefetch_worker(items)
```

后台 worker：

```text
等待每层 CUDA event
读取 attn_logits
softmax 成 attention 概率
按 block 聚合到 attention history
批量跑 CNN
得到下一步 hot tokens
把下一步缺失的 KV 从 CPU 预取到 GPU
记录完成 event
```

下一步 decode 再到第 L 层时：

```text
get_layer_store_view(L)
  -> 如果后台 future 完成，则消费新 lease
  -> 如果没完成且未超过 stale 上限，则继续旧 lease
  -> 如果超过 stale 上限，则等待
```

## 8. 一条完整时序总结

下面是最重要的调用时序。

### prefill 最后一块

```text
CacheManager._prepare_prefill()
  -> 分配 CPU slots 和 GPU slots

每层 Attention.forward()
  -> get_layer_store_view()
  -> store_kvcache()
  -> on_kv_stored()                  # prefill: K/V 写入 CPU backing
  -> get_read_view()
  -> prepare_prefill_predictor_inputs()
       -> 创建 block 级 tail_score
  -> context_attention_fwd()
       -> 完整 attention
       -> 旁路写 tail_score
  -> sparse_controller.on_attention_end()
       -> on_prefill_layer_end()
          -> 提交 prefill predictor/prefetch worker

SparseController.post_forward()
  -> on_prefill_step_end()
     -> 等首个 decode lease 准备好
```

### decode 每一步每层

```text
CacheManager._prepare_decode()
  -> 为当前 token 分配 CPU backing slot
  -> 记录 rows/current_positions

每层 Attention.forward()
  -> get_layer_store_view()
       -> 消费已完成 prefetch future
       -> 组合 sink + hot lease + recent + current
       -> residency diff: 释放不需要的，加载缺失的
       -> 分配当前 token 的本层 GPU slot
  -> store_kvcache()
       -> 当前 token K/V 写入 GPU active pool
  -> on_kv_stored()
       -> decode: 标记 dirty，不立即写 CPU
  -> get_read_view()
  -> build_decode_view()
       -> sparse view positions 转 packed GPU slots
  -> decode attention stage1
       -> 如果需要 refresh，用 with_score kernel 写 attn_score
  -> decode attention stage2
       -> 合并输出
  -> sparse_controller.on_attention_end()
       -> predict_next_mask()
          -> 记录 CUDA event，暂存 logits/view

SparseController.post_forward()
  -> on_decode_step_end()
     -> 批量提交后台 predictor + prefetch
```

## 9. 当前实现中最容易误解的点

### 9.1 sparse view 和 packed view 不是一回事

```text
sparse view：逻辑 token 位置，表示看哪些 token。
packed view：GPU slot 表，表示这些 token 的 KV 在 GPU 哪些槽位。
```

### 9.2 attn_score 缩宽不是缩短上下文

它只是缩小 predictor 用的分数缓冲区。

真正 attention 读多少 token，是由 `build_decode_view()` 返回的 packed view 决定的。

### 9.3 offload 不一定每次都真的发生 CPU/GPU 搬运

如果 full-resident fast path 生效，完整 KV 已经在 GPU，代码会跳过驱逐和加载。

这种情况下速度提升主要来自：

```text
attention 只读 sparse packed view
而不是读完整上下文
```

### 9.4 hot lease 复用会影响“预测准确性”和“速度”的权衡

`reuse_steps` 越大：

```text
刷新 predictor 越少
速度越快
但 attention 模式漂移时适应越慢
```

当前默认 `100000` 很激进。做算法研究时，最好显式记录并比较不同 `reuse_steps`。

### 9.5 dirty token 不是错误，而是懒写回机制

dirty 表示：

```text
GPU 上有新 KV
CPU backing 还没同步
```

只有 token 要被驱逐时才写回 CPU。

## 10. 读代码时建议按这个顺序看

1. `src/sparsevllm/layers/attention.py`
   - 先理解通用 attention 主流程。
2. `src/sparsevllm/engine/sparse_controller.py`
   - 看 `_needs_attn_score()` 和 `on_attention_end()`。
3. `src/sparsevllm/engine/cache_manager/attnpredict_offload.py`
   - 看 `_prepare_prefill()` 和 `_prepare_decode()`。
   - 看 `get_layer_store_view()`。
   - 看 `build_decode_view()`。
   - 看 `predict_next_mask()` 和 `on_decode_step_end()`。
   - 看 `_ensure_positions_resident()`。
4. `src/sparsevllm/triton_kernel/context_flashattention_nopad.py`
   - 看 prefill tail_score 如何按 block 写入。

一句话总括：

```text
attnpredict-offload 的推理内部流程是：
CPU 保存完整 KV，GPU 保存当前可见 KV；
prefill 完整 attention 并初始化 predictor；
decode 每层先准备 sparse view 对应的 GPU KV，再打包成 packed view 做 attention；
attention 结束后异步预测下一步 hot tokens，并预取缺失 KV；
下一步到同一层时消费预取结果。
```
