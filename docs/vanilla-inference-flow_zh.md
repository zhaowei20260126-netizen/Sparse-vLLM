# Sparse-vLLM 中 Vanilla（全注意力）推理流程详解

> 本文从 nanovllm 开发者的视角，系统讲解 Sparse-vLLM 中 vanilla 默认推理的完整流程。
> 适合有一定 LLM 基础但思路需要理顺的开发者。

---

## 0. 先看全局：你的代码会经历什么？

```python
from sparsevllm import LLM, SamplingParams

# 第一步：创建引擎（只做一次初始化）
llm = LLM(
    model_path="/path/to/model",
    vllm_sparse_method="",  # 空字符串 = Vanilla（全注意力）
    gpu_memory_utilization=0.8,
    max_num_batched_tokens=8192,
)

# 第二步：发起推理请求
sampling_params = SamplingParams(max_tokens=100, temperature=0.7)
outputs = llm.generate(
    prompts=["Write a story about..."],
    sampling_params=sampling_params,
)

# 👆 这两行代码背后会发生什么？这就是本文要讲的！
```

---

## 1. 初始化阶段（LLM.__init__）：启动完整推理系统

### 1.1 总体步骤（4 个关键模块）

```
LLM.__init__(model_path, **config_kwargs)
│
├─ [第1步] Config 解析
│          配置参数的规范化和验证
│
├─ [第2步] ModelRunner 初始化（Rank 0，主进程）
│          ├─ 加载模型权重到 GPU
│          ├─ 初始化 KV Cache 物理存储
│          └─ 初始化稀疏控制器（即使 vanilla 也有）
│
├─ [第3步] 多进程 TP 环境setup（如果 tensor_parallel_size > 1）
│          启动多个 GPU worker 进程
│
├─ [第4步] Scheduler 初始化
│          ├─ 连接到 Rank 0 的 CacheManager（作为显存参考）
│          └─ 初始化 waiting/decoding 两个队列
│
└─ [完成] _warmup()
           执行一个虚拟 forward，让 Triton kernel 编译完成
```

### 1.2 关键类和文件映射

| 组件 | 文件 | 职责 |
|-----|------|------|
| **Config** | `src/sparsevllm/config.py` | 全局参数中心：方法选择、预算管理、层配置 |
| **LLMEngine** | `src/sparsevllm/engine/llm_engine.py` | 推理引擎主类，协调 Scheduler 和 ModelRunner |
| **ModelRunner** | `src/sparsevllm/engine/model_runner.py` | GPU 上的模型执行单元，管理前向计算和 TP 通信 |
| **Scheduler** | `src/sparsevllm/engine/scheduler.py` | CPU 端的批调度器，决定每一步运行哪些序列 |
| **CacheManager** | `src/sparsevllm/engine/cache_manager/` | KV Cache 生命周期管理（对于 vanilla 是 `StandardCacheManager`） |

### 1.3 Vanilla 的特殊性

当你设置 `vllm_sparse_method=""` (空字符串) 时：

```python
# 在 cache_manager/base.py 的工厂函数
if sparse_method == "":  # Vanilla case
    from .standard import StandardCacheManager
    return StandardCacheManager(config, rank, world_size)
```

**这意味着：**
- 不使用 DeltaKV、OmniKV、SnapKV 等稀疏逻辑
- KV Cache 简单线性分配：全部 token 按生成顺序存储在物理槽位中
- 注意力计算时访问全部历史 token（因此叫 "vanilla"）
- 无复杂的驱逐/压缩/掩码逻辑

---

## 2. 推理请求流程（add_request + generate）

### 2.1 发起请求的路径

```python
# 用户代码
outputs = llm.generate(prompts=["..."], sampling_params=sampling_params)

# 内部展开为（位置：src/sparsevllm/llm.py 继承自 LLMEngine）
def generate(self, prompts, sampling_params):
    # 第1步：将每个 prompt 转为 token_ids，加入 Scheduler 的等待队列
    for prompt in prompts:
        self.add_request(prompt, sampling_params)
    
    # 第2步：反复执行 step() 直到所有请求完成
    while not self.is_finished():
        self.step()  # ← 这是单次推理的核心
    
    # 第3步：返回生成结果
    return outputs
```

### 2.2 add_request：把请求加入队列

```
add_request(prompt_text, sampling_params)
│
├─ 第1步：Tokenize
│         prompt_text → token_ids (list of int)
│
├─ 第2步：创建 Sequence 对象
│         Sequence(token_ids=[1, 2, 3, ...], sampling_params=...)
│         这个对象存储序列的完整生命周期信息：
│         ├─ seq_id: 唯一序列号
│         ├─ num_tokens: 已有的 token 数（初始 = prompt 长度）
│         ├─ num_prefilled_tokens: 已经在 GPU 上处理的 token 数（初始 = 0）
│         ├─ status: WAITING（刚加入时）
│         └─ ...其他采样参数
│
└─ 第3步：加入 Scheduler 的 waiting 队列
          scheduler.waiting.append(seq)
          （等待调度器在下一个 step() 中择机处理）
```

---

## 3. 单个推理步骤（step）：核心循环的一次迭代

这是整个系统的心脏。每次调用 `step()` 都会：

```
step()
│
├──── [阶段1] Scheduler.schedule() ──────────────────────┐
│                                                         │
│     决定本次 GPU 计算用哪些序列、是 prefill 还是 decode │
│     返回：(seqs, is_prefill, preempted_seqs)            │
│                                                         │
│     ┌─ 如果 waiting 队列不空 → 优先 Prefill            │
│     │  └─ 把最多 max_num_batched_tokens 个 token 调进来 │
│     │     （对长序列，可能分块：chunk_prefill_size）    │
│     │                                                  │
│     └─ 否则 → Decode                                   │
│        └─ 从 decoding 队列选择序列进行单 token 生成     │
│                                                         │
└───────────────────────────────────┬────────────────────┘
                                    │
├─── [阶段2] 处理被抢占序列 ──────────┤
│     for seq in preempted_seqs:     │
│         call("free_slots", seq.id) │ (通知 GPU 释放显存)
│                                    │
└────────────────────────────────┬──┘
                                 │
├─── [阶段3] ModelRunner.run() ──┤ (GPU 前向计算)
│                                 │
│     call("run", seqs, is_prefill)
│     ↓
│     内部步骤：
│     1. prepare_step()     :  获取 input_ids, positions
│     2. sparse_controller.prepare_forward() : 初始化稀疏状态
│     3. self.model.forward() : 执行模型前向
│     4. sampler() : 采样 token (only on Rank 0)
│     5. sparse_controller.post_forward() : 后处理（vanilla 无操作）
│     ↓
│     返回：token_ids (采样得到的新 token)
│                                 │
└────────────────────────────────┬──┘
                                 │
├─── [阶段4] 状态更新 ───────────┤
│     scheduler.postprocess(seqs, token_ids, is_prefill)
│     ├─ 对每个序列 seq，追加新 token：seq.append_token(token_id)
│     ├─ 如果是 prefill，更新 num_prefilled_tokens += chunk_size
│     │  并判断是否 prefill 完成（转入 decoding 队列）
│     └─ 如果是 decode，增加 num_tokens，判断 EOS 或达到 max_length
│                                 │
└────────────────────────────────┬──┘
                                 │
└─── [阶段5] 完成序列回收 ────────┘
      for seq in seqs:
          if seq.is_finished:
              call("free_slots", seq.seq_id)  (释放 GPU 显存)
              add_to_outputs(seq)              (返回给用户)
```

---

## 4. 详细拆解：关键子流程

### 4.1 Scheduler.schedule() - 调度逻辑

**Vanilla 方法下的调度策略：**

```
schedule()
│
├─ 初始化计数器
│  ├─ num_batched_tokens = 0
│  ├─ num_batched_seqs = 0
│  ├─ physical_free_slots = cache_manager.num_free_slots (GPU 可用槽位)
│  └─ logical_free_slots = 保守估计的逻辑可用空间
│
├─ [如果 waiting 队列非空] → 优先 Prefill
│  │
│  └─ while waiting && physical_free_slots > 0:
│     ├─ seq = waiting.pop()  (取出一个等待的序列)
│     │
│     ├─ 计算该序列本次能处理的 token 数：
│     │  can_prefill = min(
│     │      remaining_prompt_tokens,  # 还剩多少 prompt token
│     │      chunk_prefill_size,        # 一次最多处理多少（如 8192）
│     │      max_num_batched_tokens - num_batched_tokens,  # 整个 batch 限制
│     │      physical_free_slots        # GPU 显存限制
│     │  )
│     │
│     ├─ 如果 can_prefill <= 0 → 放回队列，暂停 prefill
│     │
│     └─ 否则 → 加入本次 batch
│        ├─ scheduled_seqs.append(seq)
│        ├─ num_batched_tokens += can_prefill
│        └─ num_batched_seqs += 1
│
├─ [如果 waiting 队列空 且 decoding 队列非空] → 进行 Decode
│  │
│  └─ while decoding && num_batched_seqs < max_num_seqs_in_batch:
│     ├─ seq = decoding.pop()
│     ├─ scheduled_seqs.append(seq)
│     └─ num_batched_seqs += 1  (decode 每个序列只 1 token)
│
└─ 返回：(scheduled_seqs, is_prefill, preempted_seqs)
   其中 is_prefill = (len(scheduled_seqs) > 0 and waiting 非空)
```

**关键观察：** Vanilla 方法无稀疏逻辑，所以这个调度很直接 — 就是堆满 token 预算。

### 4.2 ModelRunner.run() - GPU 推理执行

```
run(seqs, is_prefill)
│
├─ [步骤1] prepare_step() - 准备输入和位置编码
│  │
│  ├─ 从 cache_manager 获取本次 batch 的：
│  │  ├─ input_ids: [sum(batch_seq_len)] (所有 token 的 ID 按序列拼接)
│  │  ├─ positions: [sum(batch_seq_len)] (位置编码)
│  │  └─ cu_seqlens_q: cumsum ([第1个seq的len, 第1+2个seq的len, ...])
│  │             (用于 Triton kernel 快速索引每个序列的起始位置)
│  │
│  └─ set_context() 保存这些信息供后续层访问
│
├─ [步骤2] sparse_controller.prepare_forward() - 初始化稀疏状态
│  │
│  └─ (对 vanilla，这是 no-op，只是为了统一接口)
│     初始化每层的稀疏视图（vanilla 时全为 None）
│
├─ [步骤3] self.model(input_ids, positions) - 前向计算
│  │
│  ├─ Embedding 层：input_ids → token embeddings
│  │
│  ├─ 逐层前向（num_layers 次循环，如 32 层）：
│  │  │
│  │  └─ for layer_idx in range(num_layers):
│  │     ├─ MultiHeadAttention：
│  │     │  ├─ set_context(now_layer_idx = layer_idx)  (告诉全局上下文当前在哪一层)
│  │     │  │
│  │     │  ├─ Q, K, V = linear_qkv(hidden_states)
│  │     │  │
│  │     │  ├─ k, v = apply_rotary_pos_emb(K, V, positions)
│  │     │  │          (RoPE 位置编码)
│  │     │  │
│  │     │  ├─ 【Vanilla 的关键：写入 KV Cache】
│  │     │  │  store_kvcache(k, v, k_cache, v_cache, slot_mapping)
│  │     │  │  ├─ slot_mapping 告诉每个 token 的物理位置
│  │     │  │  │  (StandardCacheManager 在 allocate 时生成)
│  │     │  │  └─ K 和 V 被直接写入 GPU 显存的固定位置
│  │     │  │
│  │     │  ├─ 【Vanilla 的关键：计算全量注意力】
│  │     │  │  attn_output = attention(Q, K_cache, V_cache, causal_mask)
│  │     │  │  ├─ Q shape: [num_tokens_in_batch, num_heads, head_dim]
│  │     │  │  ├─ K_cache shape: [num_cached_tokens, num_heads, head_dim]
│  │     │  │  │  （跨所有已处理的 token）
│  │     │  │  ├─ scores = Q @ K_cache.T / sqrt(d)  [num_tokens × num_cached_tokens]
│  │     │  │  ├─ attn_weights = softmax(scores + causal_mask)
│  │     │  │  └─ output = attn_weights @ V_cache
│  │     │  │
│  │     │  └─ (vanilla 无稀疏逻辑，K_cache/V_cache 就是全部历史 KV)
│  │     │
│  │     ├─ FeedForward：linear(act(linear(x)))
│  │     └─ LayerNorm：normalize(residual_connection)
│  │
│  └─ LMHead：hidden_states → logits [batch_size, vocab_size]
│
├─ [步骤4] sampler(logits, temperatures) - Token 采样
│  │
│  ├─ 对每个序列应用对应的 temperature
│  ├─ 按 logits 概率分布采样下一个 token
│  └─ 返回：token_ids (list of int)  [batch_size]
│
├─ [步骤5] sparse_controller.post_forward() - 后处理
│  │
│  └─ (对 vanilla，这是 no-op)
│
└─ 返回：token_ids
```

**关键点：Vanilla 的注意力计算**
- **没有 token 选择**：访问全部历史 KV
- **因果掩码**：只允许当前 token 看前面的
- **计算复杂度**：O(seq_len²) (这是为什么长上下文很慢的原因，也是为什么有 DeltaKV 等方法)

### 4.3 CacheManager：KV 缓存的物理管理

#### 4.3.1 StandardCacheManager（Vanilla 对应的实现）

```python
# src/sparsevllm/engine/cache_manager/standard.py

class StandardCacheManager:
    def __init__(self, config, rank, world_size):
        self.allocate_kv_cache()  # 一次性分配全部显存
        
        # 物理结构（全部在 GPU 显存上）
        self.kv_cache = torch.empty(
            size=[
                2,                          # K, V 两部分
                num_layers,                 # 比如 32 层
                num_kvcache_slots,          # 比如 1M 个槽位
                num_kv_heads,               # 比如 8
                head_dim,                   # 比如 128
            ],
            device="cuda"
        )
        
        # 槽位管理
        self.free_slots_stack = [0, 1, 2, ..., num_kvcache_slots-1]  # 可用槽位栈
        self._num_free_slots = num_kvcache_slots
        
        # 序列→行的映射（每个序列一行，记录该序列的所有 token 位置）
        self.buffer_req_to_token_slots = torch.zeros(
            [max_num_seqs, max_model_len],
            dtype=torch.int32,
            device="cuda"
        )
        # 例如：buffer_req_to_token_slots[seq_id, :] = [100, 101, 102, 103, ...]
        #       表示这个序列的 token 存在槽位 100, 101, 102, 103...
```

#### 4.3.2 Token 分配的生命周期

```
[序列1 生命周期]

① 序列刚到达时（add_request）：
   - 无显存占用
   - 序列在 CPU 端

② Prefill 阶段（某个 step 中被调度）：
   假设序列有 100 个 prompt token，chunk_prefill_size=50
   
   第1个 prefill chunk：
   ├─ cache_manager._allocate(seq_id=10, size=50)
   │  ├─ n = 50
   │  ├─ free_slots_stack[-50:] = [1000..1049]  抽出 50 个槽位
   │  ├─ buffer_req_to_token_slots[10, 0:50] = [1000..1049]
   │  └─ _num_free_slots -= 50  （显存预算减少）
   │
   └─ GPU 上处理这 50 个 token，K/V 写入槽位 [1000..1049]
   
   第2个 prefill chunk（下一个 step）：
   ├─ cache_manager._allocate(seq_id=10, size=50)
   │  ├─ buffer_req_to_token_slots[10, 50:100] = [1050..1099]
   │
   └─ GPU 上处理剩余 50 个 token，K/V 写入槽位 [1050..1099]

③ Decode 阶段（序列从 prefill 转入 decode）：
   每个 step 生成 1 个新 token
   
   第1个 decode step：
   ├─ cache_manager._allocate_batch([10], size=1)
   │  ├─ n = 1
   │  ├─ free_slots_stack[-1] = [1100]
   │  ├─ buffer_req_to_token_slots[10, 100] = 1100
   │  └─ _num_free_slots -= 1
   │
   └─ GPU 生成1个 token，K/V 写入槽位 1100

   第2个 decode step：
   ├─ cache_manager._allocate_batch([10], size=1)
   │  ├─ buffer_req_to_token_slots[10, 101] = 1101
   │
   └─ GPU 生成1个 token，K/V 写入槽位 1101

④ 序列完成（生成到 EOS 或达到 max_tokens）：
   cache_manager.free_seq(seq_id=10)
   ├─ 收集所有已占用的槽位：[1000..1101]
   ├─ 把这些槽位放回 free_slots_stack
   └─ _num_free_slots += 102
```

**核心数据结构：**

```
┌────────────────────────────────────────────────────────────────┐
│  free_slots_stack (栈结构，LIFO)                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [1100, 1099, 1098, ..., 3, 2, 1, 0]                    │   │
│  │ ▲top (需要的时候从这里抽)                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
├────────────────────────────────────────────────────────────────┤
│  buffer_req_to_token_slots [max_num_seqs, max_model_len]      │
│                                                                │
│  seq_id=0: [100, 101, 102, ..., -1, -1, ...]  (前103个是有效的)│
│           (这个序列占用槽位 100-102)                            │
│                                                                │
│  seq_id=1: [1000, 1001, ..., -1, -1, ...]                    │
│            (这个序列占用槽位 1000-1049)                        │
│                                                                │
│  ...                                                           │
└────────────────────────────────────────────────────────────────┘
```

---

## 5. Attention 计算：从缓存到输出

这是最核心的部分。Vanilla 的注意力就是标准的全量注意力。

### 5.1 Attention 写入阶段（KV Cache 存储）

```python
# src/sparsevllm/layers/attention.py

def forward(self, q, k, v):
    # 1. 获取缓存信息
    cache_manager = context.cache_manager
    store_k_cache, store_v_cache, slot_mapping = cache_manager.get_layer_store_view(layer_idx)
    
    # slot_mapping: [num_tokens_in_batch]
    # 例如 [1000, 1001, 1002] 表示这 3 个 token 要写入槽位 1000, 1001, 1002
    
    # 2. 写入 KV（物理操作）
    store_kvcache(k, v, store_k_cache, store_v_cache, slot_mapping)
    #         ↓       ↓     ↓              ↓               ↓
    #     当前KV  全部cache  全部cache    slot位置信息
    
    # 写入后的效果：
    # store_k_cache[1000] = k[0]
    # store_k_cache[1001] = k[1]
    # store_k_cache[1002] = k[2]
    
    # 3. 标记"已存储"
    cache_manager.on_kv_stored(layer_idx, k, slot_mapping)
```

### 5.2 Attention 读取阶段（计算注意力）

```python
def forward(self, q, k, v):
    # ... 写入阶段（上面已示）...
    
    # 4. 获取读取视图（对 vanilla，就是全部缓存）
    layer_active_slots = cache_manager.get_layer_buffer_req_to_token_slots(layer_idx)
    # vanilla 时，layer_active_slots 包含当前 batch 所有序列的所有 token 槽位
    
    # 例如 batch 中有 2 个序列：
    # seq_0: tokens在槽位 [100, 101, 102, 103]，现在要加第4个
    # seq_1: tokens在槽位 [200, 201]，现在要加第3个
    # layer_active_slots = [[100,101,102,103,-1, ...],  ← seq_0 的 req-to-token
    #                       [200,201,-1, -1, ..., ...]]  ← seq_1 的 req-to-token
    
    k_cache, v_cache = cache_manager.get_layer_kv_cache(layer_idx)
    # k_cache shape: [num_total_slots, num_heads, head_dim] e.g. [1M, 8, 128]
    # v_cache shape: [num_total_slots, num_heads, head_dim] e.g. [1M, 8, 128]
    
    # 5. 执行注意力计算（Triton kernel）
    attn_output = context_attention_fwd(
        q,                      # [num_tokens, num_heads, head_dim]
        k_cache,                # [num_total_slots, num_heads, head_dim]
        v_cache,                # [num_total_slots, num_heads, head_dim]
        o,                      # 输出 buffer
        req_indices,            # [num_tokens] 每个 token 属于哪个序列
        start_locs,             # [batch_size] prefill 时的序列起始位置
        context_lens,           # [num_tokens] 每个 token 能看多少个历史 token
        layer_active_slots,     # req-to-token 映射表
    )
    
    # kernel 内部逻辑（伪代码）：
    # for token_idx in range(num_tokens):
    #     seq_id = req_indices[token_idx]
    #     max_cache_len = context_lens[token_idx]
    #     
    #     # prefill 时的因果掩码：当前 token 只能看自己之前的
    #     # decode 时：当前 token 什么都能看（因为只有 1 个新 token）
    #     
    #     cache_indices = layer_active_slots[seq_id, :max_cache_len]  # 这个序列的有效缓存槽位
    #     K_cached = k_cache[cache_indices]  # 从全局缓存中 gather 出来
    #     V_cached = v_cache[cache_indices]  # 从全局缓存中 gather 出来
    #     
    #     scores = Q[token_idx] @ K_cached.T / sqrt(d)  [max_cache_len]
    #     attn_weights = softmax(scores)  [max_cache_len]
    #     output[token_idx] = attn_weights @ V_cached  [head_dim]
    
    return attn_output
```

---

## 6. Vanilla vs. 稀疏方法：关键差异

| 特性 | Vanilla | DeltaKV / OmniKV / SnapKV |
|-----|---------|--------------------------|
| **稀疏方法标识** | `""` (空) | `"deltakv"` / `"omnikv"` / ... |
| **CacheManager** | `StandardCacheManager` | `DeltaKVCacheManager` / ... |
| **KV 存储** | 线性存储全部 token | 压缩存储 (DeltaKV) / 物理驱逐 (SnapKV) |
| **注意力计算** | $$O(seq\_len^2)$$ | $$O(seq\_len \cdot K)$$ (K 是稀疏窗口大小) |
| **复杂性** | 无稀疏逻辑 | 复杂的重构/驱逐/掩码逻辑 |
| **长上下文性能** | 很差 (O(n²) 灾难) | 好得多 |

---

## 7. 完整的推理流程图表

```
llm.generate(prompts, sampling_params)
│
├─ [初始化] LLM.__init__() 
│  ├─ Config 解析
│  ├─ ModelRunner + CacheManager 初始化
│  ├─ Scheduler 初始化
│  └─ _warmup()
│
├─ [循环] while not is_finished():
│  │
│  │  step()
│  │  │
│  │  ├─── [1] Scheduler.schedule()
│  │  │     ├─ if waiting: prefill chunks
│  │  │     └─ elif decoding: next tokens
│  │  │
│  │  ├─── [2] Free preempted seqs
│  │  │
│  │  ├─── [3] ModelRunner.run(seqs, is_prefill)
│  │  │     │
│  │  │     ├─ prepare_step() → input_ids, positions
│  │  │     │
│  │  │     ├─ model.forward(input_ids, positions)
│  │  │     │  │
│  │  │     │  ├─ [Prefill] 处理最多 chunk_prefill_size 个 token
│  │  │     │  │  ├─ Layer 0-31 的 MultiHeadAttention
│  │  │     │  │  │  ├─ store_kvcache(K, V)  ← KV 写入显存
│  │  │     │  │  │  └─ context_attention_fwd()  ← 全量注意力
│  │  │     │  │  └─ Layer 0-31 的 FeedForward
│  │  │     │  │
│  │  │     │  └─ [Decode] 处理 1 个新生成的 token per seq
│  │  │     │     └─ 同样的 layer loop，但 Q 只有 1 个 token
│  │  │     │
│  │  │     ├─ sampler(logits) → token_ids
│  │  │     │
│  │  │     └─ sparse_controller.post_forward()  (vanilla 无操作)
│  │  │
│  │  ├─── [4] scheduler.postprocess(seqs, token_ids, is_prefill)
│  │  │     ├─ Append token_ids to seq.token_ids
│  │  │     ├─ Update num_tokens, num_prefilled_tokens
│  │  │     └─ Check for EOS / max_length → mark as FINISHED
│  │  │
│  │  └─── [5] Free finished seqs
│  │        for seq in seqs:
│  │            if seq.is_finished:
│  │                cache_manager.free_seq(seq.seq_id)
│  │                (槽位放回 free_slots_stack)
│  │
│  │  [回到 while 循环顶部]
│  │
│  └─ [当 waiting + decoding 队列都空时]
│     is_finished() == True，推出循环
│
└─ 返回所有完成序列的生成结果
```

---

## 8. 关键代码位置速查表

如果你想自己深入代码，这张表会很有用：

| 功能 | 文件 | 函数 / 类 |
|-----|------|----------|
| **配置管理** | `config.py` | `Config` dataclass |
| **推理入口** | `llm.py` | `LLM` class |
| **推理引擎** | `engine/llm_engine.py` | `LLMEngine` class |
| **调度逻辑** | `engine/scheduler.py` | `Scheduler.schedule()` |
| **模型执行** | `engine/model_runner.py` | `ModelRunner.run()` |
| **缓存管理** | `engine/cache_manager/standard.py` | `StandardCacheManager` |
| **注意力计算** | `layers/attention.py` | `Attention.forward()` |
| **采样** | `layers/sampler.py` | `Sampler` class |
| **稀疏控制** | `engine/sparse_controller.py` | `SparseController` class |

---

## 9. 常见问题 & 排查

### Q1: 为什么第一次运行会很慢？
**A**: `_warmup()` 在初始化时执行，会触发 Triton kernels 的 JIT 编译。这是一次性成本。

### Q2: Vanilla 和 OmniKV 的性能差异？
**A**: 
- **Vanilla**: 注意力 O(seq_len²)，长文本时显存和计算都爆炸
- **OmniKV**: 只保留 top-K token，O(seq_len·K)，快很多

### Q3: KV Cache 是怎么分配的？
**A**: 一开始根据 `gpu_memory_utilization` 一次性分配全部显存给 KV Cache（转化为 slots），然后按需分配给序列。

### Q4: 如果显存爆了怎么办？
**A**: 减少 `max_num_batched_tokens`、`max_num_seqs_in_batch`、`gpu_memory_utilization`，或者减少提示词长度。

---

## 10. 性能分析建议

### 看吞吐量瓶颈：

```bash
# 开启 profiler
llm = LLM(..., enable_profiler=True)

# 运行推理
llm.generate(prompts, sampling_params)

# 查看 profiler 输出
# 看 schedule / model_run_prefill / model_run_decode / sampler 各占多少时间
```

### 常见瓶颈：
1. **schedule 很慢** → 调度器有问题（通常不会）
2. **prefill 很慢** → 矩阵大，正常；减小 chunk_prefill_size
3. **decode 很慢** → 通常是显存延迟，考虑 GPU 架构
4. **sampler 很慢** → logits 转移和采样，通常忽略不计

---

## 总结

Sparse-vLLM 的 Vanilla 推理流程就是：
1. **初始化**：加载模型、分配 KV Cache、启动 Scheduler
2. **循环**：反复 schedule → run_model → update_state → free_resources
3. **Attention**：标准全量注意力，无稀疏优化

这套框架设计得很模块化，DeltaKV/OmniKV/SnapKV 等稀疏方法就是通过替换 `CacheManager` 和增加 `SparseController` 的逻辑来实现的。一旦你理解了 Vanilla，稀疏方法就相对容易理解了。

---

**作者笔记**：本文对应 Sparse-vLLM 代码库在 2026年4月的版本。如有变动，以代码为准。
