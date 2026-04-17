# Vanilla 推理流程快速参考

> 思路乱的时候，看这个

## 三句话总结

1. **初始化**：`LLM()` 启动模型、分配显存、创建调度器
2. **循环**：重复调用 `step()`，每次决定运行哪些序列 → GPU 执行 → 更新状态
3. **Attention**：标准全量注意力，所有历史 token 都参与计算

---

## 单个 step() 的核心 5 阶段

```
step() 执行以下 5 个阶段（缺一不可）：

阶段1️⃣  | Scheduler.schedule()
        决定本次运行谁
        returns: (seqs_to_run, is_prefill, evicted_seqs)

阶段2️⃣  | Free evicted seqs
        释放不要的序列占用的显存

阶段3️⃣  | ModelRunner.run(seqs, is_prefill)
        GPU上执行前向计算
        ├─ 写入 KV Cache （存储K/V到显存槽位）
        ├─ 计算注意力    （全量，O(n²)）
        ├─ 采样 token   （选择下一个词）
        └─ 返回 token_ids

阶段4️⃣  | Postprocess
        更新序列内部状态
        ├─ append token_ids
        ├─ 更新 num_tokens, num_prefilled_tokens
        └─ 检查是否 EOS/达到最大长度

阶段5️⃣  | Free finished seqs
        完成的序列释放所有显存
```

---

## Vanilla 最小化实现（伪代码）

```python
class VanillaInference:
    
    def __init__(self, model_path):
        self.config = load_config(model_path)
        self.model = load_model(model_path)
        self.cache_mgr = StandardCacheManager(self.config)
        self.scheduler = Scheduler(self.config, self.cache_mgr)
    
    def add_request(self, prompt_text):
        token_ids = tokenizer.encode(prompt_text)
        seq = Sequence(token_ids)
        self.scheduler.waiting.append(seq)
    
    def step(self):
        # 1. 调度
        seqs, is_prefill, _ = self.scheduler.schedule()
        
        # 2. 准备输入
        input_ids, positions = self.cache_mgr.prepare_step(seqs, is_prefill)
        
        # 3. 前向计算
        logits = self.model(input_ids, positions)
        
        # 4. 采样
        token_ids = sample_topk(logits)
        
        # 5. 写入 KV Cache
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            # (K/V 已在 model.forward() 中自动写入)
        
        # 6. 状态更新
        if is_prefill:
            self.scheduler.move_to_decode(seqs)  # if prefill完成
        else:
            pass  # decode序列的更新更简单
        
        # 7. 完成清理
        for seq in seqs:
            if seq.is_finished():
                self.cache_mgr.free_seq(seq.seq_id)
    
    def generate(self, prompt_text, max_tokens=100):
        self.add_request(prompt_text)
        while not self.scheduler.is_finished():
            self.step()
        return collect_results()
```

---

## KV Cache 的物理图像

```
显存布局（简化图）：

┌─────────────────────────────────────┐
│  KV Cache 缓存池                    │
│  [槽位 0]  ← K/V for token 0      │
│  [槽位 1]  ← K/V for token 1      │
│  [槽位 2]  ← K/V for token 2      │
│  ...                               │
│  [槽位 N]  ← K/V for token N      │
│  [槽位 N+1] ← 可用（空）           │
│  ...                               │
└─────────────────────────────────────┘
            ↑
       物理槽位号
       （显存地址）


逻辑视图（per 序列）：

序列1：tokens=[1,2,3,4,5]
       槽位=[100,101,102,103,104]
       意思是：token 1 存在槽位 100；token 2 存在槽位 101；...

序列2：tokens=[10,11]
       槽位=[200,201]
```

---

## Attention 计算流程（简化）

```
当前 forward() 处理 B 个 token（来自多个序列）：

step 1: Compute Q, K, V
        Q: [B, H, d]
        K: [B, H, d]
        V: [B, H, d]

step 2: Store K, V to GPU cache
        K → GPU_cache[slot[0]], GPU_cache[slot[1]], ...
        V → GPU_cache[slot[0]], GPU_cache[slot[1]], ...
        其中 slot = cache_mgr 分配的物理位置

step 3: Load all historical K, V
        K_all = GPU_cache[all historical slots]  [总长度, H, d]
        V_all = GPU_cache[all historical slots]  [总长度, H, d]
        
step 4: Compute attention
        scores = Q @ K_all^T / sqrt(d)  [B, 总长度]
        attn_w = softmax(scores)         [B, 总长度]
        output = attn_w @ V_all          [B, H, d]
        
        ↑ 这就是 O(n²) 为什么这么贵！n 是序列总长度
```

---

## 常用参数一览

| 参数 | 范围 | 对 Vanilla 的影响 |
|-----|------|-----------------|
| `gpu_memory_utilization` | 0.7-0.9 | 值大 = 分配更多显存给 KV Cache |
| `max_num_batched_tokens` | 1024-65536 | 值大 = 一个 batch 能处理更多 token |
| `chunk_prefill_size` | 512-16384 | 只对 prefill 的长序列有影响，分块处理 |
| `max_num_seqs_in_batch` | 1-128 | 最多同时处理多少个序列 |
| `max_model_len` | 4096-1M | 单个序列最大长度 |

**对 Vanilla 的建议**：
- `gpu_memory_utilization=0.85` （激进）
- `max_num_batched_tokens=16384` （中等）
- `chunk_prefill_size=4096` （中等）

---

## 排查思路

如果出问题，按这个顺序检查：

```
1️⃣  显存不足？
    → 减小 gpu_memory_utilization 或 max_num_batched_tokens

2️⃣  速度很慢？
    → step() 里哪个阶段慢？用 profiler 检查
    → 如果 model_run 慢，是 prefill 还是 decode？
    → prefill 慢很正常；decode 慢可能是显存延迟

3️⃣  某个序列卡住？
    → scheduler.waiting / scheduler.decoding 里是否有队列溢出？
    → 是否显存已满，无法分配新序列？

4️⃣  结果不正确？
    → 检查 sampling_params 的温度是否 > 1e-10
    → 检查 EOS token 设置是否正确
```

---

## 文件导航

| 想查看... | 看这个文件 |
|--------|-----------|
| 完整详细版本 | `vanilla-inference-flow_zh.md` (本目录) |
| 配置所有参数 | `../config.py` (搜索 Config dataclass) |
| 调度算法 | `../engine/scheduler.py` (Scheduler.schedule) |
| KV Cache 管理 | `../engine/cache_manager/standard.py` |
| Attention 实现 | `../layers/attention.py` (Attention.forward) |
| 模型执行入口 | `../engine/model_runner.py` (ModelRunner.run) |

---

## 记住这张图

```
User Code
   │
   ↓
llm.generate(prompts, sampling_params)
   │
   ├─ llm.__init__()  [一次性]
   │  ├─ Load model
   │  ├─ Allocate KV cache
   │  └─ Init scheduler
   │
   ├─ while not done:
   │  │
   │  └─ step()
   │     ├─ Scheduler decides who runs
   │     │
   │     ├─ GPU computes
   │     │  ├─ for each layer:
   │     │  │  ├─ Q, K, V = linear_qkv(...)
   │     │  │  ├─ store K, V to cache
   │     │  │  ├─ attn = attention(Q, K_cache, V_cache)
   │     │  │  └─ x = ffn(attn)
   │     │  │
   │     │  └─ sample token_ids
   │     │
   │     └─ Update state & free memory
   │
   └─ Return results

简单！就这么多。
```
