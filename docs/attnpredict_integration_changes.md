# AttentionPredictor 集成变更文档

本文档记录将 AttentionPredictor 方法集成到 Sparse-vLLM 推理引擎的所有代码变更。

## 变更概览

- **新增文件**: 3 个 (~495 行)
- **修改文件**: 6 个 (~63 行)
- **后续修正**: `src/sparsevllm/layers/attention.py` 已新增通用 prefill observer hook 调用，保持方法无关。

> 当前状态说明：本文档保留初始集成记录，同时补充 Codex review 后的修正。当前实现中，decode `with_score` kernel 写入的是未乘 `sm_scale`、未 softmax 的 raw logits，所以 `predict_next_mask()` 中的 `softmax(logits * attn_scale)` 是必要转换，不是二次 softmax。当前 v1 采用整层共享 token mask，不是原始 `OffloadedCache` 的 per-head KV mask；这是现有 Sparse-vLLM decode view/kernel 接口下的精度取舍。

## 一、新增文件

### 1. `src/sparsevllm/models/llama.py` (195 行)

**功能**: LLaMA 3.1 模型定义，使 Sparse-vLLM 支持加载 LLaMA 权重。

**类结构**:

| 类名 | 基类 | 功能 |
|------|------|------|
| `LlamaAttention` | `nn.Module` | 分离的 q/k/v 投影 + RoPE + `Attention` forward |
| `LlamaMLP` | `nn.Module` | gate/up 融合投影 + SiLU 激活 + down 投影 (SwiGLU) |
| `LlamaDecoderLayer` | `nn.Module` | Attention + MLP + RMSNorm (pre/post) |
| `LlamaModel` | `nn.Module` | Embed + Layers loop + SparseController 回调注入 |
| `LlamaForCausalLM` | `nn.Module` | Model + LM head + `packed_modules_mapping` |

**关键设计点**:

- 线性层使用 `ColumnParallelLinear` / `RowParallelLinear` 支持 Tensor Parallelism（与 Qwen2 一致）
- 区别于 Qwen2 的融合 `QKVParallelLinear`：LLaMA 用三个独立的 `q_proj/k_proj/v_proj`
- `packed_modules_mapping` 将 `gate_proj+up_proj` 合并为 `MergedColumnParallelLinear`
- RoPE 通过 `get_rope()` 创建，LLaMA 3.1 标准版 `rope_scaling=None`，兼容现有代码（无 YaRN）
- 逐层循环中通过 `context.now_layer_idx` 和 `sparse_controller.on_layer_end()` 注入稀疏回调

---

### 2. `src/sparsevllm/engine/cache_manager/attnpredict_cnn.py` (78 行)

**功能**: AttentionPredictor 的 CNN 预测器模型定义。

**结构**:

```
输入: (batch_size, history_steps=64, pooled_seq_len)
  ↓
Conv2d(1→16, kernel=3x3, padding=1) → ReLU
  ↓
Conv2d(16→32, kernel=3x3, padding=1) → ReLU
  ↓
AdaptiveAvgPool2d((1, None))
  ↓
Conv1d(32→1, kernel=1)
  ↓
输出: (batch_size, pooled_seq_len)
```

**设计要点**:
- 与原始 AttentionPredictor 论文代码完全一致
- 当前 cache manager 按 head 维护 attention history，调用 CNN 时把 head 维展平到 batch 维
- 输出每个 head、每个 block 的预测重要性分数
- v1 在生成 Sparse-vLLM decode view 前再把 per-head block score 合并为 shared token mask
- 独立文件存放，便于模型加载和未来训练

---

### 3. `src/sparsevllm/engine/cache_manager/attnpredict.py`

**功能**: AttentionPredictor 的 CacheManager，管理 attention 历史和预测流程。

**类**: `AttnPredictCacheManager(StandardCacheManager)`

继承 `StandardCacheManager`（全量 KV 在 GPU，v1 无 CPU offload）。

**核心状态**:

```python
self.attn_history: list[dict[int, torch.Tensor]]  # 每层、每 cache row 的 64 步 attention 快照
self.tsp_mask: list[dict[int, torch.Tensor]]      # 每层、每 cache row 的 token keep mask
self._last_decode_view: list[dict | None]         # 本层最近一次 decode view，用于 scatter 回完整序列
self.cnn: AttnPredictCNN                          # 全层共享的 CNN 预测器 (float16)
```

**核心方法**:

| 方法 | 功能 |
|------|------|
| `observe_prefill_attention(...)` | prefill 后取最后 64 个 query 计算 attention history，并预测首个 decode mask |
| `build_decode_view(...)` | 根据上一轮 `tsp_mask` pack selected slots，并强制保留当前 decode 新 token |
| `predict_next_mask(layer_idx, attn_logits)` | 将 raw logits 转 softmax，必要时 scatter 回完整逻辑序列，再预测下一步 mask |
| `_update_attn_history(...)` | 维护每个 cache row 的 64 步滚动窗口 |
| `_time_sequence_predict(...)` | 按 head 运行 CNN 预测 block 重要性 |
| `_create_tsp_mask(...)` | 生成 shared token keep mask：保留 sink + local + topk blocks |

**预测 Pipeline 时序**:

```
Prefill:
  1. 写入完整 K/V 到 GPU cache
  2. observe_prefill_attention() 计算最后 64 个 query 的 attention softmax
  3. max-pool → 更新 attn_history → CNN 预测 → 生成首个 decode mask

Decode Step t:
  1. build_decode_view() 使用 step t-1 的 mask pack slots
  2. flash decode with_score 写出 selected view 上的 raw logits
  3. on_layer_end → predict_next_mask():
     softmax(logits * attn_scale) → scatter 回完整序列 → 更新 history
     → CNN 预测 step t+1 的 mask
```

**关键设计决策**:
- **1-step lag**: step t 预测的 mask 在 step t+1 使用；prefill 会提前初始化首个 decode mask
- **raw logits 语义**: decode kernel 在 `att_value *= sm_scale` 前写 `attn_score`，cache manager 内部负责转 softmax
- **shared token mask**: 原始实现是 per-head mask，当前 v1 用 head 维 max-pooling 合并为整层共享 mask
- **无 CPU offload**: v1 版本全量 KV 在 GPU，当前稀疏来自逻辑 read view，不做 CPU-GPU 异步预取
- **释放序列**: `free_seq()` 只清理对应 cache row 的 history/mask，不全局清空 `_last_decode_view`

---

## 二、修改文件

### 4. `src/sparsevllm/config.py`

**变更**: 添加 AttentionPredictor 配置字段 + 方法名文档。

**新增字段** (在 QuEST 配置块下方):

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `attnpredict_topk` | `int` | `1024` | 保留的 token 总数上限 |
| `attnpredict_history_steps` | `int` | `64` | attention 历史快照步数 |
| `attnpredict_pooling_block_size` | `int` | `16` | max pooling 的 block 大小 |
| `attnpredict_sink_tokens` | `int` | `64` | 强制保留的 sink token 数 |
| `attnpredict_local_tokens` | `int` | `64` | 强制保留的 local token 数 |
| `attnpredict_model_path` | `str` | `""` | CNN checkpoint 路径 |

**`vllm_sparse_method` 文档**: 添加 `"attnpredict"` 到可选值列表。

---

### 5. `src/sparsevllm/engine/cache_manager/base.py`

**变更**: `CacheManager.create()` 工厂方法中添加 `"attnpredict"` 分支。

**位置**: `quest` 分支之前。

```python
if sparse_method == "attnpredict":
    from .attnpredict import AttnPredictCacheManager
    return AttnPredictCacheManager(config, rank, world_size)
```

**功能**: 当 `vllm_sparse_method="attnpredict"` 时，工厂方法创建 `AttnPredictCacheManager` 实例。

---

### 6. `src/sparsevllm/engine/cache_manager/__init__.py`

**变更**: 导出 `AttnPredictCacheManager` 类和懒加载分支。

- `__all__` 列表添加 `"AttnPredictCacheManager"`
- `__getattr__` 添加懒加载分支（仅在首次引用时 import）

---

### 7. `src/sparsevllm/engine/model_runner.py`

**变更**: 添加 `llama` model_type 分发。

**位置**: `qwen2` 和 `deepseek_v2` 分支之间。

```python
elif hf_config.model_type == "llama":
    from sparsevllm.models.llama import LlamaForCausalLM
    self.model = LlamaForCausalLM(hf_config)
```

**功能**: 当 HuggingFace config 中 `model_type == "llama"` 时，创建 `LlamaForCausalLM` 实例。

---

### 8. `src/sparsevllm/engine/sparse_controller.py`

**变更 1**: `_needs_attn_score()` — 添加 `"attnpredict"` 分支

```python
if self.sparse_method == 'attnpredict':
    return not is_prefill  # decode 每层都需要收集 attn_score
```

- decode 时所有层都返回 `True`（写入 attn_score 供 CNN 消费）
- prefill 返回 `False`（不做预测，全量 attention）

**变更 2**: `get_read_view()` — 添加 `"attnpredict"` 到全量 slots 方法列表

```python
if (self.sparse_method in ("omnikv", "deltakv", "attnpredict") and layer_idx in self.full_attn_layers) or \
    self.sparse_method in ('snapkv', 'pyramidkv', 'quest', 'streamingllm', 'attention-sink', 'attention_sink', 'attnpredict', ''):
```

- `attnpredict` 在此阶段返回全量 slots
- 实际的 KV 筛选发生在后续的 `build_decode_view()` 中

**变更 3**: `on_layer_end()` — 添加 `"attnpredict"` dispatch

```python
if self.sparse_method == 'attnpredict' and not context.is_prefill:
    state = self.layer_batch_sparse_states[layer_idx]
    if state.attn_score is not None:
        self.cache_manager.predict_next_mask(layer_idx, state.attn_score)
    return
```

- decode 时每层结束时触发，不等 obs layers 判断
- 不在 controller 中提前做 head max-pooling，保留 head 维交给 cache manager
- 调用 `cache_manager.predict_next_mask()` 更新预测

---

## 三、`src/sparsevllm/layers/attention.py` 当前修改

AttentionPredictor 通过通用 hook 接入，仍保持 `attention.py` 方法无关：

- `build_decode_view()` 调用 (line 220) 在 decode 分支自动生效
- prefill 分支在 `context_attention_fwd(...)` 前调用 `cache_manager.observe_prefill_attention(...)`
- 其他 sparse method 通过 `CacheManager` 基类默认 no-op 实现，不需要方法分支
- `attn_score` 写入机制通过 `gqa_flash_decode_stage1_with_score` 完成
- `finally` 块的临时 slot 清理对 `attnpredict` 是 no-op（无 temp slots）

---

## 四、调用示例

```python
from sparsevllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/Meta-Llama-3.1-8B-Instruct",
    vllm_sparse_method="attnpredict",
    attnpredict_model_path="/path/to/CNN_llama3.1_alltask_5case/best_model.pth",
    attnpredict_topk=1024,
    attnpredict_sink_tokens=64,
    attnpredict_local_tokens=64,
    gpu_memory_utilization=0.5,
)

sampling_params = SamplingParams(temperature=1e-5, max_tokens=256)
outputs = llm.generate(["Your prompt here"], sampling_params)
```

## 五、待完成工作

1. **CNN 训练**: 当前使用的是 LLaMA 3.1 训练的 CNN checkpoint。为支持 Qwen2 模型，需要：
   - 在 Qwen2 decoder layers 上收集 attention 数据
   - 训练新的 CNN 预测器
   - 将 checkpoint 路径传入 `attnpredict_model_path`

2. **CPU Offload**: v1 版本全量 KV 在 GPU。后续可参考原始论文中的 `OffloadedCache` 实现 CPU-GPU 异步 offload 以减少显存占用。

3. **端到端测试**: 需要 LLaMA 3.1-8B 权重和 CNN checkpoint 运行 `scripts/test_sparse_vllm_correctness.py` 验证输出正确性。

## 六、Codex Review 后修正

- 修正 `attn_history` 形状：按原始实现维护 `(heads, 64, pooled_len)`，而不是把 64 误当作 pooled 序列长度。
- mask/history 改为按 cache row 绑定，避免连续 batching 时按 batch 下标串序列。
- decode sparse view 后将 attention logits 转 softmax，并按逻辑位置 scatter 回完整序列，再更新历史，等价于原始实现的 `expand_attn()`。
- 明确 decode `attn_score` 是 raw logits：kernel 在 `att_value *= sm_scale` 前写出分数，因此 cache manager 内的 softmax 不是二次 softmax。
- 当前 v1 使用 shared token mask：per-head CNN 预测会在 `_create_tsp_mask()` 中合并为整层共享 keep mask，后续若要完全对齐原始 per-head mask 需要扩展 decode view/kernel。
- `free_seq()` 只释放对应 cache row 的 `attn_history` 和 `tsp_mask`，避免清空整层 `_last_decode_view` 影响同一 step 的其他活跃序列。
- prefill 阶段新增通用 cache-manager observer，计算最后 64 个 query 的 attention 来初始化 CNN 历史，使首个 decode step 可以使用预测 mask。
- CNN 固定使用 fp16，与原始实现一致；`attnpredict_model_path` 现在是必填。
- 修正 Llama 3.1 初始化所需的 `rope_scaling="llama3"` 支持。

### 跨步预取状态

当前 Sparse-vLLM 集成实现了跨步预测：step `t` 生成的 mask 会在 step `t+1` 的 `build_decode_view()` 中生效。

但它**没有实现**原始 `OffloadedCache` 的 CPU→GPU 异步 KV 预取：没有 `ThreadPoolExecutor`、`prefetch_stream`、`key_buffer/value_buffer` 或按 mask 从 CPU cache 拉取 KV 的 `get_kv()`。当前版本仍是全量 KV 常驻 GPU，只做逻辑 read view 筛选。
