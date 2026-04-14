# HF 与 Sparse-vLLM 后端参数指南

本文说明本仓库当前在你使用 `hf` 后端与 `sparsevllm` 后端运行时，是如何解释参数的。

本文有意采用“与实现细节强绑定（implementation-specific）”的视角。目标不是描述一个理想化 API。目标是描述本仓库代码今天实际在做什么，尤其是在以下场景中：

- 相同的参数名代表不同含义，
- 一个后端会忽略另一个后端会使用的参数，
- 同一种方法在两侧都存在，但配置方式不同，
- 某个基准脚本把同一个共享的 `infer_config` 字典传给两个差异很大的运行时。

## 范围

本指南覆盖以下位置暴露的后端分流：

- `src/deltakv/get_chat_api.py`
- 基准入口，例如 `benchmark/long_bench/pred.py`、`benchmark/math_bench/pred.py` 和 `benchmark/niah/test_niah.py`

这些基准脚本都会先构建一个 `infer_config` 字典，然后将其传入 `get_generate_api(...)`。这个共享前端很方便，但它会掩盖两种后端在语义上的巨大差异。

## 1. 心智模型

先用下面这个一阶近似来理解：

| 问题 | `backend="hf"` | `backend="sparsevllm"` |
| --- | --- | --- |
| 由谁选择方法？ | `model_cls` | `infer_config["vllm_sparse_method"]` |
| 由谁加载压缩器？ | 顶层 `compressor_path` | `infer_config["deltakv_path"]` |
| 由谁负责调度与 KV 内存布局？ | HF 模型 wrapper / adapter | Sparse-vLLM 引擎与 cache manager |
| 由谁负责 tokenizer / 设备初始化？ | `get_generate_api(...)` | Sparse-vLLM 内部 |
| 如何处理未知 `infer_config` 键？ | 通常记录为 unknown 然后忽略 | 在构建 `Config(...)` 前静默过滤掉 |

最重要的结论是：

> 同一个 `infer_config` 字典默认并不可移植。请把后端切换视为一次配置迁移，而不是简单切一个开关。

## 2. `get_generate_api(...)` 的顶层参数

函数签名是共享的，但两个后端并不会消费相同的顶层参数。

| 参数 | HF 后端 | Sparse-vLLM 后端 | 说明 |
| --- | --- | --- | --- |
| `model_path` | 使用 | 使用 | 两边都需要。 |
| `infer_config` | 使用 | 使用 | 容器相同，但含义不同。 |
| `compressor_path` | DeltaKV 家族 HF 路径会使用 | 忽略 | Sparse-vLLM 期望在 `infer_config` 里传 `deltakv_path`。 |
| `tokenizer_path` | 使用 | 后端构建时忽略 | 基准脚本可能仍会为截断单独加载 tokenizer，但这里 Sparse-vLLM 本身不会使用这个顶层参数。 |
| `model_cls` | 必需且有语义 | 忽略 | Sparse-vLLM 的方法选择不来自 `model_cls`。 |
| `use_cache` | 必须为 `True` | 忽略 | HF 会断言走基于 cache 的推理。 |
| `cuda_device` | 用于 `device_map` 和某些压缩器加载路径 | 忽略 | Sparse-vLLM 在自身引擎内部处理设备。 |
| `backend` | 选择路由 | 选择路由 | 这才是真正的后端开关。 |
| `return_kv_cache` | 部分支持 | 调用形状兼容，但缓存始终返回 `None` | 见下文生成章节。 |
| `return_model` | HF 支持 | 不支持 | 若 `return_model=True`，Sparse-vLLM 会抛错。 |

## 3. 生成阶段关键字参数

`get_generate_api(...)` 返回后，你会用 prompts 和生成 kwargs 调用生成函数。这里也是两个后端明显分叉的地方。

### 3.1 当前 wrapper 实际会转发什么

| 生成 kwarg | HF 手写路径 | HF `model.generate(...)` 路径 | Sparse-vLLM wrapper 路径 | 说明 |
| --- | --- | --- | --- | --- |
| `max_new_tokens` | 使用 | 使用 | 使用 | Sparse-vLLM 会将其映射到 `SamplingParams.max_tokens`。 |
| `max_tokens` | 非主参数 | 非主参数 | 作为回退使用 | Sparse-vLLM 优先接收 `max_new_tokens`，再回退到 `max_tokens`。 |
| `do_sample` | 使用 | 使用 | 间接使用 | Sparse-vLLM 仅用它决定 `temperature` 是否变成 `0.0`。 |
| `temperature` | 使用 | 采样时使用 | 使用 | Sparse-vLLM 在采样时会把极小正数钳到 `1e-5`。 |
| `top_p` | 使用 | 采样时使用 | 当前 wrapper 忽略 | 即便以后 Sparse-vLLM 内部支持更多采样参数，这个 wrapper 今天也不会转发 `top_p`。 |
| `top_k` | 使用 | 采样时使用 | 当前 wrapper 忽略 | 与 `top_p` 相同的注意事项。 |
| `eos_token_id` | 使用 | 使用 | 当前 wrapper 忽略 | Sparse-vLLM wrapper 不会把它映射进 `SamplingParams`。 |
| `num_beams` | 在通用 HF 路径中基本不支持 | 在通用 HF 路径中基本不支持 | 忽略 | `kvzip` 明确要求 `num_beams=1`。不要假设 beam search 可用。 |
| `past_key_values` | 支持 | 明确拒绝 | 忽略 | Sparse-vLLM wrapper 为保持调用签名兼容会接收该参数，但并不会使用。 |

### 3.2 返回侧行为

| 行为 | HF 后端 | Sparse-vLLM 后端 |
| --- | --- | --- |
| `return_kv_cache=True` | 仅尽力支持 | 缓存载荷始终返回 `None` |
| 实际返回 cache 对象 | 仅 HF 手写生成路径 | 从不返回 |
| `return_model=True` | 返回 `(generate_fn, model)` | 抛出错误 |

换句话说：

- 如果你需要复用外部 `past_key_values`，这属于 HF 专属能力。
- 如果你需要在两个后端间可移植的接口，请默认只有文本输出可移植。

## 4. 方法选择机制不同

不要尝试在两边使用同一套方法选择字段。

| 目标 | HF 后端 | Sparse-vLLM 后端 |
| --- | --- | --- |
| 选择 DeltaKV | `model_cls="deltakv"` 或相关 HF 分支 | `vllm_sparse_method="deltakv"` 或某个 Sparse-vLLM DeltaKV 变体 |
| 选择 OmniKV | `model_cls="omnikv"` | `vllm_sparse_method="omnikv"` |
| 选择 SnapKV | `model_cls="snapkv"` | `vllm_sparse_method="snapkv"` |
| 选择 PyramidKV | `model_cls="pyramidkv"` | `vllm_sparse_method="pyramidkv"` |
| 选择 Quest | `model_cls="quest"` | `vllm_sparse_method="quest"` |

两边支持的方法集合并不相同。

### 本仓库中的 HF `model_cls` 分支

- `deltakv`
- `full_deltakv`
- `origin_residual_quant`
- `all_origin_residual_quant`
- `snapkv`
- `deltasnapkv`
- `pyramidkv`
- `omnikv`
- `auto`
- `quest`
- `palu`
- `kivi`
- `adakv`
- `kvzip`

### 本仓库中的 Sparse-vLLM `vllm_sparse_method` 字符串

- `""`：用于稠密/默认引擎行为
- `streamingllm`
- `attention-sink` / `attention_sink`：作为 `streamingllm` 的别名
- `snapkv`
- `omnikv`
- `quest`
- `deltakv`
- `deltakv-triton`
- `deltakv-triton-v2`
- `deltakv-triton-v3`
- `deltakv-triton-v4`
- `deltakv-triton-v3-offload`
- `deltakv-triton-v3-cuda-offload`
- `deltakv-standalone`
- `deltakv-snapkv`
- `pyramidkv`
- `dsa`

有些方法只存在于一侧。`kvzip`、`palu`、`kivi`、`adakv` 是本仓库中的 HF 侧概念。`dsa` 是 Sparse-vLLM 侧概念。

## 5. `infer_config` 键：同名，不同义

以下参数最需要额外注意。

| 键 | HF 后端含义 | Sparse-vLLM 含义 | 建议 |
| --- | --- | --- | --- |
| `chunk_prefill_size` | 主要是模型侧或 wrapper 侧的分块控制参数 | 引擎调度与内存预算参数 | 不要把同一个数值在后端间盲目照搬。 |
| `max_model_len` | 主要是模型侧逻辑或外层基准截断看到的上限 | 影响分配与准入控制的硬性引擎容量参数 | 在 Sparse-vLLM 上需要重新调参，不要把它当元数据。 |
| `num_top_tokens` | 在 HF 的 token 选择代码里可能是整数计数或浮点比例 | 在 Sparse-vLLM 里按“类整数 token 预算”处理 | 迁移到 Sparse-vLLM 前，把浮点比例换成显式计数。 |
| `num_top_tokens_in_prefill` | 模型侧 prefill 保留预算 | 引擎可见保留预算，也参与 warmup 与容量规划 | 需要与 `chunk_prefill_size`、批处理上限一起重看。 |
| `cluster_ratio` | 在 HF wrapper 中主要是算法比例 | 在 Sparse-vLLM 中既是算法比例，也是内存/容量输入 | 在 Sparse-vLLM 上把它当作同时影响性能与内存的参数。 |
| `full_attn_layers` | DeltaKV / OmniKV 风格逻辑使用的 wrapper 配置 | 可自动派生 `obs_layer_ids` 的混合层路由信号 | 不要假设同一层列表会导向相同运行行为。 |
| `num_recent_tokens` | 对外暴露为 `num_recent_tokens`，但很多 HF wrapper 内部用 `tail_token_size` | 由引擎 cache manager 与调度数学直接消费 | 直觉相近，但实现管线不同。 |
| `pool_kernel_size` | HF token 选择 helper 的平滑核参数 | 依方法而定的引擎侧稀疏控制参数 | 名字相同，但消费路径不同。 |

### 5.1 `chunk_prefill_size`

这是最容易误导人的“共享”名称之一。

在 HF 上：

- 它通常控制 wrapper 如何对长 prefill 分块，
- 某些方法会用非常大的哨兵值等效关闭分块，
- 它并不表示“在一个持久的、由调度器拥有的 KV 池中预留这个 chunk 大小”。

在 Sparse-vLLM 上：

- 它参与 warmup 长度计算，
- 它约束 prefill 调度，
- 它影响 `max_num_batched_tokens`，
- 它直接进入 cache 容量计算与 OOM 规避逻辑。

实践规则：

> 在 HF 中，较大的 `chunk_prefill_size` 往往表示“避免 wrapper 分块”；在 Sparse-vLLM 中，较大的 `chunk_prefill_size` 表示“改变引擎行为与内存规划”。

### 5.2 `num_top_tokens`

在 HF 上，token 选择 helper 接收 `Union[int, float]`，并将 `<= 1.0` 的浮点数解释为候选 token 的比例。

示例：

- `num_top_tokens=2048` 表示“保留 2048 个 token”
- `num_top_tokens=0.11` 表示“保留大约 11% 的候选 token”

在 Sparse-vLLM 上，这个参数在多个引擎和 cache-manager 路径里按类整数预算使用。传 `0.11` 不是一种可移植的“11%”表达方式，它会变成无效或误导性的预算。

实践规则：

> 浮点比例是 HF 侧的便利语法。Sparse-vLLM 期望显式预算值。

### 5.3 `full_attn_layers`

同一份层列表仍可能导向不同的行为。

在 HF 上：

- 它会传入 wrapper 配置，
- 它帮助定义在 DeltaKV/OmniKV 风格方法中哪些层保持全注意力，
- `deltasnapkv` 明确要求它必须为空。

在 Sparse-vLLM 上：

- 它会被解析为列表并可能驱动混合层路由，
- `obs_layer_ids` 可能由它自动派生，
- 对于 `deltakv-standalone` 和 `deltakv-snapkv`，Sparse-vLLM 会强制清空 `full_attn_layers` 与 `obs_layer_ids`。

实践规则：

> 仅匹配字符串值还不够。请检查每个后端的方法特定归一化规则。

## 6. `infer_config` 键：同一概念，不同名称

有些想法两边都有，但拼写不同。

| 概念 | HF 侧参数 | Sparse-vLLM 侧参数 | 说明 |
| --- | --- | --- | --- |
| DeltaKV 检查点路径 | 顶层 `compressor_path` | `infer_config["deltakv_path"]` | 这是最常见的迁移错误。 |
| DeltaKV 最近中心数量 | `k_neighbors`（可从 `seq_chunk_size` 的旧逻辑回退） | `deltakv_k_neighbors` | 概念类似，但不是共享键。 |
| Quest 页/块大小 | `chunk_size` | `quest_chunk_size` | HF Quest 使用它自己的 adapter 命名。 |
| Quest token 预算 | `num_top_tokens` | `quest_token_budget` | 高层意图相同，但键名不同。 |
| 方法选择 | `model_cls` | `vllm_sparse_method` | 一个在顶层，一个在 `infer_config`。 |

如果你切后端时不重命名这些参数，任务可能仍能启动，但可能在使用默认值，而不是你想要的配置。

## 7. `infer_config` 键：直觉上大多共享

这类参数更接近可移植，但仍不能保证在不同运行时下数值等价。

| 键 | 高层含义 |
| --- | --- |
| `num_sink_tokens` | sink 风格稀疏方法中保留的 sink/prefix token 数 |
| `snapkv_window_size` | SnapKV 类逻辑里的局部 recent-window 预算 |
| `kv_compressed_size` | DeltaKV 风格路径中的潜变量/压缩 KV 宽度 |
| `cluster_metric` | DeltaKV 聚类 / 匹配的距离或相似度度量 |
| `kv_quant_bits` | DeltaKV 风格状态的量化/压缩控制 |

这些键比前几节更容易理解，但你仍应先验证方法特定代码路径，再假设性能或内存行为是一一对应的。

## 8. HF 专属或主要面向 HF 的参数

如果把这些参数传给 Sparse-vLLM 后端，通常会被忽略，因为它们不是 Sparse-vLLM `Config` 字段。

### 8.1 HF 模型加载与量化参数

| 键 | 在 HF 上的含义 | Sparse-vLLM 行为 |
| --- | --- | --- |
| `torch_dtype` | 控制 HF 模型加载 dtype | 在 Sparse-vLLM `infer_config` 过滤中被忽略 |
| `load_in_4bit` / `load_in_8bit` | BitsAndBytes 加载期量化 | 忽略 |
| `quant_skip_modules` | 从低比特加载中排除模块 | 忽略 |
| `llm_int8_threshold` | BitsAndBytes int8 调优参数 | 忽略 |
| `llm_int8_enable_fp32_cpu_offload` | BitsAndBytes int8 offload | 忽略 |
| `llm_int8_has_fp16_weight` | BitsAndBytes int8 选项 | 忽略 |
| `bnb_4bit_compute_dtype` | BitsAndBytes 4-bit 计算 dtype | 忽略 |
| `bnb_4bit_use_double_quant` | BitsAndBytes 4-bit 选项 | 忽略 |
| `bnb_4bit_quant_type` | BitsAndBytes 4-bit 选项 | 忽略 |
| `bnb_4bit_quant_storage` | BitsAndBytes 4-bit 存储 dtype | 忽略 |

### 8.2 HF DeltaKV-wrapper 参数

这些参数对 HF wrapper 家族有意义，但与 Sparse-vLLM 引擎构建无关：

- `use_compression`
- `use_cluster`
- `collect_kv_before_rope`
- `split_kv`
- `recon_mode`
- `ref_mode`
- `cluster_on_kv`
- `stride_alpha`
- `cluster_temp`
- `cluster_soft_assignment`
- `k_neighbors`
- `seq_chunk_size`
- `layer_chunk_size`

### 8.3 HF baseline 特有参数

这些参数存在的原因是 HF 后端可以分发到仓库内 baseline adapter。

| 方法 | 代表性参数 |
| --- | --- |
| `palu` | `lt_bits`、`lt_group_size`、`lt_sym`、`lt_clip_ratio`、`lt_hadamard` |
| `kivi` | `k_bits`、`v_bits`、`group_size`、`residual_length` |
| `adakv` | `use_adaptive`、`kernel_size`、`pooling`、`floor_alpha`、`pyram_mode`、`pyram_beta`、`gqa_support`、`gqa_func` |
| `kvzip` | `kv_type`、`prefill_chunk_size`、`ratio`、`level`、`load_score`、`do_score`、`update_cache`、`iterative_expand_chunk_size`、`iterative_keep_tokens`、`iterative_decode_chunk_size`、`iterative_decode_keep_tokens` |

这些都不是 Sparse-vLLM 设置，即使在别处有类似研究思路也一样。

## 9. Sparse-vLLM 专属参数

如果把这些参数传给 HF 后端，通常会出现以下结果之一：

- 它们会被记录为未知的自定义配置键，
- 它们被忽略，
- 只有在所选 HF `model_cls` 恰好手动消费它们时才会生效。

### 9.1 Sparse-vLLM 引擎容量与调度参数

| 键 | 在 Sparse-vLLM 上的含义 |
| --- | --- |
| `gpu_memory_utilization` | 为引擎 KV/cache 规划预算的 GPU 显存占比 |
| `max_num_batched_tokens` | 调度器全局 token 吞吐上限 |
| `max_num_seqs_in_batch` | 调度批次基数上限 |
| `max_decoding_seqs` | 最大并发解码序列数 |
| `tensor_parallel_size` | Sparse-vLLM worker rank 数量 |
| `enforce_eager` | 引擎执行策略开关 |
| `num_kvcache_slots` | 显式 KV 槽位覆盖值 |
| `enable_profiler` | 引擎 profiler 开关 |
| `throughput_log_interval_s` | 吞吐日志间隔 |

### 9.2 Sparse-vLLM 方法路由与 cache-manager 参数

| 键 | 在 Sparse-vLLM 上的含义 |
| --- | --- |
| `vllm_sparse_method` | 主方法选择器 |
| `deltakv_path` | DeltaKV 检查点路径 |
| `obs_layer_ids` | 观察层显式覆盖 |
| `chunk_prefill_accel_omnikv` | OmniKV prefill 加速开关 |
| `quest_chunk_size` | Quest 页面大小 |
| `quest_token_budget` | Quest token 预算 |
| `quest_skip_layers` | Quest 跳层控制 |
| `snapkv_num_full_layers` | SnapKV 开始驱逐前保留全注意力的层数 |
| `pyramid_layer_ratios` | PyramidKV 的显式逐层 ratio |
| `pyramidkv_start_layer`, `pyramidkv_start_ratio`, `pyramidkv_least_layer`, `pyramidkv_least_ratio` | PyramidKV 自动 ratio 生成控制参数 |
| `deltakv_full_pool_reserve_ratio` | Sparse-vLLM DeltaKV 全量 KV 池预留比例 |
| `deltakv_offload_latent` 及相关 offload 参数 | Sparse-vLLM DeltaKV 的 offload 策略 |
| `dsa_*` 字段 | DeepSeek Sparse Attention / FlashMLA 引擎参数 |

## 10. 未知键行为

这很重要，因为“任务没崩溃”并不等于“参数生效了”。

### HF 后端

在很多 HF 路径里，自定义配置对象会调用 `set_infer_args(**infer_config)`。

如果键未知：

- 代码会记录类似 `There is NO <key> in Custom Config!` 的错误，
- 该键通常会被忽略，
- 有一个特殊兼容分支会把 `num_recent_tokens` 映射到 `tail_token_size`。

### Sparse-vLLM 后端

引擎会构建：

- `config_fields = {field.name for field in fields(Config)}`
- `config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}`

这意味着：

- 未知键会在构建 `Config(...)` 之前被静默丢弃，
- 拼写错误不会快速失败，
- HF 专属键可以留在同一个 benchmark 配置里而不报错，但它们也不会产生任何作用。

实践规则：

> Sparse-vLLM 更“安静”，HF 更“吵”。两边都不能保证拼写错误会阻止任务继续运行。

## 11. 迁移模式

切换后端通常需要三步：

1. 删除只存在于旧后端的参数。
2. 重命名那些两边概念相同但键名不同的参数。
3. 对“同名但异义”的参数重新调参。

### 11.1 示例：HF DeltaKV 迁移到 Sparse-vLLM DeltaKV

典型 HF 风格配置：

```python
backend = "hf"
model_cls = "deltakv"
compressor_path = "/path/to/compressor"
infer_config = {
    "num_top_tokens": 0.17,
    "num_top_tokens_in_prefill": 4096,
    "num_recent_tokens": 128,
    "num_sink_tokens": 8,
    "full_attn_layers": "0,1,2,8,18",
    "use_compression": True,
    "use_cluster": True,
    "cluster_ratio": 0.1,
    "chunk_prefill_size": 2048000,
}
```

典型 Sparse-vLLM 风格配置：

```python
backend = "sparsevllm"
model_cls = "deltakv"  # ignored
compressor_path = "/path/to/compressor"  # ignored
infer_config = {
    "vllm_sparse_method": "deltakv-triton-v4",
    "deltakv_path": "/path/to/compressor",
    "num_top_tokens": 2048,
    "num_top_tokens_in_prefill": 4096,
    "num_recent_tokens": 128,
    "num_sink_tokens": 8,
    "full_attn_layers": "0,1,2,8,18",
    "cluster_ratio": 0.1,
    "chunk_prefill_size": 512,
    "max_model_len": 131000,
    "gpu_memory_utilization": 0.9,
    "max_num_batched_tokens": 8192,
}
```

这次迁移中的关键差异：

- `compressor_path` 必须改为 `deltakv_path`
- `model_cls` 不再重要
- 比例型浮点 `num_top_tokens` 应改为显式 token 预算
- `chunk_prefill_size` 必须围绕引擎调度重新调参，不能直接照搬
- 引擎容量参数现在变得重要

### 11.2 示例：HF Quest 迁移到 Sparse-vLLM Quest

HF 风格 Quest 参数：

```python
backend = "hf"
model_cls = "quest"
infer_config = {
    "num_top_tokens": 1024,
    "chunk_size": 16,
}
```

Sparse-vLLM 风格 Quest 参数：

```python
backend = "sparsevllm"
infer_config = {
    "vllm_sparse_method": "quest",
    "quest_token_budget": 1024,
    "quest_chunk_size": 16,
    "quest_skip_layers": 2,
}
```

这是同一个高层方法家族，但不是同一套配置接口。

## 12. 后端特定注意事项

### Qwen3 + DeltaKV

Sparse-vLLM 当前会在 cache-manager 层拒绝 Qwen3 的 DeltaKV 推理，并提示你改用 HF 后端。

实践规则：

> 如果你今天在本仓库里需要 Qwen3 + DeltaKV，请使用 `backend="hf"`。

### `deltasnapkv` 与 Sparse-vLLM 混合 DeltaKV 路径

HF `model_cls="deltasnapkv"` 有一个硬性要求：`full_attn_layers` 必须为空。

Sparse-vLLM 有不同的混合 DeltaKV 变体：

- `deltakv-snapkv`
- `deltakv-standalone`
- `deltakv-triton-*`

不要把 HF `deltasnapkv` 的配置直接当作这些引擎方法的可移植配置。

## 13. 常见坑位清单

- `compressor_path` 在 `backend="sparsevllm"` 下不起作用。
- `deltakv_path` 在 `backend="hf"` 下不起作用。
- `model_cls` 不会选择 Sparse-vLLM 方法。
- `num_top_tokens=0.11` 在 HF 有意义，但不是可移植的 Sparse-vLLM 设置。
- 在 `get_chat_api.py` 中，Sparse-vLLM wrapper 当前会忽略 `top_p`、`top_k` 和 `eos_token_id`。
- `past_key_values` 复用无法在后端间可移植。
- `return_kv_cache=True` 不是可移植的后端契约。
- `return_model=True` 仅 HF 支持。
- `chunk_prefill_size` 在数值上不可直接移植。
- Sparse-vLLM 会静默丢弃未知配置键。
- HF 通常会记录未知配置键但仍继续执行。

## 14. 推荐工作流

当你要在两个后端上新增或对比实验时：

1. 从后端专用的最小配置开始，不要从共享“巨型配置”开始。
2. 只添加该后端实际会消费的参数。
3. 对任意两边同名键，先确认它是真共享语义还是仅共享名字。
4. 从 HF 迁移到 Sparse-vLLM 时，除非代码明确接受比例，否则请把比例预算转换为显式计数。
5. 把 `chunk_prefill_size`、`max_model_len`、批处理上限与显存利用率参数当作一个联合调参问题处理。

如果你只记住本文一条规则，请记这一条：

> 在本仓库里，后端切换是一次运行时迁移，而不只是方法开关。
