# `add-baseline` 相对 `main` 的代码变更说明

## 1. 对比范围与总体结论

本文对当前分支 `add-baseline` 相对于 `main` 的差异进行比较，具体如下：

* 对比区间：`main...add-baseline`
* 提交数量：41
* 代码统计：89 个文件发生变更，`+12413 / -1708`

总体来看，这个分支并不是一次单点修复，而是一次覆盖“baseline 支持 + 运行时支持 + 评测流水线补齐”的较为全面的扩展：

1. `src/deltakv/` 已从主要面向 Qwen2/Llama 的 DeltaKV 实现，扩展为更宽的家族，包含更多运行时变体、Qwen3 支持、量化加载、更多 baseline 适配器，以及训练/数据生成流水线。
2. `src/sparsevllm/` 新增 `deltakv-standalone` 与 `deltakv-snapkv` 两个一等稀疏方法，并进一步夯实 DeltaKV 的配置同步、调度、重构、容量预算和内核组织。
3. `benchmark/`、`scripts/` 和 `baselines/kvzip/` 得到系统性增强，使批量运行、复现、样本过滤、baseline 切换，以及在 LongBench、SCBench、MathBench、NIAH 上覆盖新方法组合都变得更加容易。

换句话说，这个分支的核心目标不只是“让某一个方法跑通”，而是把 Sparse-vLLM 仓库从“有一个 DeltaKV demo”推进到“能够支持多 baseline 对比、大规模批量评测、训练/分析，以及 HF 与 sparse-vLLM 运行时切换”。

---

## 2. 高层变更轨迹

### 2.1 DeltaKV 模型家族明显扩容

这一组改动把 DeltaKV 从单一路径扩展成了更完整的方法家族，而不再是单一实现：

* 新增 `full_deltakv`
* 新增 `origin_residual_quant`
* 新增 `all_origin_residual_quant`
* 新增 `deltasnapkv`
* 为 `e2e / cluster_e2e / cluster_e2e_big / inference` 新增 Qwen3 支持
* 新增 OmniKV / KIVI 加载适配器
* 新增低比特 base model 加载辅助逻辑

换言之，这个分支把过去纠缠在一起的多个维度拆开了：基座模型加载策略、压缩缓存形态、cluster/reference token 选择策略、静态裁剪行为，以及残差量化。

### 2.2 sparse-vLLM 从“支持 DeltaKV”演进为“支持更多 DeltaKV 变体”

`src/sparsevllm/` 这部分改动的主轴有两条：

* 在 cache-manager-first 设计下，将 `deltakv-standalone` 与 `deltakv-snapkv` 拆分为独立 cache manager。
* 调度器、注意力层、runner、内核与配置同步进一步统一到可稳定运行这些新方法的程度。

这里最大的结构变化在于 `attention.py` 变得更通用。它现在主要负责“写入当前层 KV”和“读取 cache manager 给出的计算视图”，而数据是直接从持久存储读取，还是先重构到临时池再读取，交由 cache manager 决定。

### 2.3 评测与脚本层基本演化成实验平台

LongBench、SCBench、MathBench、NIAH 都被增强了，而且方式非常一致：

* 路径改为环境变量可配置，减少硬编码。
* 增加对 Qwen3、KVzip、多个 DeltaKV 变体的兼容。
* 补齐多 GPU 数据并行 worker、结果合并与日志。
* 增加子集/上下文长度过滤、thinking 清理、预处理 parquet 读取。
* 引入一组批量排队与 sweep 脚本。

这意味着该分支不仅新增了方法实现，也解决了“如何把这些方法作为一套一致的大规模实验体系运行起来”的工程问题。

---

## 3. 关键代码变更

### 3.1 `src/deltakv/`：配置、加载、模型家族、训练与分析的全面扩展

#### 配置层

`src/deltakv/configs/model_config_cls.py` 引入了多个关键扩展：

* 新增 `KVQwen3Config`，把 Qwen3 正式纳入 DeltaKV 训练/推理配置体系。
* 新增 `parse_full_attn_layers()`，将字符串/列表形式的层配置统一规范化为整数列表。
* 新增 `k_neighbors`，并通过 `finalize_cluster_args()` 与旧 `seq_chunk_size` 兼容，在概念上将“聚类邻居数量”与“chunk 粒度”分离。
* 新增 `stride_alpha`，用于动态 stride 聚类中心调度。
* 新增 `deltakv_use_omnikv_selection`，用于控制 DeltaKV 是否采用 OmniKV 风格 token 选择。
* 新增 `deltasnapkv_total_budget` 与 `deltasnapkv_ref_budget`，分别为 DeltaSnapKV 提供总预算和 reference-token 预算。
* 若干默认值发生明显变化：`use_cluster=True`、`chunk_prefill_accel_omnikv=False`、`num_top_tokens_in_prefill=8192`。

这些变化的含义是：配置层不再只是原始 DeltaKV 参数的直接容器，而是成为了可承载多种压缩、选择与静态裁剪路径的统一抽象层。

#### 模型加载与推理入口

`src/deltakv/get_chat_api.py` 是本分支主要汇聚点之一，新增内容包括：

* 在 `model_cls='deltakv'` 下官方支持 Qwen3
* 为 `full_deltakv`、`origin_residual_quant`、`all_origin_residual_quant`、`deltasnapkv` 增加新加载分支
* 增加 `omnikv` 与 `kivi` baseline 的适配入口
* HF 路径新增低比特 base model 加载辅助逻辑
* 在加载 compressor 权重后，对量化跳过模块进行 dtype 恢复
* 严格要求 `deltasnapkv` 必须使用空的 `full_attn_layers`，明确不支持混合 full-attention 层

这使得一个统一加载函数就能覆盖更多实验配置，而不再强迫外部脚本自行拼接自定义逻辑。

#### 量化与 baseline 适配

新增了两个关键辅助模块：

* `src/deltakv/quantization.py`

  * 解析 `load_in_4bit / load_in_8bit / torch_dtype / quant_skip_modules` 等推理设置
  * 构建 `BitsAndBytesConfig`
  * 定义量化默认跳过模块集合，如 `compress_down`、`compress_up`、`cluster`、`transform`
  * 提供 `restore_modules_to_dtype()`，将 compressor 与 clustering 模块恢复到目标精度

* `src/deltakv/baseline_adapters.py`

  * 通过复用 DeltaKV 推理家族并关闭 compression/cluster，新增 OmniKV 加载适配器
  * 通过注入 `k_bits / v_bits / group_size / residual_length`，并按模型类型切换 Llama 或 Mistral 变体，新增 KIVI 加载适配器

两者结合后，大幅减少了在同一套 benchmark 脚本中切换 baseline 所需的胶水代码。

#### 新增与扩展的 cache/model 实现

新增的 cache 实现包括：

* `src/deltakv/modeling/full_deltakv_compress_cache.py`
* `src/deltakv/modeling/origin_residual_quant_cache.py`
* `src/deltakv/modeling/all_origin_residual_quant_cache.py`

具体对应关系：

* `full_deltakv_compress_cache.py` 对应 full DeltaKV compressed-cache 路径
* `origin_residual_quant_cache.py` 对应 origin residual quant
* `all_origin_residual_quant_cache.py` 对应 all-origin residual quant

`src/deltakv/modeling/kv_cache.py` 本身也被显著增强，关键更新包括：

* 增加基于 `stride_alpha` 的动态中心构建逻辑
* 将 cluster 相关逻辑切换到新的 `get_cluster_k_neighbors()`
* 提供动态 stride 中心推进、父节点选择与 cluster 维护所需的底层能力

也就是说，新运行时变体并非各自重复造轮子，共享功能被下沉到了 `kv_cache.py` 与专用 cache 类。

#### Llama / Qwen2 / Qwen3 模型家族

Llama 侧：

* 新增 `llama_deltasnapkv.py`
* 新增 `llama_full_deltakv_compress_inference.py`
* 新增 `llama_origin_residual_quant_inference.py`
* 新增 `llama_all_origin_residual_quant_inference.py`
* 修改 `llama_with_compress_inference.py`、`llama_e2e.py`、`llama_e2e_cluster.py`、`llama_pyramidkv.py`

Llama 侧最关键的实质变化是：

* DeltaSnapKV 被实现为构建在 DeltaKV cluster cache 之上的“静态裁剪 + reference-token 保留 + 受保护后缀”路径
* `llama_with_compress_inference.py` 支持 `deltakv_use_omnikv_selection`
* `llama_e2e_cluster.py` 采用新的 `k_neighbors` 语义
* 其他推理变体与新 cache/config/量化加载逻辑对齐

Qwen2 侧：

* 新增 `qwen2_deltasnapkv.py`
* 新增 `qwen2_full_deltakv_compress_inference.py`
* 新增 `qwen2_origin_residual_quant_inference.py`
* 新增 `qwen2_all_origin_residual_quant_inference.py`
* 修改 `qwen2_with_compress_inference.py`、`qwen2_e2e.py`、`qwen2_e2e_cluster.py`、`qwen2_e2e_cluster_for_big_model.py`、`qwen2_snapkv.py`、`qwen2_pyramidkv.py`

Qwen2 侧与 Llama 侧作用一致：把新的 cache 变体、cluster 参数与 DeltaSnapKV 支持完整接入。

Qwen3 是本分支新增规模最大的模型家族之一：

* 新增 `src/deltakv/modeling/qwen3/__init__.py`
* 新增 `qwen3_e2e.py`
* 新增 `qwen3_e2e_cluster.py`
* 新增 `qwen3_e2e_cluster_for_big_model.py`
* 新增 `qwen3_with_compress_inference.py`
* 新增 `qwen3_full_deltakv_compress_inference.py`
* 新增 `qwen3_origin_residual_quant_inference.py`
* 新增 `qwen3_all_origin_residual_quant_inference.py`

这意味着 HF 侧 DeltaKV 家族已覆盖 Qwen3 的训练、cluster 训练、大模型 cluster 训练、标准推理与多个推理变体。

一个重要细节：Qwen3 仅在 HF 路径扩展；sparse-vLLM 侧仍明确禁用 `qwen3 + deltakv`。

#### 训练、保存与数据准备

训练路径关键改动集中在以下文件：

* `src/deltakv/train_compressor.py`

  * 增加 `deepspeed` 参数
  * 增加 `k_neighbors`，并在配置构建后做 finalize
  * 新增 Qwen3 训练入口
  * 将 `LOCAL_RANK` 绑定到 CUDA 设备
  * 在多 GPU 下跨 rank 广播统一时间戳，避免不同 rank 写入不同目录
  * 默认 `device_map` 从 `auto` 改为固定 local-rank 映射
  * 支持 `cluster_e2e / cluster_e2e_big` 的 Qwen3 实现

* `src/deltakv/save_trainable_trainer.py`

  * 新增 `_collect_trainable_state_dict()`
  * 使用 `accelerator.unwrap_model()` 获取真实模型
  * 更稳健地仅保存 `requires_grad=True` 参数，并处理 `module.` 前缀

* `src/deltakv/configs/ds_zero2_bf16.json`

  * 新增 DeepSpeed ZeRO-2 BF16 配置文件

* `src/deltakv/data_prepare/generate_train_data.py`

  * 调整 `vr1.0` 处理流程，支持自定义 `dataset_path` / `output_root`
  * 新增 `vi1.0` 混合训练数据生成流水线
  * `vi1.0` 将 FineWeb、推理对话与合成 UUID-KV 混合为训练集
  * 支持从本地 parquet 目录加载数据
  * 新增 reasoning 样本变换、chat template 渲染与 UUID-KV 多轮对话合成
  * 新增 `fineweb_skip_factor`、`reasoning_ratio`、`uuid_kv_ratio` 等控制参数
  * 新增对 Qwen3 tokenizer 路径的识别

这些变化让训练侧从单用途数据处理脚本，走向支持混合数据配方和 Qwen3 训练。

#### 分析与可视化

* `src/deltakv/analysis/verify_insight_pdf.py`

  * 显著扩展为多实验分析脚本
  * 增加 `stride_alpha` 序列解析
  * 增加基于 `stride_alpha` 的中心构建与 cluster 相似度统计
  * 区分 `history_only` 与 `runtime_visible` 相似度
  * 增加跨多 alpha 值汇总与 PDF 输出，如 `exp2_cluster_similarity_vs_alpha.pdf`
  * 保留距离分布、SVD 方差、范数分布等既有实验

* `src/deltakv/analysis/visualize_ablation_loss.py`

  * 将 cluster chunk 参数统一到 `k_neighbors` 语义下，减少新旧参数混用

这些改动主要服务于论文/报告导向分析，而非运行时行为。

### 3.2 `src/sparsevllm/`：新增 standalone/snapkv 变体，并强化调度与内核支持

#### 配置与方法注册

* `src/sparsevllm/config.py`

  * 在 `vllm_sparse_method` 中新增 `deltakv-standalone` 与 `deltakv-snapkv`
  * 默认 `chunk_prefill_accel_omnikv` 改为 `False`
  * 默认 `num_top_tokens_in_prefill` 改为 `8192`
  * 更稳健地解析空字符串 `full_attn_layers`
  * 所有 `deltakv*` 方法都强制要求 `deltakv_path`
  * 对新增 standalone/snapkv 方法自动清空 `full_attn_layers` 与 `obs_layer_ids`

* `src/sparsevllm/engine/cache_manager/__init__.py`

  * 注册 `DeltaKVStandaloneCacheManager` 与 `DeltaKVSnapKVCacheManager`

* `src/sparsevllm/engine/cache_manager/base.py`

  * 为两个新 cache manager 增加工厂分支
  * 在 sparse-vLLM 路径显式禁用 `qwen3 + deltakv`
  * 新增 `debug_live_seq_slots()` 调试接口

#### 新 cache manager：`deltakv-standalone`

`src/sparsevllm/engine/cache_manager/deltakv_standalone.py` 是 sparse-vLLM 侧新增文件中体量最大的之一，其核心语义是：

* 所有层都使用 DeltaKV 压缩
* 不再保留混合 full-attention 层
* 每层维护持久 DeltaKV cache
* 共享一个全局临时重构池
* 在 decode/prefill 注意力时，先把当前可见上下文重构到临时池，再用于注意力计算

它还实现了：

* 对 persistent/full/latent/temp 资源的独立容量分配策略
* 按序列的行管理与 slot 映射
* 加载 compressor 后逐层 `compress_down / compress_up`
* `prompt_admission_costs()`、`prompt_admission_budgets()`、`reserved_prefill_slots()` 等准入相关接口
* `free_slot_stats()`、`debug_live_seq_slots()` 等调试与统计工具

可以把它理解为：在 sparse-vLLM 中移除 full-attention 层，并让 DeltaKV 全面接管全层 cache 布局的版本。

#### 新 cache manager：`deltakv-snapkv`

`src/sparsevllm/engine/cache_manager/deltakv_snapkv.py` 基于 standalone 再增加一步静态裁剪：

* 在 prefill 末尾保留更长的受保护后缀
* 对压缩中间区施加 SnapKV 风格静态裁剪
* 使用 `row_deltakv_comp_abs_pos` 维护“压缩逻辑位置”到“原始绝对位置”的映射
* 将保留的 latent token 回写物化到持久 raw slot
* 为 SnapKV window 与 keep budget 预留更多持久 slot

这使 sparse-vLLM 侧具备了第一个“DeltaKV 压缩 + SnapKV 静态保留”的混合运行时。

#### 现有 DeltaKV cache manager 强化

`src/sparsevllm/engine/cache_manager/deltakv.py` 虽然不是新文件，但增强幅度很大：

* 更系统地使用显式配置字段，而不是 `getattr(..., default)`
* 准入与空闲 slot 统计接口进一步完善
* `deltakv_k_neighbors`、`num_top_tokens_in_prefill`、`full_pool_reserve_ratio` 等参数被更直接使用
* 与新 loader/config 同步逻辑对齐

这是 sparse-vLLM 侧关键稳定性提升之一，因为 cache 分配必须在真实 latent 维度与 compressor 结构确定之后进行。

#### 调度器、runner 与注意力路径

* `src/sparsevllm/engine/model_runner.py`

  * 创建 cache manager 前调用 `sync_deltakv_config_from_checkpoint(config)`
  * 基于当前 batch 长度动态计算长文本阈值
  * 针对 `deltakv-standalone` / `deltakv-snapkv` 与旧 DeltaKV 使用不同长文本定义
  * 将推理置于 `torch.inference_mode()`，减少不必要 autograd 图

* `src/sparsevllm/engine/scheduler.py`

  * 按稀疏方法分别计算长文本阈值
  * 在 prompt 准入中增加预算检查、defer 逻辑，以及更详细的死锁/饥饿诊断
  * 避免 prefill/decode/preempt 之间反复抖动
  * 更清晰地区分正常空闲状态与应报错的真实卡死状态

* `src/sparsevllm/engine/llm_engine.py`

  * 对 standalone/snapkv 分别处理 warmup 长度
  * 当调度器返回无可运行序列时，增加 busy loop 防护与更清晰报错

* `src/sparsevllm/layers/attention.py`

  * 写 KV 时使用 `get_layer_store_view()`
  * 注意力计算读取 `get_layer_compute_tensors()`
  * 这使 cache manager 能决定数据“存在哪”和“从哪读来计算”，从而支撑 standalone/snapkv 使用的临时池重构设计

* `src/sparsevllm/engine/sparse_controller.py`

  * 显式识别 `deltakv-standalone` 与 `deltakv-snapkv`
  * 为 standalone 类方法使用统一重构路径
  * 在 chunk prefill 末尾执行 `deltakv-snapkv` 的 finalize 逻辑
  * 相应调整注意力分数收集与读视图逻辑

#### 标准 cache manager 与辅助逻辑的小改动

* `src/sparsevllm/engine/cache_manager/standard.py`

  * 在 `SPARSEVLLM_DEBUG_SLOTS` 下增加释放日志
  * 实现 `debug_live_seq_slots()`

* `src/sparsevllm/engine/cache_manager/snapkv.py`

* `src/sparsevllm/engine/cache_manager/streamingllm.py`

  * 切换到显式配置字段
  * 简化 `prefill_batched_tokens_margin()` 与 `remaining_prefill_tokens()` 逻辑

#### Triton 内核重组

* `src/sparsevllm/triton_kernel/deltakv_kernels.py`

  * 新增或合并 `deltakv_reconstruct_writeback_grouped_heads_srcdst()`
  * 把原 `rekv_kernels.py` 中的 reconstruct writeback 与 blockwise L2 top-k 逻辑并入
  * 持续强化 grouped-head 内核变体

* `src/sparsevllm/triton_kernel/rekv_kernels.py`

  * 已移除

这表明内核层做了清理：把分散文件中的功能收敛到主 DeltaKV 内核文件。

#### Loader 与测试支持

* `src/sparsevllm/utils/compressor.py`

  * 优先使用 `hf_config.head_dim`，提升更多模型配置兼容性

* `src/sparsevllm/utils/loader.py`

  * 新增 `sync_deltakv_config_from_checkpoint()`
  * 支持从 checkpoint 目录或单文件解析 compressor 配置
  * 优先从 `config.json` 读取 `kv_compressed_size`、compressor 类型、中间层大小、bias 等
  * 必要时回退为从权重形状推断
  * 对 split-KV checkpoint 显式报错
  * 在 `load_deltakv_compressors_to_cache_manager()` 中复用统一文件解析逻辑

这是 sparse-vLLM 的关键稳定性升级之一，因为 cache 分配必须等真实 compressor 结构确定后再进行。

### 3.3 `benchmark/`：LongBench、SCBench、MathBench、NIAH 全部加强

#### LongBench

* `benchmark/long_bench/eval.py`

  * 通过 `DELTAKV_OUTPUT_DIR` 让输出目录可配置
  * 新增 `TASK_HIERARCHY`
  * 新增类别级聚合分数与总体类别平均分

* `benchmark/long_bench/pred.py`

  * 通过 `DELTAKV_LONGBENCH_DATA_DIR / DELTAKV_DATA_DIR` 让数据根目录可配置
  * 启动前校验数据目录与必需 JSONL 文件
  * 新增 `NO_CHAT_TEMPLATE_DATASETS`
  * 新增 `thinking_mode`，包括清理 Qwen3 `<think>` 输出
  * 新增 KVzip LongBench prompt 适配，拆分为 `prefill_text + query_text`
  * 新增 `temperature / top_p / top_k / max_new_tokens_override`
  * 对 KVzip 且 `ws > 1` 的情况，每个 GPU 启动一个单 GPU worker
  * 父进程可自动聚合日志、运行 `eval.py` 并写入评测日志

这里最有价值的提升是：LongBench 现在可以更自然地同时支持常规 HF 路径、KVzip、Qwen3 thinking 模板和多 GPU 数据并行。

#### MathBench

* `benchmark/math_bench/pred.py`

  * 新增 KVzip prompt 适配
  * 使用 `apply_chat_template(add_generation_prompt=False/True)` 拆分 prefill/query 段
  * 对 KVzip 显式禁用 `aime2024`

#### NIAH

* `benchmark/niah/test_niah.py`

  * 扩展参数以覆盖新的 DeltaKV 配置项
  * 输出根目录改为环境变量驱动
  * 新增 `num_top_tokens_in_prefill`、`stride_alpha`、`deltakv_use_omnikv_selection`、`chunk_prefill_accel_omnikv`、`omnikv_score_method`
  * 允许 `context_lengths` 接收 list/tuple 形式

#### SCBench CLI 与主流程

* `benchmark/scbench/args.py`

  * 新增 `load_in_4bit`、`load_in_8bit`、`model_torch_dtype`
  * 新增 `tensor_parallel_size` 与 `copy_on_gpu`
  * 新增 `context_min_tokens`、`context_max_tokens`、`subset_indices_file`
  * 新增 `num_data_shards` 与 `data_shard_id`
  * 将 `full_deltakv`、`origin_residual_quant`、`all_origin_residual_quant` 纳入 `attn_type` 候选

* `benchmark/scbench/run_scbench.py`

  * `BASE_PATH` 改为输出目录环境变量驱动
  * 通过 `_build_result_dir()` 统一结果目录命名
  * 新增子集索引加载与上下文长度过滤
  * 新增每 GPU 一 worker 的数据并行执行与结果合并
  * 模型加载支持 `tensor_parallel_size`、`copy_on_gpu` 与量化参数
  * 支持 `use_chat_template`
  * 结果日志使用统一的 `real_model_name_tag`
  * 单 GPU 与多 GPU 路径都输出更标准化的 `result.json` 与日志

* `benchmark/scbench/eval_utils.py`

  * 为 `GreedySearch` 新增 KV 状态快照与恢复逻辑
  * 在多轮/同上下文多查询评测中，后续 query 可从保存的 cache 状态继续，而不是全部重算
  * 处理 `prepare_inputs_for_generation()` 不返回 `past_key_values` 的情况
  * 调用 `clear_temp_kv_cache()` 前先做 `hasattr` 检查
  * EOS 处理更稳健

#### 预处理数据的新 SCBench runner

* `benchmark/scbench/run_scbench_preprocessed.py`

  * 直接在 `SCBench-preprocessed` parquet 数据上运行 DeltaKV/HF 路径
  * 内置 KVzip 风格 prompt 模板构造
  * 支持数据并行 worker 与结果合并
  * 可直接从预测文件计算分数

* `benchmark/scbench/run_kvzip_preprocessed.py`

  * 直接在 `SCBench-preprocessed` parquet 数据上运行 KVzip
  * 支持 `ratio / level / kv_type / prefill_chunk_size`
  * 同样支持数据并行、结果合并与分数报告

* `benchmark/scbench/scripts/run_scbench_three_eval.sh`

  * 新增批处理脚本，可一次跑三个 SCBench 任务，并同时覆盖 KVzip 与 DeltaKV

这几部分共同构成了“预处理数据评测流水线”的首个完整版本。

### 3.4 `baselines/kvzip/`：本地数据支持、Qwen 兼容、调试与结果解析增强

尽管 `baselines/kvzip/` 没有新增太多文件，但几乎每项改动都很实用：

* `attention/kvcache.py`

  * 增加 debug 与 VRAM 监控输出
  * 在 cache update OOM 时打印更丰富上下文信息
  * 在驱逐日志中扩展 CUDA alloc/reserved/max alloc 信息

* `csrc/build.py`

  * 动态收集 CUDA 架构
  * 除 `sm_80/sm_90` 外，也为当前机器 GPU 能力编译

* `data/load.py`

  * 新增 `SCBENCH_PREPROCESSED_ROOT`
  * 优先从本地 parquet 加载 `SCBench-preprocessed`，否则回退到 HuggingFace datasets

* `model/monkeypatch.py`

  * 扩展 Qwen 匹配，支持 `qwen2` 与 `distill-qwen`

* `model/wrapper.py`

  * KV score 长度不匹配时打印详细调试信息
  * 输出裁剪不再盲目删除最后一个 token，而是基于 EOS/PAD/EOT 集合裁剪
  * 为 `test_scdq()` 增加 `@torch.inference_mode()`

* `results/parse.py`

  * 与 `data/load.py` 一样，优先使用本地 `SCBench-preprocessed` 数据

这些改动本质上都在提高 KVzip 融入统一 benchmark 框架的便利性与可调试性。

### 3.5 `scripts/`：批量实验与补跑脚本形成一致工具集

* `scripts/bench_sparse_vllm.py`

  * 支持通过 JSON 传递 `hyper_params`
  * 使用一组更稳定的 benchmark 默认值
  * 更细粒度拆分 prefill/decode/TTFT/ITL/AvgBS 统计
  * 新增 staged-admission / wave-decode bench 逻辑
  * 失败时保存更清晰状态

* `scripts/queue_scbench_llama_jobs.py`

  * 新增统一排队脚本，覆盖 Qwen LongBench alpha sweep、Qwen SCBench alpha sweep、Llama DeltaKV 与 KVzip SCBench 等
  * 将模型路径、compressor 路径、任务组合、`stride_alpha`、`token_budget`、`ratio` 等参数编码为任务队列

* `scripts/run_llama31_alpha_sweep_longbench_scbench_b0p17.sh`

  * 新增批量脚本，在跑 LongBench 与 SCBench 的同时对 `stride_alpha` 做 sweep

* `scripts/run_llama_deltasnapkv_missing.sh`

* `scripts/run_llama_deltasnapkv_missing_ws2.sh`

* `scripts/run_llama_deltasnapkv_remaining.sh`

  * 用于补跑 Llama DeltaSnapKV 缺失 LongBench 任务，覆盖不同 `ws` 设置

* `scripts/simulate_linear_stride_ref_tokens.py`

  * 纯模拟脚本，用于估算不同 `stride_alpha` 下的 reference token 数与总保留量

这些脚本确保分支引入的新方法和参数不只是“代码里有”，而是已经打包成可直接执行的大规模实验入口。

### 3.6 README、忽略规则、测试与文档

* `README.md`

  * 将 `flash-attn` 安装从固定版本改为不锁定版本
  * 新增 OmniKV 与 KIVI baseline 使用示例

* `.gitignore`

  * 新增 `wandb/`、`tmp/`、`outputs/`、`benchmark/scbench/results/`

* 新增测试：

  * `tests/test_deltakv_checkpoint_config_sync.py`

    * 测试 `sync_deltakv_config_from_checkpoint()` 能否从 `config.json` 或权重形状推断 compressor 配置
  * `tests/test_quantization_helpers.py`

    * 测试低比特加载参数构建、`chunk_prefill_size` 保留，以及量化跳过模块的 dtype 恢复

* 文档与目录整理：

  * `blog/1.md` 重命名为 `docs/1.md`
  * 新增 `docs/todo.md`

* 打包元数据：

  * `src/deltakv.egg-info/PKG-INFO`
  * `src/deltakv.egg-info/SOURCES.txt`
  * 主要用于把包元数据与新增文件/包内容同步

---

## 4. 按目录分组的逐文件清单

下节按目录组织列出当前分支相对 `main` 的每个变更文件，便于文件级回溯。

### 4.1 仓库根目录

| 文件         | 变更说明 |
| ------------ | -------- |
| `.gitignore` | 为常见训练/评测输出目录添加忽略规则。 |
| `README.md`  | 更新依赖安装说明，并增加 OmniKV、KIVI 等 baseline 使用示例。 |

### 4.2 `baselines/kvzip/`

| 文件 | 变更说明 |
| --- | --- |
| `baselines/kvzip/attention/kvcache.py` | 增加 KVzip cache 更新/debug/OOM 输出与更详细的驱逐 VRAM 日志。 |
| `baselines/kvzip/csrc/build.py` | 动态加入当前机器 GPU 架构，提升 CUDA 扩展构建兼容性。 |
| `baselines/kvzip/data/load.py` | 新增本地 `SCBench-preprocessed` parquet 数据加载。 |
| `baselines/kvzip/model/monkeypatch.py` | 扩展 Qwen 模型匹配，支持 `qwen2` 与 `distill-qwen`。 |
| `baselines/kvzip/model/wrapper.py` | 强化 score 长度检查，修复输出裁剪，并为 SCDQ 路径增加 inference mode。 |
| `baselines/kvzip/results/parse.py` | 结果解析时优先使用本地 `SCBench-preprocessed` 数据。 |

### 4.3 `benchmark/long_bench/`

| 文件 | 变更说明 |
| --- | --- |
| `benchmark/long_bench/eval.py` | 新增环境变量输出目录、类别聚合分数和总体类别平均分。 |
| `benchmark/long_bench/pred.py` | 新增数据路径校验、Qwen3 thinking 处理、KVzip prompt 适配、采样参数、单 GPU worker 启动与自动日志/评测。 |

### 4.4 `benchmark/math_bench/` 与 `benchmark/niah/`

| 文件 | 变更说明 |
| --- | --- |
| `benchmark/math_bench/pred.py` | 新增 KVzip prompt 拆分，并禁用 `aime2024` 任务。 |
| `benchmark/niah/test_niah.py` | 扩展新 DeltaKV 配置参数，并切换为环境变量驱动的输出目录。 |

### 4.5 `benchmark/scbench/`

| 文件 | 变更说明 |
| --- | --- |
| `benchmark/scbench/args.py` | 扩展 CLI 参数：量化、dtype、上下文过滤、子集文件、分片、TP、copy_on_gpu 等。 |
| `benchmark/scbench/eval_utils.py` | 增加多轮评测 KV 状态快照/恢复，并改进 EOS/临时缓存兼容性。 |
| `benchmark/scbench/run_kvzip_preprocessed.py` | 新增直接在 `SCBench-preprocessed` 上跑 KVzip 的数据并行评测脚本。 |
| `benchmark/scbench/run_scbench.py` | 重构 SCBench 主流程，支持数据并行 worker、子集过滤、上下文长度过滤、量化加载与统一结果目录。 |
| `benchmark/scbench/run_scbench_preprocessed.py` | 新增直接在 `SCBench-preprocessed` 上跑 DeltaKV/HF 的评测脚本。 |
| `benchmark/scbench/scripts/run_scbench_three_eval.sh` | 新增批处理脚本，可同时跑三个 SCBench 任务并覆盖 KVzip 与 DeltaKV。 |

### 4.6 `docs/`

| 文件 | 变更说明 |
| --- | --- |
| `docs/1.md` | 由 `blog/1.md` 重命名迁入 `docs/`。 |
| `docs/todo.md` | 新增与 cache manager/rope 工作相关的后续 TODO。 |

### 4.7 `scripts/`

| 文件 | 变更说明 |
| --- | --- |
| `scripts/bench_sparse_vllm.py` | 升级 benchmark 逻辑，支持 JSON 超参数与更细粒度阶段统计。 |
| `scripts/queue_scbench_llama_jobs.py` | 新增排队脚本，用于组织 alpha sweep、SCBench、KVzip 等相关实验。 |
| `scripts/run_llama31_alpha_sweep_longbench_scbench_b0p17.sh` | 新增 Llama `stride_alpha` 联合 sweep 脚本，覆盖 LongBench 与 SCBench。 |
| `scripts/run_llama_deltasnapkv_missing.sh` | 新增 DeltaSnapKV LongBench 缺失任务补跑脚本。 |
| `scripts/run_llama_deltasnapkv_missing_ws2.sh` | 新增 DeltaSnapKV 补跑脚本的双 GPU 版本。 |
| `scripts/run_llama_deltasnapkv_remaining.sh` | 新增 DeltaSnapKV 剩余任务补跑脚本。 |
| `scripts/simulate_linear_stride_ref_tokens.py` | 新增用于估算 `stride_alpha` 下 reference-token 数的模拟脚本。 |

### 4.8 `src/deltakv/analysis/`

| 文件 | 变更说明 |
| --- | --- |
| `src/deltakv/analysis/verify_insight_pdf.py` | 扩展 `stride_alpha`、cluster 相似度汇总与更多 PDF 实验图输出。 |
| `src/deltakv/analysis/visualize_ablation_loss.py` | 将 cluster chunk 参数迁移到 `k_neighbors` 语义。 |

### 4.9 `src/deltakv/configs/`

| 文件 | 变更说明 |
| --- | --- |
| `src/deltakv/configs/ds_zero2_bf16.json` | 新增 DeepSpeed ZeRO-2 BF16 配置。 |
| `src/deltakv/configs/model_config_cls.py` | 新增 Qwen3 配置、`k_neighbors`、`stride_alpha`、DeltaSnapKV 预算及更统一的配置解析逻辑。 |

### 4.10 顶层 `src/deltakv/` 及训练/加载辅助

| 文件 | 变更说明 |
| --- | --- |
| `src/deltakv/baseline_adapters.py` | 新增 OmniKV 与 KIVI 加载适配器。 |
| `src/deltakv/data_prepare/generate_train_data.py` | 扩展 `vr1.0`，新增 `vi1.0` 混合训练数据生成，并支持本地 parquet 加载。 |
| `src/deltakv/get_chat_api.py` | 扩展统一 HF 推理入口，覆盖 Qwen3、量化、OmniKV/KIVI、DeltaSnapKV 与多 DeltaKV 变体。 |
| `src/deltakv/quantization.py` | 新增低比特模型加载与跳过模块 dtype 恢复辅助。 |
| `src/deltakv/save_trainable_trainer.py` | 仅保存可训练参数，并支持 accelerator unwrap。 |
| `src/deltakv/train_compressor.py` | 训练扩展到 Qwen3、DeepSpeed、多 GPU 时间戳同步与 `k_neighbors`。 |

### 4.11 通用 `src/deltakv/modeling/` 层

| 文件 | 变更说明 |
| --- | --- |
| `src/deltakv/modeling/all_origin_residual_quant_cache.py` | 新增 all-origin residual quant cache 实现。 |
| `src/deltakv/modeling/full_deltakv_compress_cache.py` | 新增 full DeltaKV compressed-cache 实现。 |
| `src/deltakv/modeling/kv_cache.py` | 扩展动态 stride 中心构建、cluster 选择与新参数语义。 |
| `src/deltakv/modeling/origin_residual_quant_cache.py` | 新增 origin residual quant cache 实现。 |

### 4.12 `src/deltakv/modeling/llama/`

| 文件 | 变更说明 |
| --- | --- |
| `src/deltakv/modeling/llama/llama_all_origin_residual_quant_inference.py` | 新增 Llama all-origin residual quant 推理。 |
| `src/deltakv/modeling/llama/llama_deltasnapkv.py` | 新增 Llama DeltaSnapKV 推理。 |
| `src/deltakv/modeling/llama/llama_e2e.py` | 同步新配置与共享能力到 Llama 端到端训练。 |
| `src/deltakv/modeling/llama/llama_e2e_cluster.py` | 将 cluster 训练逻辑迁移到新 `k_neighbors` 语义。 |
| `src/deltakv/modeling/llama/llama_full_deltakv_compress_inference.py` | 新增 Llama full DeltaKV 推理。 |
| `src/deltakv/modeling/llama/llama_origin_residual_quant_inference.py` | 新增 Llama origin residual quant 推理。 |
| `src/deltakv/modeling/llama/llama_pyramidkv.py` | 将新配置/加载接口同步到 Llama PyramidKV。 |
| `src/deltakv/modeling/llama/llama_with_compress_inference.py` | 新增 `deltakv_use_omnikv_selection` 等 DeltaKV 选择逻辑。 |

### 4.13 `src/deltakv/modeling/qwen2/`

| 文件 | 变更说明 |
| --- | --- |
| `src/deltakv/modeling/qwen2/qwen2_all_origin_residual_quant_inference.py` | 新增 Qwen2 all-origin residual quant 推理。 |
| `src/deltakv/modeling/qwen2/qwen2_deltasnapkv.py` | 新增 Qwen2 DeltaSnapKV 推理。 |
| `src/deltakv/modeling/qwen2/qwen2_e2e.py` | 同步新配置与共享能力到 Qwen2 端到端训练。 |
| `src/deltakv/modeling/qwen2/qwen2_e2e_cluster.py` | 将 cluster 训练逻辑迁移到新 `k_neighbors` 语义。 |
| `src/deltakv/modeling/qwen2/qwen2_e2e_cluster_for_big_model.py` | 将新 cluster 参数语义同步到大模型训练路径。 |
| `src/deltakv/modeling/qwen2/qwen2_full_deltakv_compress_inference.py` | 新增 Qwen2 full DeltaKV 推理。 |
| `src/deltakv/modeling/qwen2/qwen2_origin_residual_quant_inference.py` | 新增 Qwen2 origin residual quant 推理。 |
| `src/deltakv/modeling/qwen2/qwen2_pyramidkv.py` | 将新配置/加载接口同步到 Qwen2 PyramidKV。 |
| `src/deltakv/modeling/qwen2/qwen2_snapkv.py` | 将新配置/加载接口同步到 Qwen2 SnapKV。 |
| `src/deltakv/modeling/qwen2/qwen2_with_compress_inference.py` | 新增 DeltaKV 选择与配置逻辑。 |

### 4.14 `src/deltakv/modeling/qwen3/`

| 文件 | 变更说明 |
| --- | --- |
| `src/deltakv/modeling/qwen3/__init__.py` | 新增 Qwen3 modeling 包标记文件。 |
| `src/deltakv/modeling/qwen3/qwen3_all_origin_residual_quant_inference.py` | 新增 Qwen3 all-origin residual quant 推理。 |
| `src/deltakv/modeling/qwen3/qwen3_e2e.py` | 新增 Qwen3 端到端 DeltaKV 训练。 |
| `src/deltakv/modeling/qwen3/qwen3_e2e_cluster.py` | 新增 Qwen3 cluster 训练。 |
| `src/deltakv/modeling/qwen3/qwen3_e2e_cluster_for_big_model.py` | 新增 Qwen3 大模型 cluster 训练。 |
| `src/deltakv/modeling/qwen3/qwen3_full_deltakv_compress_inference.py` | 新增 Qwen3 full DeltaKV 推理。 |
| `src/deltakv/modeling/qwen3/qwen3_origin_residual_quant_inference.py` | 新增 Qwen3 origin residual quant 推理。 |
| `src/deltakv/modeling/qwen3/qwen3_with_compress_inference.py` | 新增标准 Qwen3 DeltaKV 推理。 |

### 4.15 `src/sparsevllm/`

| 文件 | 变更说明 |
| --- | --- |
| `src/sparsevllm/config.py` | 增加 `deltakv-standalone` / `deltakv-snapkv` 配置入口并调整多个默认值。 |
| `src/sparsevllm/engine/cache_manager/__init__.py` | 注册新的 standalone/snapkv cache manager。 |
| `src/sparsevllm/engine/cache_manager/base.py` | 扩展 cache manager 工厂逻辑，并禁用 qwen3+sparsevllm deltakv。 |
| `src/sparsevllm/engine/cache_manager/deltakv.py` | 进一步强化 DeltaKV cache manager 的容量预算与参数访问逻辑。 |
| `src/sparsevllm/engine/cache_manager/deltakv_snapkv.py` | 新增 sparse-vLLM DeltaKV+SnapKV 混合 cache manager。 |
| `src/sparsevllm/engine/cache_manager/deltakv_standalone.py` | 新增 sparse-vLLM 全层 DeltaKV standalone cache manager。 |
| `src/sparsevllm/engine/cache_manager/snapkv.py` | 调整 SnapKV prefill margin 与 remaining-token 计算。 |
| `src/sparsevllm/engine/cache_manager/standard.py` | 新增 slot 调试日志与 live-slot 调试接口。 |
| `src/sparsevllm/engine/cache_manager/streamingllm.py` | 调整 StreamingLLM prefill margin 与 remaining-token 计算。 |
| `src/sparsevllm/engine/llm_engine.py` | 为新稀疏方法改进 warmup 与无可运行序列保护。 |
| `src/sparsevllm/engine/model_runner.py` | 在 cache 分配前同步 DeltaKV checkpoint 配置，并适配新的长文本语义。 |
| `src/sparsevllm/engine/scheduler.py` | 强化 admission/defer/preempt 逻辑并适配 standalone/snapkv 长文本阈值。 |
| `src/sparsevllm/engine/sparse_controller.py` | 识别新 DeltaKV 变体并管理对应 reconstruct/finalize 流程。 |
| `src/sparsevllm/layers/attention.py` | 抽象“store view”与“compute view”以支持临时池重构。 |
| `src/sparsevllm/triton_kernel/deltakv_kernels.py` | 合并更多 reconstruct/top-k 内核并新增 src-dst writeback 变体。 |
| `src/sparsevllm/triton_kernel/rekv_kernels.py` | 移除；相关内核已并入 `deltakv_kernels.py`。 |
| `src/sparsevllm/utils/compressor.py` | 优先使用 HF 配置中的显式 `head_dim`。 |
| `src/sparsevllm/utils/loader.py` | 新增 DeltaKV checkpoint 配置自动同步与统一 compressor 权重解析。 |

### 4.16 测试与打包元数据

| 文件 | 变更说明 |
| --- | --- |
| `src/deltakv.egg-info/PKG-INFO` | 打包元数据已更新。 |
| `src/deltakv.egg-info/SOURCES.txt` | 打包源码文件列表已更新。 |
| `tests/test_deltakv_checkpoint_config_sync.py` | 新增 DeltaKV checkpoint 配置同步测试。 |
| `tests/test_quantization_helpers.py` | 新增量化辅助函数测试。 |

---

## 5. 这个分支最终交付了什么

若用一句话概括，这个分支把仓库从“有一个 Sparse-vLLM/DeltaKV 原型”推进到“多个 baseline、多个 DeltaKV 变体、两条运行时路径、多个 benchmark、以及持续训练/分析/补跑流程都能稳定对比和运行”的工程状态。

更具体地说，主要新增包括：

* 方法覆盖更完整：DeltaKV、full DeltaKV、origin residual quant、all-origin residual quant、DeltaSnapKV、OmniKV、KIVI、KVzip 等都进入了统一实验框架。
* 模型覆盖更完整：Qwen3 正式接入 HF 训练/推理体系。
* 运行时覆盖更完整：sparse-vLLM 新增 standalone 与 snapkv 混合变体。
* 实验覆盖更完整：LongBench、SCBench、MathBench、NIAH 都补齐了大规模实验所需的工程能力。

因此，`add-baseline` 分支的本质不只是“加了几个 baseline 文件”，而是围绕“baseline 对比与大规模实验”做的一次系统性扩展。
