# `add-baseline` 相对 `main` 的代码变更说明

## 1. 对比范围与总体结论

本文件比较的是当前分支 `add-baseline` 相对 `main` 的差异，即：

- 比较区间：`main...add-baseline`
- 提交数：41
- 代码统计：89 个文件变更，`+12413 / -1708`

从整体上看，这个分支不是单点修复，而是一次较完整的“基线扩容 + 运行时扩容 + 评测链路补齐”：

1. `src/deltakv/` 从原本偏向 Qwen2/Llama 的 DeltaKV 实现，扩展到了更多运行时变体、Qwen3 支持、量化加载、更多 baseline 适配和训练/数据生成链路。
2. `src/sparsevllm/` 新增了 `deltakv-standalone` 和 `deltakv-snapkv` 两条一等公民 sparse method，并把 DeltaKV 的配置同步、调度、重建、容量预算和 kernel 组织继续做实。
3. `benchmark/`、`scripts/`、`baselines/kvzip/` 被系统性补强，使 LongBench / SCBench / MathBench / NIAH 的实验更容易批量运行、复现、筛样本、切换 baseline，并且能覆盖新的方法组合。

换句话说，这个分支的核心目标不是“把一个方法跑通”，而是把 Sparse-vLLM 仓库从“已有 DeltaKV demo”推进到“可比较多种 baseline、可批量评测、可训练/可分析、可在 HF 与 sparse-vLLM 两条运行时上切换”的状态。

---

## 2. 高层变化脉络

### 2.1 DeltaKV 模型族明显扩容

这一批改动把 DeltaKV 相关实现扩成了一个更完整的方法家族，而不再只有单一路径：

- 新增 `full_deltakv`
- 新增 `origin_residual_quant`
- 新增 `all_origin_residual_quant`
- 新增 `deltasnapkv`
- 新增 Qwen3 的 `e2e / cluster_e2e / cluster_e2e_big / inference` 路径
- 新增 OmniKV / KIVI 的加载适配
- 新增低比特基础模型加载辅助逻辑

也就是说，当前分支已经把“基础模型加载方式”“压缩缓存形式”“cluster/ref token 选择策略”“是否静态 prune”“是否量化残差”等几个轴都拉开了。

### 2.2 sparse-vLLM 运行时从“支持 DeltaKV”变成“支持更多 DeltaKV 变体”

`src/sparsevllm/` 的变化重点是两件事：

- 在 cache-manager-first 的设计下，把 `deltakv-standalone` 和 `deltakv-snapkv` 独立成新的 cache manager；
- 把调度器、attention 层、runner、kernel、配置同步逻辑统一到这些新方法能稳定运行的程度。

这里最大的结构性变化是：`attention.py` 更加泛化，只负责“写入当前层 KV + 读取 cache manager 提供的 compute view”，而具体是直接读持久池、还是先 reconstruct 到 temp pool，更多由 cache manager 决定。

### 2.3 评测和脚本层面基本形成实验平台

LongBench、SCBench、MathBench、NIAH 这四块都被增强了，且增强方向很一致：

- 路径改成可配置环境变量，减少硬编码；
- 新增对 Qwen3 / KVzip / 多种 DeltaKV 变体的兼容；
- 补齐多卡数据并行 worker、结果 merge、日志记录；
- 增加 subset/context-length 过滤、thinking strip、preprocessed parquet 数据读取；
- 加入一批批量排队和扫参脚本。

这意味着当前分支不仅新增了方法实现，也同步解决了“如何把这些方法跑成一套大规模实验”的工程问题。

---

## 3. 重点代码变更

### 3.1 `src/deltakv/`：配置、加载、模型族、训练与分析全面扩展

#### 配置层

`src/deltakv/configs/model_config_cls.py` 做了几类关键扩展：

- 新增 `KVQwen3Config`，把 Qwen3 正式纳入 DeltaKV 训练/推理配置体系。
- 新增 `parse_full_attn_layers()`，统一把字符串/列表形式的层配置转成整型列表。
- 新增 `k_neighbors`，并通过 `finalize_cluster_args()` 将它与旧的 `seq_chunk_size` 兼容，含义上把“聚类邻居数”和“chunk 粒度”拆开。
- 新增 `stride_alpha`，用于动态 stride 的 cluster center 调度。
- 新增 `deltakv_use_omnikv_selection`，控制 DeltaKV 是否使用 OmniKV 风格的 token selection。
- 新增 `deltasnapkv_total_budget` 与 `deltasnapkv_ref_budget`，为 DeltaSnapKV 提供总预算和 ref token 预算。
- 默认值发生明显变化：`use_cluster=True`、`chunk_prefill_accel_omnikv=False`、`num_top_tokens_in_prefill=8192`。

这些改动的含义是：配置层已经不再只是“把原始 DeltaKV 参数搬进来”，而是在为多种压缩/选择/静态裁剪路径做统一参数抽象。

#### 模型加载与推理入口

`src/deltakv/get_chat_api.py` 是本分支最核心的汇聚点之一，扩展内容包括：

- `model_cls='deltakv'` 正式支持 Qwen3；
- 新增 `full_deltakv`、`origin_residual_quant`、`all_origin_residual_quant`、`deltasnapkv` 的加载分支；
- 新增 `omnikv`、`kivi` baseline 适配入口；
- 在 HF 路径下接入低比特基础模型加载辅助逻辑；
- 加载压缩器权重后，对被跳过量化的模块执行 dtype 恢复；
- 对 `deltasnapkv` 强制要求 `full_attn_layers` 为空，明确它不支持 mixed full-attention layers。

这部分让统一入口函数可以覆盖更多实验配置，而不是靠外部脚本各自拼接加载逻辑。

#### 量化与 baseline 适配

新增两个关键辅助模块：

- `src/deltakv/quantization.py`
  - 负责解析 `load_in_4bit / load_in_8bit / torch_dtype / quant_skip_modules` 等推理配置；
  - 生成 `BitsAndBytesConfig`；
  - 定义默认跳过量化的模块集合，例如 `compress_down`、`compress_up`、`cluster`、`transform`；
  - 提供 `restore_modules_to_dtype()`，把压缩器和聚类模块恢复到目标精度。

- `src/deltakv/baseline_adapters.py`
  - 新增 OmniKV 加载适配：复用现有的 DeltaKV inference family，但关闭 compression 和 cluster；
  - 新增 KIVI 加载适配：对 `k_bits / v_bits / group_size / residual_length` 做注入，并按模型类型切到 Llama/Mistral 变体。

这两块配合起来，显著降低了“同一套 benchmark 脚本切换 baseline”的胶水代码量。

#### 新增/扩展的缓存与模型实现

新增缓存实现：

- `src/deltakv/modeling/full_deltakv_compress_cache.py`
- `src/deltakv/modeling/origin_residual_quant_cache.py`
- `src/deltakv/modeling/all_origin_residual_quant_cache.py`

其中：

- `full_deltakv_compress_cache.py` 对应完整 DeltaKV 压缩缓存路径；
- `origin_residual_quant_cache.py` 对应 origin residual quant；
- `all_origin_residual_quant_cache.py` 对应 all-origin residual quant。

`src/deltakv/modeling/kv_cache.py` 本身也被明显增强，关键点是：

- 增加了基于 `stride_alpha` 的动态 center 构造逻辑；
- 把 cluster 相关逻辑改为使用新的 `get_cluster_k_neighbors()`；
- 为动态 stride 下的 center 位置推进、父节点选择、cluster 维护提供基础能力。

也就是说，新的运行时变体并不是各自重复造轮子，而是把公共能力往 `kv_cache.py` 和专门 cache 类中下沉。

#### Llama / Qwen2 / Qwen3 模型族

Llama 侧：

- 新增 `llama_deltasnapkv.py`
- 新增 `llama_full_deltakv_compress_inference.py`
- 新增 `llama_origin_residual_quant_inference.py`
- 新增 `llama_all_origin_residual_quant_inference.py`
- 修改 `llama_with_compress_inference.py`、`llama_e2e.py`、`llama_e2e_cluster.py`、`llama_pyramidkv.py`

Llama 侧的重要实质变化：

- DeltaSnapKV 被实现为 DeltaKV cluster cache 上的“静态 prune + ref token 保存 + 受保护尾部”路径；
- `llama_with_compress_inference.py` 接入 `deltakv_use_omnikv_selection`；
- `llama_e2e_cluster.py` 改成使用新的 `k_neighbors` 语义；
- 其他 inference 变体与新的 cache / config / 量化加载逻辑对齐。

Qwen2 侧：

- 新增 `qwen2_deltasnapkv.py`
- 新增 `qwen2_full_deltakv_compress_inference.py`
- 新增 `qwen2_origin_residual_quant_inference.py`
- 新增 `qwen2_all_origin_residual_quant_inference.py`
- 修改 `qwen2_with_compress_inference.py`、`qwen2_e2e.py`、`qwen2_e2e_cluster.py`、`qwen2_e2e_cluster_for_big_model.py`、`qwen2_snapkv.py`、`qwen2_pyramidkv.py`

Qwen2 侧的角色与 Llama 对应，重点也是把新的 cache 变体、cluster 参数与 DeltaSnapKV 接进来。

Qwen3 侧是本分支新增量最大的模型族之一：

- 新增 `src/deltakv/modeling/qwen3/__init__.py`
- 新增 `qwen3_e2e.py`
- 新增 `qwen3_e2e_cluster.py`
- 新增 `qwen3_e2e_cluster_for_big_model.py`
- 新增 `qwen3_with_compress_inference.py`
- 新增 `qwen3_full_deltakv_compress_inference.py`
- 新增 `qwen3_origin_residual_quant_inference.py`
- 新增 `qwen3_all_origin_residual_quant_inference.py`

这意味着 HF 侧 DeltaKV 族已经把 Qwen3 纳入了训练、聚类训练、大模型聚类训练、普通推理和各类变体推理路径。

需要注意的一点是：Qwen3 只在 HF 路径上扩展；sparse-vLLM 侧仍然显式禁用了 `qwen3 + deltakv`。

#### 训练、保存与数据准备

训练链路的关键变化主要落在下面几处：

- `src/deltakv/train_compressor.py`
  - 新增 `deepspeed` 参数；
  - 新增 `k_neighbors` 参数，并在 config 构造后做 finalize；
  - 新增 Qwen3 训练入口；
  - 绑定 `LOCAL_RANK` 到 CUDA 设备；
  - 多卡场景下广播统一时间戳，避免不同 rank 写入不同目录；
  - 默认 `device_map` 改为按 local rank 固定映射，而不是 `auto`；
  - 支持 `cluster_e2e / cluster_e2e_big` 对应的 Qwen3 实现。

- `src/deltakv/save_trainable_trainer.py`
  - 新增 `_collect_trainable_state_dict()`；
  - 通过 `accelerator.unwrap_model()` 获取真实模型；
  - 更稳地只保存 `requires_grad=True` 的参数，并兼容 `module.` 前缀。

- `src/deltakv/configs/ds_zero2_bf16.json`
  - 新增 DeepSpeed ZeRO-2 BF16 配置文件。

- `src/deltakv/data_prepare/generate_train_data.py`
  - `vr1.0` 数据处理改成支持自定义 `dataset_path` / `output_root`；
  - 新增 `vi1.0` 混合训练数据生成流程；
  - `vi1.0` 将 FineWeb、reasoning chat、synthetic UUID-KV 混合成训练集；
  - 支持从本地 parquet 数据目录加载数据；
  - 新增 reasoning 样本 transform、chat template 渲染、UUID-KV 多轮对话合成；
  - 新增 `fineweb_skip_factor`、`reasoning_ratio`、`uuid_kv_ratio` 等控制项；
  - 新增对 Qwen3 tokenizer 路径的识别。

这些变化把训练侧从“单一数据加工脚本”推进到了“支持混合数据方案和 Qwen3 训练”的阶段。

#### 分析与可视化

- `src/deltakv/analysis/verify_insight_pdf.py`
  - 明显扩展为多实验一体化分析脚本；
  - 新增 `stride_alpha` 序列解析；
  - 新增基于 `stride_alpha` 的 center 构造与 cluster similarity 统计；
  - 区分 `history_only` 与 `runtime_visible` 两种 similarity；
  - 新增对多个 alpha 的 summary 与 PDF 输出，如 `exp2_cluster_similarity_vs_alpha.pdf`；
  - 继续保留原有的距离分布、SVD 方差、norm 分布等实验图。

- `src/deltakv/analysis/visualize_ablation_loss.py`
  - 将 cluster chunk 参数统一映射到 `k_neighbors` 语义，减少新旧参数混用。

这块改动服务于论文/报告导向的分析工作，而不是线上运行时。

### 3.2 `src/sparsevllm/`：新增 standalone/snapkv 变体并补强调度与 kernel

#### 配置和方法注册

- `src/sparsevllm/config.py`
  - `vllm_sparse_method` 新增 `deltakv-standalone`、`deltakv-snapkv`；
  - 默认 `chunk_prefill_accel_omnikv` 改成 `False`；
  - 默认 `num_top_tokens_in_prefill` 改成 `8192`；
  - 更稳地解析空字符串形式的 `full_attn_layers`；
  - 对所有 `deltakv*` 方法统一要求 `deltakv_path`；
  - 对 standalone/snapkv 两个新方法，自动清空 `full_attn_layers` 和 `obs_layer_ids`。

- `src/sparsevllm/engine/cache_manager/__init__.py`
  - 新注册 `DeltaKVStandaloneCacheManager` 与 `DeltaKVSnapKVCacheManager`。

- `src/sparsevllm/engine/cache_manager/base.py`
  - `CacheManager.create()` 新增上述两类构造分支；
  - 显式禁用 `qwen3 + deltakv` 的 sparse-vLLM 路径；
  - 新增 `debug_live_seq_slots()` 调试接口。

#### 新增 cache manager：`deltakv-standalone`

`src/sparsevllm/engine/cache_manager/deltakv_standalone.py` 是本分支 sparse-vLLM 侧最大的新增文件之一。它的核心语义是：

- 所有层都走 DeltaKV 压缩；
- 不再保留 mixed full-attention layers；
- 每层维护持久 DeltaKV cache；
- 全局共享一个临时 reconstruct pool；
- decode / prefill attention 时，把当前可见上下文 reconstruct 到 temp pool 再计算。

它还实现了：

- 独立的容量分配策略：persistent/full/latent/temp 四类资源预算；
- per-sequence row 管理与 slot map；
- compressor 加载后的每层 `compress_down / compress_up`；
- `prompt_admission_costs()`、`prompt_admission_budgets()`、`reserved_prefill_slots()` 等 admission 接口；
- `free_slot_stats()`、`debug_live_seq_slots()` 等调试统计。

可以把它理解为“去掉 full-attention 层、让 DeltaKV 完整接管全层缓存布局”的 sparse-vLLM 版本。

#### 新增 cache manager：`deltakv-snapkv`

`src/sparsevllm/engine/cache_manager/deltakv_snapkv.py` 建立在 standalone 之上，但多做了一步静态剪枝：

- 在 prefill 末尾保留更长的 protected suffix；
- 对压缩中间段做 SnapKV 风格的 static prune；
- 通过 `row_deltakv_comp_abs_pos` 维护“压缩逻辑位置 -> 原始绝对位置”的映射；
- 把保留下来的 latent token materialize 回持久 raw slots；
- 为 SnapKV 窗口和 keep budget 预留更多 persistent slots。

这让 sparse-vLLM 端首次具备了“DeltaKV 压缩 + SnapKV 静态保留”的混合运行时。

#### 现有 DeltaKV cache manager 的补强

`src/sparsevllm/engine/cache_manager/deltakv.py` 虽然不是新文件，但逻辑上也被明显强化：

- 更系统地使用 config 上的显式字段，而不是 `getattr(..., default)`；
- admission 与 free-slot 统计接口继续完善；
- 代码里对 `deltakv_k_neighbors`、`num_top_tokens_in_prefill`、`full_pool_reserve_ratio` 等参数的使用更直接；
- 与新的 loader/config 同步逻辑对齐。

#### 调度、runner 和 attention 路径

- `src/sparsevllm/engine/model_runner.py`
  - 在创建 cache manager 前调用 `sync_deltakv_config_from_checkpoint(config)`；
  - 根据当前 batch 长度动态计算 long-text 阈值；
  - `deltakv-standalone` / `deltakv-snapkv` 的 long-text 定义与旧 DeltaKV 不同；
  - 保持推理过程在 `torch.inference_mode()` 下，减少不必要 autograd 图。

- `src/sparsevllm/engine/scheduler.py`
  - long-text 阈值现在按 sparse method 分别计算；
  - 对 prompt admission 增加预算检查、defer 逻辑和更详细的死锁/饥饿诊断；
  - 避免 prefill/decode/preempt 之间反复 thrash；
  - 在无可运行序列时更明确地区分“正常 idle”和“应该抛错的挂死状态”。

- `src/sparsevllm/engine/llm_engine.py`
  - warmup 长度对 standalone/snapkv 分别处理；
  - 当 scheduler 没有返回可运行序列时，增加防 busy loop 的保护和更明确的错误信息。

- `src/sparsevllm/layers/attention.py`
  - 写入 KV 时使用 `get_layer_store_view()`；
  - 计算注意力时允许从 `get_layer_compute_tensors()` 取 compute view；
  - 这使 cache manager 可以决定“存在哪里”和“算的时候从哪里读”，从而支持 standalone/snapkv 的 temp pool reconstruct 设计。

- `src/sparsevllm/engine/sparse_controller.py`
  - 显式识别 `deltakv-standalone`、`deltakv-snapkv`；
  - 对 standalone-like 方法走统一 reconstruct 路径；
  - 在 chunk prefill 结束时对 `deltakv-snapkv` 执行 finalize；
  - 对 attn score 的收集与 read view 逻辑做对应调整。

#### 标准 cache manager 和辅助逻辑的小改动

- `src/sparsevllm/engine/cache_manager/standard.py`
  - 增加 `SPARSEVLLM_DEBUG_SLOTS` 下的释放日志；
  - 实现 `debug_live_seq_slots()`。

- `src/sparsevllm/engine/cache_manager/snapkv.py`
- `src/sparsevllm/engine/cache_manager/streamingllm.py`
  - 统一改为使用显式 config 字段；
  - `prefill_batched_tokens_margin()` 与 `remaining_prefill_tokens()` 的逻辑更直接。

#### Triton kernel 组织调整

- `src/sparsevllm/triton_kernel/deltakv_kernels.py`
  - 新增/并入 `deltakv_reconstruct_writeback_grouped_heads_srcdst()`；
  - 并入原 `rekv_kernels.py` 中的 reconstruct writeback 与 blockwise L2 top-k 逻辑；
  - 继续强化 grouped-heads kernel 变体。

- `src/sparsevllm/triton_kernel/rekv_kernels.py`
  - 删除。

这说明 kernel 层做了一次“从分散文件向 DeltaKV 主 kernel 文件收拢”的整理。

#### Loader 与测试支撑

- `src/sparsevllm/utils/compressor.py`
  - `head_dim` 优先读取 `hf_config.head_dim`，兼容更多模型配置。

- `src/sparsevllm/utils/loader.py`
  - 新增 `sync_deltakv_config_from_checkpoint()`；
  - 支持从 checkpoint 目录或单文件中解析 compressor 配置；
  - 优先从 `config.json` 读取 `kv_compressed_size`、compressor 类型、intermediate size、bias 等；
  - 若缺失则回退到权重 shape 推断；
  - 检测到 split-kv checkpoint 时显式报错；
  - `load_deltakv_compressors_to_cache_manager()` 重用统一文件解析逻辑。

这块是 sparse-vLLM 稳定性的关键改动之一，因为 cache allocation 必须在知道真实 latent dim 和 compressor 结构之后再做。

### 3.3 `benchmark/`：LongBench、SCBench、MathBench、NIAH 都被补强

#### LongBench

- `benchmark/long_bench/eval.py`
  - 输出目录改成支持 `DELTAKV_OUTPUT_DIR`；
  - 新增任务层级 `TASK_HIERARCHY`；
  - 新增 category-level 聚合分数和 overall category average。

- `benchmark/long_bench/pred.py`
  - 数据根目录改成支持 `DELTAKV_LONGBENCH_DATA_DIR / DELTAKV_DATA_DIR`；
  - 启动前验证数据目录和 jsonl 文件是否存在；
  - 新增 `NO_CHAT_TEMPLATE_DATASETS`；
  - 新增 `thinking_mode`，并支持去掉 Qwen3 `<think>` 产物；
  - 新增 KVzip 的 LongBench prompt 适配，拆成 `prefill_text + query_text`；
  - 新增 `temperature / top_p / top_k / max_new_tokens_override`；
  - KVzip 在 `ws > 1` 时走“每张卡一个单 GPU worker”的启动方式；
  - 父进程负责汇总日志、自动运行 `eval.py` 并写回评测日志。

这里最有价值的变化是：LongBench 终于能比较自然地同时支持普通 HF 路径、KVzip、Qwen3 thinking template 和多卡数据并行。

#### MathBench

- `benchmark/math_bench/pred.py`
  - 新增 KVzip 的 prompt 适配逻辑；
  - 通过 `apply_chat_template(add_generation_prompt=False/True)` 切出 prefill/query 两段；
  - 为 KVzip 显式禁用 `aime2024`。

#### NIAH

- `benchmark/niah/test_niah.py`
  - 输出基目录支持环境变量；
  - 增加 `num_top_tokens_in_prefill`、`stride_alpha`、`deltakv_use_omnikv_selection`、`chunk_prefill_accel_omnikv`、`omnikv_score_method`；
  - `context_lengths` 支持 list/tuple 形式。

#### SCBench CLI 与主流程

- `benchmark/scbench/args.py`
  - 新增 `load_in_4bit`、`load_in_8bit`、`model_torch_dtype`；
  - 新增 `tensor_parallel_size`、`copy_on_gpu`；
  - 新增 `context_min_tokens`、`context_max_tokens`、`subset_indices_file`；
  - 新增 `num_data_shards`、`data_shard_id`；
  - 将 `full_deltakv`、`origin_residual_quant`、`all_origin_residual_quant` 纳入 `attn_type` 候选。

- `benchmark/scbench/run_scbench.py`
  - `BASE_PATH` 改为输出目录环境变量驱动；
  - 结果目录命名统一通过 `_build_result_dir()`；
  - 新增 subset index 文件读取与 context-length 过滤；
  - 新增每张 GPU 一个 worker 的数据并行执行与结果 merge；
  - 模型加载支持 `tensor_parallel_size`、`copy_on_gpu`、量化参数；
  - 支持 `use_chat_template`；
  - 结果记录使用统一的 `real_model_name_tag`；
  - 单卡和多卡路径都能输出更规范的 `result.json` 与日志。

- `benchmark/scbench/eval_utils.py`
  - `GreedySearch` 新增 KV 状态快照与恢复；
  - 多轮/同上下文多查询评测中，后续 query 可以基于保存的 cache 继续走，而不是每轮全重算；
  - 兼容 `prepare_inputs_for_generation()` 不返回 `past_key_values` 的情况；
  - 调用 `clear_temp_kv_cache()` 前先做 `hasattr` 检查；
  - EOS 处理更稳。

#### 新增 SCBench 预处理数据运行器

- `benchmark/scbench/run_scbench_preprocessed.py`
  - 针对 `SCBench-preprocessed` parquet 数据直接跑 DeltaKV/HF 路径；
  - 内置 KVzip 风格 prompt template 构造；
  - 支持数据并行 worker 与结果合并；
  - 直接从 prediction 文件回算得分。

- `benchmark/scbench/run_kvzip_preprocessed.py`
  - 针对 `SCBench-preprocessed` parquet 数据直接跑 KVzip；
  - 支持 `ratio / level / kv_type / prefill_chunk_size`；
  - 同样支持数据并行 worker、合并与 score 统计。

- `benchmark/scbench/scripts/run_scbench_three_eval.sh`
  - 新增批处理脚本，对三项 SCBench 任务同时跑 KVzip 与 DeltaKV。

这三项一起构成了“预处理数据评测链路”的第一版完整实现。

### 3.4 `baselines/kvzip/`：本地数据、Qwen 兼容、debug 与结果解析增强

`baselines/kvzip/` 这次的改动虽然没有大规模新增文件，但每处都很实用：

- `attention/kvcache.py`
  - 新增 debug/显存监控输出；
  - cache update OOM 时打印更具体上下文；
  - eviction 日志中附带 CUDA alloc/reserved/max alloc。

- `csrc/build.py`
  - 动态收集 CUDA 架构；
  - 在已有 `sm_80/sm_90` 基础上，把当前机器的 GPU capability 也编进来。

- `data/load.py`
  - 新增 `SCBENCH_PREPROCESSED_ROOT`；
  - 优先从本地 parquet 加载 `SCBench-preprocessed`，否则回退到 HuggingFace 数据集。

- `model/monkeypatch.py`
  - 扩大 Qwen 匹配范围，支持 `qwen2`、`distill-qwen`。

- `model/wrapper.py`
  - 在 KV score 长度不匹配时打印详细调试信息；
  - 输出 token 截取不再简单丢掉最后一位，而是按 EOS/PAD/EOT 集合裁尾；
  - `test_scdq()` 路径加入 `@torch.inference_mode()`。

- `results/parse.py`
  - 与 `data/load.py` 类似，优先从本地 `SCBench-preprocessed` 读取数据。

这些改动基本都围绕“让 KVzip 更容易参与统一 benchmark，并且更容易 debug”展开。

### 3.5 `scripts/`：批量实验与补跑脚本成体系

- `scripts/bench_sparse_vllm.py`
  - 允许通过 JSON 传入 `hyper_params`；
  - 基准测试时固定一组稳定默认值；
  - 细化 prefill/decode/TTFT/ITL/AvgBS 统计；
  - 新增 staged admission / wave decode 的 bench 逻辑；
  - 失败时保存更明确的状态。

- `scripts/queue_scbench_llama_jobs.py`
  - 新增统一排队脚本，覆盖 Qwen LongBench alpha sweep、Qwen SCBench alpha sweep、Llama DeltaKV 与 KVzip SCBench 等多类任务；
  - 把模型路径、压缩器路径、任务组合、`stride_alpha` / `token_budget` / `ratio` 等参数固化成 job queue。

- `scripts/run_llama31_alpha_sweep_longbench_scbench_b0p17.sh`
  - 批量扫 `stride_alpha`，同时跑 LongBench 和 SCBench。

- `scripts/run_llama_deltasnapkv_missing.sh`
- `scripts/run_llama_deltasnapkv_missing_ws2.sh`
- `scripts/run_llama_deltasnapkv_remaining.sh`
  - 用于补跑 Llama DeltaSnapKV 在 LongBench 上遗漏的任务，分别覆盖不同 `ws` 配置。

- `scripts/simulate_linear_stride_ref_tokens.py`
  - 纯模拟脚本，用来估算不同 `stride_alpha` 下 reference token 数量及总保留量。

这些脚本使得本分支引入的新方法和参数不只是“代码里存在”，而是已经被包装成可以直接执行的大规模实验入口。

### 3.6 README、忽略规则、测试与文档

- `README.md`
  - `flash-attn` 安装从固定版本改为不锁死版本；
  - baseline 使用说明里新增 OmniKV 和 KIVI 的例子。

- `.gitignore`
  - 新增 `wandb/`、`tmp/`、`outputs/`、`benchmark/scbench/results/`。

- 新增测试：
  - `tests/test_deltakv_checkpoint_config_sync.py`
    - 测 `sync_deltakv_config_from_checkpoint()` 是否能从 `config.json` 或权重 shape 推断 compressor 配置。
  - `tests/test_quantization_helpers.py`
    - 测低比特加载参数构造、chunk_prefill_size 保留、量化跳过模块的 dtype 恢复。

- 文档与目录整理：
  - `blog/1.md` 更名为 `docs/1.md`
  - 新增 `docs/todo.md`

- 打包元数据：
  - `src/deltakv.egg-info/PKG-INFO`
  - `src/deltakv.egg-info/SOURCES.txt`
  - 主要是把新增文件和包信息同步进元数据。

---

## 4. 逐目录逐文件清单

下面这一节按目录列出当前分支相对 `main` 的每一个变更文件及其主要作用，便于按文件回溯。

### 4.1 仓库根目录

| 文件 | 变更说明 |
| --- | --- |
| `.gitignore` | 新增训练/评测常见输出目录忽略规则。 |
| `README.md` | 更新依赖安装说明，并补充 OmniKV、KIVI 等 baseline 使用示例。 |

### 4.2 `baselines/kvzip/`

| 文件 | 变更说明 |
| --- | --- |
| `baselines/kvzip/attention/kvcache.py` | 增加 KVzip cache update/debug/OOM 输出与更详细的 eviction 显存日志。 |
| `baselines/kvzip/csrc/build.py` | 动态加入当前机器 GPU 架构，改善 CUDA extension 编译兼容性。 |
| `baselines/kvzip/data/load.py` | 增加本地 `SCBench-preprocessed` parquet 数据加载入口。 |
| `baselines/kvzip/model/monkeypatch.py` | 扩大 Qwen 模型匹配范围，支持 `qwen2` 和 `distill-qwen`。 |
| `baselines/kvzip/model/wrapper.py` | 强化 score 长度检查，修复输出裁尾逻辑，并为 SCDQ 路径加上 inference mode。 |
| `baselines/kvzip/results/parse.py` | 结果解析时优先使用本地 `SCBench-preprocessed` 数据。 |

### 4.3 `benchmark/long_bench/`

| 文件 | 变更说明 |
| --- | --- |
| `benchmark/long_bench/eval.py` | 增加环境变量输出目录、category 聚合分数和 overall category average。 |
| `benchmark/long_bench/pred.py` | 增加数据路径校验、Qwen3 thinking 处理、KVzip prompt 适配、采样参数、单卡 worker 启动和自动日志/评测。 |

### 4.4 `benchmark/math_bench/` 与 `benchmark/niah/`

| 文件 | 变更说明 |
| --- | --- |
| `benchmark/math_bench/pred.py` | 为 KVzip 增加 prompt 拆分适配，并禁用 `aime2024` 任务。 |
| `benchmark/niah/test_niah.py` | 参数面向新的 DeltaKV 配置扩展，并改成支持环境变量输出目录。 |

### 4.5 `benchmark/scbench/`

| 文件 | 变更说明 |
| --- | --- |
| `benchmark/scbench/args.py` | 扩充 CLI 参数：量化、dtype、上下文过滤、subset 文件、shard、TP、copy_on_gpu 等。 |
| `benchmark/scbench/eval_utils.py` | 为多轮评测新增 KV 状态快照/恢复逻辑，并提升 EOS/temp-cache 兼容性。 |
| `benchmark/scbench/run_kvzip_preprocessed.py` | 新增 KVzip 直跑 `SCBench-preprocessed` 的数据并行评测脚本。 |
| `benchmark/scbench/run_scbench.py` | 重构 SCBench 主流程，支持数据并行 worker、子集过滤、上下文长度过滤、量化加载和统一结果目录。 |
| `benchmark/scbench/run_scbench_preprocessed.py` | 新增 DeltaKV/HF 直跑 `SCBench-preprocessed` 的评测脚本。 |
| `benchmark/scbench/scripts/run_scbench_three_eval.sh` | 新增同时对 KVzip 与 DeltaKV 跑三项 SCBench 任务的批处理脚本。 |

### 4.6 `docs/`

| 文件 | 变更说明 |
| --- | --- |
| `docs/1.md` | 由 `blog/1.md` 重命名到 `docs/`。 |
| `docs/todo.md` | 新增 cache manager/rope 相关后续 TODO。 |

### 4.7 `scripts/`

| 文件 | 变更说明 |
| --- | --- |
| `scripts/bench_sparse_vllm.py` | 基准测试逻辑升级，支持 JSON 超参和更细的分阶段统计。 |
| `scripts/queue_scbench_llama_jobs.py` | 新增排队脚本，统一组织 alpha sweep、SCBench、KVzip 等实验任务。 |
| `scripts/run_llama31_alpha_sweep_longbench_scbench_b0p17.sh` | 新增 Llama 版 `stride_alpha` 联合扫参脚本。 |
| `scripts/run_llama_deltasnapkv_missing.sh` | 新增 DeltaSnapKV LongBench 漏跑任务补跑脚本。 |
| `scripts/run_llama_deltasnapkv_missing_ws2.sh` | 新增双卡版本的 DeltaSnapKV 漏跑任务补跑脚本。 |
| `scripts/run_llama_deltasnapkv_remaining.sh` | 新增剩余 DeltaSnapKV 任务补跑脚本。 |
| `scripts/simulate_linear_stride_ref_tokens.py` | 新增 `stride_alpha` 下 reference token 数量模拟脚本。 |

### 4.8 `src/deltakv/analysis/`

| 文件 | 变更说明 |
| --- | --- |
| `src/deltakv/analysis/verify_insight_pdf.py` | 扩展为支持 `stride_alpha`、cluster similarity summary 和更多 PDF 实验图。 |
| `src/deltakv/analysis/visualize_ablation_loss.py` | 将 cluster chunk 参数逻辑向 `k_neighbors` 语义迁移。 |

### 4.9 `src/deltakv/configs/`

| 文件 | 变更说明 |
| --- | --- |
| `src/deltakv/configs/ds_zero2_bf16.json` | 新增 DeepSpeed ZeRO-2 BF16 配置。 |
| `src/deltakv/configs/model_config_cls.py` | 新增 Qwen3 config、`k_neighbors`、`stride_alpha`、DeltaSnapKV 预算与更统一的配置解析逻辑。 |

### 4.10 `src/deltakv/` 顶层与训练/加载辅助

| 文件 | 变更说明 |
| --- | --- |
| `src/deltakv/baseline_adapters.py` | 新增 OmniKV、KIVI baseline 加载适配。 |
| `src/deltakv/data_prepare/generate_train_data.py` | 扩展 `vr1.0`，新增 `vi1.0` 混合训练数据生成与本地 parquet 数据读取。 |
| `src/deltakv/get_chat_api.py` | 统一扩展 HF 推理入口，纳入 Qwen3、量化、OmniKV/KIVI、DeltaSnapKV 和多种 DeltaKV 变体。 |
| `src/deltakv/quantization.py` | 新增低比特模型加载和跳过模块 dtype 恢复辅助。 |
| `src/deltakv/save_trainable_trainer.py` | 只保存 trainable 参数，并兼容 accelerator unwrap。 |
| `src/deltakv/train_compressor.py` | 扩展到 Qwen3、Deepspeed、多卡时间戳同步和 `k_neighbors` 配置。 |

### 4.11 `src/deltakv/modeling/` 公共层

| 文件 | 变更说明 |
| --- | --- |
| `src/deltakv/modeling/all_origin_residual_quant_cache.py` | 新增 all-origin residual quant 缓存实现。 |
| `src/deltakv/modeling/full_deltakv_compress_cache.py` | 新增 full DeltaKV 压缩缓存实现。 |
| `src/deltakv/modeling/kv_cache.py` | 扩展动态 stride center 构造、cluster 选择与新参数语义。 |
| `src/deltakv/modeling/origin_residual_quant_cache.py` | 新增 origin residual quant 缓存实现。 |

### 4.12 `src/deltakv/modeling/llama/`

| 文件 | 变更说明 |
| --- | --- |
| `src/deltakv/modeling/llama/llama_all_origin_residual_quant_inference.py` | 新增 Llama all-origin residual quant 推理实现。 |
| `src/deltakv/modeling/llama/llama_deltasnapkv.py` | 新增 Llama DeltaSnapKV 推理实现。 |
| `src/deltakv/modeling/llama/llama_e2e.py` | 同步新的配置与公共能力，服务 Llama 端到端训练路径。 |
| `src/deltakv/modeling/llama/llama_e2e_cluster.py` | 将 cluster 训练逻辑迁移到新的 `k_neighbors` 语义。 |
| `src/deltakv/modeling/llama/llama_full_deltakv_compress_inference.py` | 新增 Llama full DeltaKV 推理实现。 |
| `src/deltakv/modeling/llama/llama_origin_residual_quant_inference.py` | 新增 Llama origin residual quant 推理实现。 |
| `src/deltakv/modeling/llama/llama_pyramidkv.py` | 同步新配置/加载接口到 Llama PyramidKV。 |
| `src/deltakv/modeling/llama/llama_with_compress_inference.py` | 接入 `deltakv_use_omnikv_selection` 等新的 DeltaKV 选择逻辑。 |

### 4.13 `src/deltakv/modeling/qwen2/`

| 文件 | 变更说明 |
| --- | --- |
| `src/deltakv/modeling/qwen2/qwen2_all_origin_residual_quant_inference.py` | 新增 Qwen2 all-origin residual quant 推理实现。 |
| `src/deltakv/modeling/qwen2/qwen2_deltasnapkv.py` | 新增 Qwen2 DeltaSnapKV 推理实现。 |
| `src/deltakv/modeling/qwen2/qwen2_e2e.py` | 同步新的配置与公共能力，服务 Qwen2 端到端训练路径。 |
| `src/deltakv/modeling/qwen2/qwen2_e2e_cluster.py` | 将 cluster 训练逻辑迁移到新的 `k_neighbors` 语义。 |
| `src/deltakv/modeling/qwen2/qwen2_e2e_cluster_for_big_model.py` | 同步新的 cluster 参数语义到大模型训练路径。 |
| `src/deltakv/modeling/qwen2/qwen2_full_deltakv_compress_inference.py` | 新增 Qwen2 full DeltaKV 推理实现。 |
| `src/deltakv/modeling/qwen2/qwen2_origin_residual_quant_inference.py` | 新增 Qwen2 origin residual quant 推理实现。 |
| `src/deltakv/modeling/qwen2/qwen2_pyramidkv.py` | 同步新配置/加载接口到 Qwen2 PyramidKV。 |
| `src/deltakv/modeling/qwen2/qwen2_snapkv.py` | 同步新配置/加载接口到 Qwen2 SnapKV。 |
| `src/deltakv/modeling/qwen2/qwen2_with_compress_inference.py` | 接入新的 DeltaKV 选择与配置逻辑。 |

### 4.14 `src/deltakv/modeling/qwen3/`

| 文件 | 变更说明 |
| --- | --- |
| `src/deltakv/modeling/qwen3/__init__.py` | 新增 Qwen3 modeling package 标记文件。 |
| `src/deltakv/modeling/qwen3/qwen3_all_origin_residual_quant_inference.py` | 新增 Qwen3 all-origin residual quant 推理实现。 |
| `src/deltakv/modeling/qwen3/qwen3_e2e.py` | 新增 Qwen3 端到端 DeltaKV 训练实现。 |
| `src/deltakv/modeling/qwen3/qwen3_e2e_cluster.py` | 新增 Qwen3 cluster 训练实现。 |
| `src/deltakv/modeling/qwen3/qwen3_e2e_cluster_for_big_model.py` | 新增 Qwen3 大模型 cluster 训练实现。 |
| `src/deltakv/modeling/qwen3/qwen3_full_deltakv_compress_inference.py` | 新增 Qwen3 full DeltaKV 推理实现。 |
| `src/deltakv/modeling/qwen3/qwen3_origin_residual_quant_inference.py` | 新增 Qwen3 origin residual quant 推理实现。 |
| `src/deltakv/modeling/qwen3/qwen3_with_compress_inference.py` | 新增 Qwen3 普通 DeltaKV 推理实现。 |

### 4.15 `src/sparsevllm/`

| 文件 | 变更说明 |
| --- | --- |
| `src/sparsevllm/config.py` | 新增 `deltakv-standalone` / `deltakv-snapkv` 配置入口并调整若干默认值。 |
| `src/sparsevllm/engine/cache_manager/__init__.py` | 注册新的 standalone/snapkv cache manager。 |
| `src/sparsevllm/engine/cache_manager/base.py` | 扩展 cache manager 工厂逻辑，并禁用 qwen3+sparsevllm deltakv。 |
| `src/sparsevllm/engine/cache_manager/deltakv.py` | 继续补强 DeltaKV cache manager 的容量预算与参数访问逻辑。 |
| `src/sparsevllm/engine/cache_manager/deltakv_snapkv.py` | 新增 sparse-vLLM 侧 DeltaKV+SnapKV 混合 cache manager。 |
| `src/sparsevllm/engine/cache_manager/deltakv_standalone.py` | 新增 sparse-vLLM 侧全层 DeltaKV standalone cache manager。 |
| `src/sparsevllm/engine/cache_manager/snapkv.py` | 调整 SnapKV prefill margin 与 remaining token 计算。 |
| `src/sparsevllm/engine/cache_manager/standard.py` | 增加 slot debug 日志和 live-slot 调试接口。 |
| `src/sparsevllm/engine/cache_manager/streamingllm.py` | 调整 StreamingLLM prefill margin 与 remaining token 计算。 |
| `src/sparsevllm/engine/llm_engine.py` | 根据新 sparse method 改进 warmup 与无可运行序列时的保护逻辑。 |
| `src/sparsevllm/engine/model_runner.py` | 在 cache 分配前同步 DeltaKV checkpoint 配置，并适配新 long-text 语义。 |
| `src/sparsevllm/engine/scheduler.py` | 强化 admission/defer/preempt 逻辑，适配 standalone/snapkv 长文本阈值。 |
| `src/sparsevllm/engine/sparse_controller.py` | 识别新的 DeltaKV 变体并管理对应 reconstruct/finalize 流程。 |
| `src/sparsevllm/layers/attention.py` | 抽象“存储视图”和“计算视图”，支持 temp pool reconstruct。 |
| `src/sparsevllm/triton_kernel/deltakv_kernels.py` | 并入更多 reconstruct/top-k kernel，并新增 src-dst writeback 变体。 |
| `src/sparsevllm/triton_kernel/rekv_kernels.py` | 删除，相关 kernel 逻辑并入 `deltakv_kernels.py`。 |
| `src/sparsevllm/utils/compressor.py` | `head_dim` 优先从 HF config 显式读取。 |
| `src/sparsevllm/utils/loader.py` | 新增从 DeltaKV checkpoint 自动同步配置并统一压缩器权重文件解析。 |

### 4.16 测试与打包元数据

| 文件 | 变更说明 |
| --- | --- |
| `src/deltakv.egg-info/PKG-INFO` | 打包元数据同步更新。 |
| `src/deltakv.egg-info/SOURCES.txt` | 打包源文件清单同步更新。 |
| `tests/test_deltakv_checkpoint_config_sync.py` | 新增 DeltaKV checkpoint 配置同步测试。 |
| `tests/test_quantization_helpers.py` | 新增量化辅助函数测试。 |

---

## 5. 这个分支最终带来了什么

如果只用一句话概括，这个分支把仓库从“已经有 Sparse-vLLM/DeltaKV 原型”推进成了“可以稳定比较多种 baseline、多种 DeltaKV 变体、两条运行时路径、多个 benchmark，并且能继续训练/分析/补跑”的工程状态。

更具体地说，它带来的增量主要是：

- 方法维度更完整：DeltaKV、full DeltaKV、origin residual quant、all-origin residual quant、DeltaSnapKV、OmniKV、KIVI、KVzip 等都进入统一实验框架。
- 模型维度更完整：Qwen3 被正式接入 HF 训练/推理体系。
- 运行时维度更完整：sparse-vLLM 新增 standalone 与 snapkv 混合变体。
- 实验维度更完整：LongBench、SCBench、MathBench、NIAH 都补齐了大量跑实验所需的工程细节。

因此，`add-baseline` 分支的本质不是“多加了几个 baseline 文件”，而是一次围绕“基线对比与大规模实验”展开的系统性扩展。
