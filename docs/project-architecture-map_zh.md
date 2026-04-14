# Sparse-vLLM 项目架构地图（含 DeltaKV）

## 1. 文档目标

本文用于快速建立对 `Sparse-vLLM` 仓库的系统认知，覆盖：

1. 项目总体结构。
2. 各目录职责。
3. 关键文件职责（按模块分组）。
4. DeltaKV 论文与代码实现的对应关系。
5. 推荐阅读顺序。

说明：仓库规模较大（尤其是 `baselines/` 与 `triton_kernel/`），本文对核心实现文件做逐项说明，对大规模算子文件做功能分组说明。

---

## 2. 论文与代码的对应关系（DeltaKV + Sparse-vLLM）

论文主线：

1. **DeltaKV**：通过“全局参考检索 + 残差压缩 + 按需重构”降低 KV 内存。
2. **Sparse-vLLM**：通过可插拔 CacheManager、稀疏控制器和稀疏友好 kernel，把压缩收益转化为真实吞吐提升。

仓库中的主要对应：

- DeltaKV 方法家族：`src/deltakv/`
- Sparse 推理引擎：`src/sparsevllm/`
- 论文实验复现入口：`benchmark/`、`scripts/`
- 论文文档：`DeltaKV.pdf`

关键映射：

- 论文中 DeltaKV 残差压缩思想：`src/deltakv/modeling/kv_cache.py`
- 论文中 Sparse-vLLM 组件化设计：
  - `src/sparsevllm/engine/cache_manager/`
  - `src/sparsevllm/engine/sparse_controller.py`
  - `src/sparsevllm/triton_kernel/`

---

## 3. 顶层目录与文件职责

| 路径 | 作用 |
| --- | --- |
| `README.md` | 项目总说明（安装、方法、参数、评测入口）。 |
| `README_ch.md` | 中文版 README。 |
| `DeltaKV.pdf` | DeltaKV 论文本地副本。 |
| `pyproject.toml` | 包管理与脚本入口（如 `deltakv-train`）。 |
| `src/` | 核心源码：`deltakv`（方法）+ `sparsevllm`（引擎）。 |
| `benchmark/` | LongBench、MathBench、NIAH、SCBench 评测脚本。 |
| `baselines/` | 各外部/对照方法（adakv、kivi、kvzip、palu、quest）。 |
| `scripts/` | 批量实验、可视化、带宽/性能测试和补跑脚本。 |
| `docs/` | 设计说明与参数差异文档。 |
| `tests/` | 单元测试。 |
| `assets/` | README 图像资源。 |
| `skills/` | 仓库内技能定义。 |
| `AGENTS.md` | 仓库代理/技能使用说明。 |
| `.gitignore` | 忽略规则。 |
| `LICENSE` | 许可证。 |

---

## 4. `src/sparsevllm/`：推理引擎主线

### 4.1 目录职责

| 路径 | 作用 |
| --- | --- |
| `src/sparsevllm/config.py` | 运行时统一配置中心（方法切换、预算、调度参数）。 |
| `src/sparsevllm/llm.py` | 用户入口类 `LLM`（继承 `LLMEngine`）。 |
| `src/sparsevllm/sampling_params.py` | 生成参数定义。 |
| `src/sparsevllm/engine/` | 调度、执行、序列状态、稀疏控制器。 |
| `src/sparsevllm/engine/cache_manager/` | 各稀疏方法的缓存管理实现。 |
| `src/sparsevllm/layers/` | 模型基础层（attention/linear/norm 等）。 |
| `src/sparsevllm/models/` | 模型适配（Qwen2/Qwen3/DeepSeek）。 |
| `src/sparsevllm/triton_kernel/` | 高性能 Triton 算子。 |
| `src/sparsevllm/cuda_kernel/` | 自定义 CUDA 扩展（offload 相关）。 |
| `src/sparsevllm/utils/` | 加载器、压缩器构建、日志、profiler 工具。 |

### 4.2 关键文件说明（engine）

- `src/sparsevllm/engine/llm_engine.py`：
  - 引擎主循环，负责请求接入、warmup、逐步 `step()` 推理。
  - 连接 Scheduler 与 ModelRunner，维护多进程 TP 生命周期。
- `src/sparsevllm/engine/scheduler.py`：
  - 负责 prefill/decode 调度。
  - 处理 prompt 准入、defer/preempt、长短文本分离策略。
- `src/sparsevllm/engine/model_runner.py`：
  - 调用模型前向并协调 cache manager 状态。
  - 在 DeltaKV 路径中会先做 checkpoint 配置同步。
- `src/sparsevllm/engine/sparse_controller.py`：
  - 稀疏视图构建与后处理。
  - 管理 DeltaKV/OmniKV/SnapKV 等方法的前后向生命周期。
- `src/sparsevllm/engine/sequence.py`：
  - 单请求状态机（prompt、已生成 token、结束状态等）。

### 4.3 关键文件说明（cache_manager）

- `src/sparsevllm/engine/cache_manager/base.py`：
  - 抽象基类 + 工厂分发逻辑。
  - 根据 `vllm_sparse_method` 创建具体 manager。
- `src/sparsevllm/engine/cache_manager/standard.py`：全注意力基线路径。
- `src/sparsevllm/engine/cache_manager/streamingllm.py`：attention-sink / StreamingLLM。
- `src/sparsevllm/engine/cache_manager/snapkv.py`：SnapKV/PyramidKV 的物理驱逐实现。
- `src/sparsevllm/engine/cache_manager/omnikv.py`：OmniKV 逻辑选择/masking。
- `src/sparsevllm/engine/cache_manager/quest.py`：Quest 的 query-aware 选择实现。
- `src/sparsevllm/engine/cache_manager/deltakv.py`：DeltaKV 主缓存实现（含 triton/offload 变体）。
- `src/sparsevllm/engine/cache_manager/deltakv_standalone.py`：全层 DeltaKV standalone。
- `src/sparsevllm/engine/cache_manager/deltakv_snapkv.py`：DeltaKV + SnapKV 混合路径。
- `src/sparsevllm/engine/cache_manager/deepseek_mla.py`：DeepSeek MLA 特殊缓存布局。
- `src/sparsevllm/engine/cache_manager/__init__.py`：注册导出。

### 4.4 关键文件说明（layers/models/utils）

- `src/sparsevllm/layers/attention.py`：
  - 把“写入 KV（store view）”和“读取计算视图（compute view）”解耦。
  - 是 DeltaKV 重构池视图接入的关键点。
- `src/sparsevllm/layers/activation.py`：激活函数封装。
- `src/sparsevllm/layers/embed_head.py`：embedding/head 封装。
- `src/sparsevllm/layers/layernorm.py`：归一化算子封装。
- `src/sparsevllm/layers/linear.py`：线性层封装。
- `src/sparsevllm/layers/rotary_embedding.py`：RoPE 相关。
- `src/sparsevllm/layers/sampler.py`：采样逻辑。

- `src/sparsevllm/models/qwen2.py`：Qwen2 模型结构接线。
- `src/sparsevllm/models/qwen3.py`：Qwen3 模型结构接线。
- `src/sparsevllm/models/deepseek_v2.py`：DeepSeek-V2 适配。
- `src/sparsevllm/models/deepseek_v32.py`：DeepSeek-V3.2 适配。

- `src/sparsevllm/utils/loader.py`：
  - 权重加载。
  - DeltaKV checkpoint 配置同步（`sync_deltakv_config_from_checkpoint`）。
  - compressor 权重解析与结构对齐。
- `src/sparsevllm/utils/compressor.py`：compressor/down-up 模块构建。
- `src/sparsevllm/utils/log.py`：日志。
- `src/sparsevllm/utils/profiler.py`：性能统计。
- `src/sparsevllm/utils/context.py`：上下文辅助。
- `src/sparsevllm/utils/flash_mla.py`：FlashMLA 辅助。

### 4.5 Triton/CUDA 内核文件（按功能分组）

`src/sparsevllm/triton_kernel/` 文件较多，按职责分组：

1. **DeltaKV 专用**
   - `deltakv_kernels.py`：重构、writeback、top-k 等核心 kernel。
2. **解码路径**
   - `flash_decoding.py`
   - `flash_decoding_stage1.py`
   - `flash_decoding_stage2.py`
   - `gqa_flash_decoding.py`
   - `gqa_flash_decoding_stage1.py`
   - `gqa_flash_decoding_stage2.py`
3. **上下文/注意力相关**
   - `context_flashattention_nopad.py`
   - `gqa_decode_flashattention_nopad.py`
   - `splitfuse_context_flashattention_nopad.py`
   - `token_attention_nopad_att1.py`
   - `token_attention_nopad_softmax.py`
   - `token_attention_nopad_reduceV.py`
   - `token_attention_softmax_and_reducev.py`
4. **OmniKV 与量化路径**
   - `omnikv_fused.py`
   - `quant.py`
   - `ppl_int4kv_copy_kv.py`
   - `ppl_int4kv_flash_decoding.py`
   - `ppl_int8kv_flash_decoding.py`
   - `ppl_quant_copy_kv.py`
   - `ppl_fp16_flash_decoding.py`
5. **基础算子**
   - `embedding.py`
   - `rmsnorm.py`
   - `rotary_emb.py`
   - `silu_and_mul.py`
   - `deepseek_v2_prefill.py`
   - `__init__.py`

`src/sparsevllm/cuda_kernel/`：

- `setup.py`：CUDA 扩展编译入口。
- `api.py`：Python 调用接口。
- `kernels/`：CUDA 核心实现。
- `test_api_simple.py`：简单功能测试。
- `README.md`：构建/使用说明。

---

## 5. `src/deltakv/`：方法、训练与 HF 推理主线

### 5.1 目录职责

| 路径 | 作用 |
| --- | --- |
| `src/deltakv/get_chat_api.py` | 统一生成接口工厂：HF backend 与 sparsevllm backend 分流。 |
| `src/deltakv/train_compressor.py` | DeltaKV 压缩器训练入口。 |
| `src/deltakv/configs/` | 配置类与训练配置文件。 |
| `src/deltakv/modeling/` | DeltaKV 相关 cache/model 实现。 |
| `src/deltakv/data_prepare/` | 数据构建、tokenize、pack、collator。 |
| `src/deltakv/analysis/` | 论文洞察与消融可视化脚本。 |
| `src/deltakv/quantization.py` | 低比特加载与 dtype 恢复工具。 |
| `src/deltakv/baseline_adapters.py` | OmniKV/KIVI 适配加载。 |
| `src/deltakv/save_trainable_trainer.py` | 训练参数保存逻辑。 |
| `src/deltakv/utils/log.py` | 日志。 |

### 5.2 关键文件说明（入口与配置）

- `src/deltakv/get_chat_api.py`：
  - 统一封装 `get_generate_api(...)`。
  - 支持 `backend='hf'` 与 `backend='sparsevllm'`。
  - 根据 `model_cls` 选择 DeltaKV/OmniKV/KIVI/SnapKV/PyramidKV/Quest/KVzip 等加载路径。
- `src/deltakv/train_compressor.py`：
  - 训练 DeltaKV 压缩器（Qwen2/Qwen3/Llama）。
  - 支持 cluster 训练、DeepSpeed、多卡时间戳同步、量化训练准备。
- `src/deltakv/configs/model_config_cls.py`：
  - 扩展 HF config，统一 DeltaKV 推理参数。
  - 支持 `k_neighbors`、`stride_alpha`、`parse_full_attn_layers`、兼容旧参数。
- `src/deltakv/configs/ds_zero2_bf16.json`：DeepSpeed 配置。
- `src/deltakv/configs/ac_2gpu.yaml` / `ac_4gpu.yaml`：多卡实验配置模板。

### 5.3 `modeling/` 关键文件说明

通用层：

- `src/deltakv/modeling/kv_cache.py`：
  - DeltaKV cache 核心逻辑。
  - 动态 stride 中心、cluster 检索、残差存储/重构相关能力。
- `src/deltakv/modeling/token_select.py`：token 选择策略。
- `src/deltakv/modeling/full_deltakv_compress_cache.py`：full DeltaKV cache 实现。
- `src/deltakv/modeling/origin_residual_quant_cache.py`：origin residual quant cache。
- `src/deltakv/modeling/all_origin_residual_quant_cache.py`：all-origin residual quant cache。

Llama 家族（`src/deltakv/modeling/llama/`）：

- `llama_with_compress_inference.py`：Llama DeltaKV 主推理。
- `llama_e2e.py`：端到端训练路径。
- `llama_e2e_cluster.py`：cluster 训练路径。
- `llama_snapkv.py`：SnapKV 路径。
- `llama_pyramidkv.py`：PyramidKV 路径。
- `llama_deltasnapkv.py`：DeltaSnapKV 路径。
- `llama_full_deltakv_compress_inference.py`：full DeltaKV 推理。
- `llama_origin_residual_quant_inference.py`：origin residual quant 推理。
- `llama_all_origin_residual_quant_inference.py`：all-origin residual quant 推理。
- `__init__.py`：包初始化。

Qwen2 家族（`src/deltakv/modeling/qwen2/`）：

- `qwen2_with_compress_inference.py`：Qwen2 DeltaKV 主推理。
- `qwen2_e2e.py`：端到端训练。
- `qwen2_e2e_cluster.py`：cluster 训练。
- `qwen2_e2e_cluster_for_big_model.py`：大模型 cluster 训练。
- `qwen2_snapkv.py`：SnapKV 路径。
- `qwen2_pyramidkv.py`：PyramidKV 路径。
- `qwen2_deltasnapkv.py`：DeltaSnapKV 路径。
- `qwen2_full_deltakv_compress_inference.py`：full DeltaKV 推理。
- `qwen2_origin_residual_quant_inference.py`：origin residual quant 推理。
- `qwen2_all_origin_residual_quant_inference.py`：all-origin residual quant 推理。
- `__init__.py`：包初始化。

Qwen3 家族（`src/deltakv/modeling/qwen3/`）：

- `qwen3_with_compress_inference.py`：Qwen3 DeltaKV 主推理。
- `qwen3_e2e.py`：端到端训练。
- `qwen3_e2e_cluster.py`：cluster 训练。
- `qwen3_e2e_cluster_for_big_model.py`：大模型 cluster 训练。
- `qwen3_full_deltakv_compress_inference.py`：full DeltaKV 推理。
- `qwen3_origin_residual_quant_inference.py`：origin residual quant 推理。
- `qwen3_all_origin_residual_quant_inference.py`：all-origin residual quant 推理。
- `__init__.py`：包初始化。

### 5.4 `data_prepare/` 与 `analysis/`

`src/deltakv/data_prepare/`：

- `generate_train_data.py`：训练数据混配和生成。
- `naive_tokenize_and_pack.py`：tokenize + pack 处理。
- `data_collator.py`：训练批数据整理。

`src/deltakv/analysis/`：

- `verify_insight_pdf.py`：多实验洞察分析脚本（stride_alpha 等）。
- `visualize_ablation_loss.py`：消融 loss 可视化。
- `visualize_detach_ablation.py`：detach 相关可视化。
- `analyze_comp_kv_range.py`：压缩值域分析。
- `inter_layer_attn_similarity_analysis.py`：跨层注意力相似性。
- `inter_layer_qkv_similarity_analysis.py`：跨层 QKV 相似性。
- `intra_layer_qkv_similarity_analysis.py`：层内 QKV 相似性。
- `intra_layer_topk_similarity_analysis.py`：层内 top-k 相似性。
- `colors.py`：绘图颜色配置。
- `__init__.py`：包初始化。

---

## 6. `benchmark/`：评测体系

### 6.1 顶层职责

| 路径 | 作用 |
| --- | --- |
| `benchmark/long_bench/` | 长上下文通用理解评测。 |
| `benchmark/math_bench/` | 数学题评测（GSM8K/AIME 风格）。 |
| `benchmark/niah/` | Needle-in-a-Haystack 评测。 |
| `benchmark/scbench/` | 多轮共享上下文评测。 |

### 6.2 `benchmark/long_bench/`

- `pred.py`：预测入口，支持 HF/sparsevllm 后端，支持 KVzip/Qwen3 thinking 兼容。
- `eval.py`：评分汇总。
- `metrics.py`：指标实现。
- `task.md` / `task_zh.md`：任务说明。
- `config/`：任务配置。
- `longbench_dataset/`：数据文件。
- `refs/`：参考答案/结果。
- `summ/`：汇总结果。
- `requirements.txt`：依赖。
- `LICENSE`：许可。

### 6.3 `benchmark/math_bench/`

- `pred.py`：预测脚本，支持 backend 切换与 KVzip prompt 适配。
- `eval.py`：评分脚本。
- `README.md`：说明文档。

### 6.4 `benchmark/niah/`

- `gen_niah.py`：NIAH 数据合成。
- `test_niah.py`：NIAH 评测与可视化。

### 6.5 `benchmark/scbench/`

- `args.py`：CLI 参数定义。
- `run_scbench.py`：主评测流程。
- `run_scbench_preprocessed.py`：预处理数据版流程。
- `run_kvzip_preprocessed.py`：KVzip 预处理流程。
- `eval_utils.py`：GreedySearch、多轮缓存管理、评测辅助。
- `compute_scores.py`：统一算分。
- `repo_qa_utils.py`：RepoQA 工具。
- `readme.md`：SCBench 原始说明。
- `cache_blend.yaml`：配置。
- `scripts/`：批量运行脚本。
- `setup/`：环境准备脚本。
- `requirements.txt`：依赖。

`benchmark/scbench/scripts/`：

- `run_all_tasks.sh`：全任务批跑。
- `run_scbench_three_eval.sh`：三任务组合评测。
- `run_single_method.sh`：单方法运行。
- `test_llama.sh`：Llama 快速测试。
- `test_minference_with_snapkv.sh`：minference+snapkv 测试。

---

## 7. `baselines/`：对照方法代码

| 路径 | 作用 |
| --- | --- |
| `baselines/adakv/` | AdaKV 方法代码、构建与实验脚本。 |
| `baselines/kivi/` | KIVI 量化方法实现与评测脚本。 |
| `baselines/kvzip/` | KVzip 方法实现、CUDA 扩展和评测。 |
| `baselines/palu/` | Palu 低秩压缩方法实现与实验脚本。 |
| `baselines/quest/` | Quest 方法实现（evaluation/kernels/scripts）。 |

说明：这些目录大多是“独立子项目”，通常包含 README、requirements、pyproject、核心模型代码和 C++/CUDA 扩展源码。

---

## 8. `scripts/`：实验与分析工具集

主要脚本职责：

- `bench_sparse_vllm.py`：吞吐基准（TTFT、prefill、decode、ITL、显存）。
- `queue_scbench_llama_jobs.py`：批量任务排队调度。
- `run_llama31_alpha_sweep_longbench_scbench_b0p17.sh`：alpha sweep。
- `run_llama_ablation.sh`：消融批跑。
- `run_llama_deltasnapkv_missing.sh`：缺失任务补跑。
- `run_llama_deltasnapkv_missing_ws2.sh`：双卡补跑。
- `run_llama_deltasnapkv_remaining.sh`：剩余任务补跑。
- `simulate_linear_stride_ref_tokens.py`：stride 参考 token 数模拟。
- `compare_hf_sparsevllm_deepseek_v2.py`：后端/实现对比。
- `compare_indices.py`：索引比较。
- `get_train_dataset_loss.py`：训练数据 loss 统计。
- `plot_throughput_chart.py`：吞吐图绘制。
- `stat_longbench_len.py`：长度统计。
- `tune_omnikv.py`：OmniKV 调参。
- `visualize_profiling.py`：profile 可视化。
- `test_compressor_chunk_loss.py`：压缩器损失检查。
- `test_derope.py`：de-rope 验证。
- `test_gather_bandwidth.py`：gather 带宽测试。
- `test_gqa_flash_decoding_score.py`：GQA 解码分数测试。
- `test_llama_snapkv_init.py`：Llama SnapKV 初始化测试。
- `test_omnikv_fused_compare.py`：OmniKV 融合对比。
- `test_pcie_bandwidth.py`：PCIe 带宽测试。
- `test_sparse_vllm_correctness.py`：引擎正确性测试。
- `test_tensor_op_overhead.py`：张量算子开销测试。
- `_exp_lst.py`：实验列表辅助。
- `kernel_bench/`：kernel 基准目录。

---

## 9. `docs/` 与 `tests/`

### 9.1 `docs/`

- `hf-vs-sparsevllm-parameter-guide.md`：两后端参数语义差异总览。
- `hf-vs-sparsevllm-parameter-guide_ch.md`：中文翻译版。
- `add-baseline-vs-main.md`：分支改动说明。
- `add-baseline-vs-main_ch.md`：中文翻译版。
- `todo.md`：后续开发事项。
- `project-architecture-map_zh.md`：当前文档。

### 9.2 `tests/`

- `test_deltakv_checkpoint_config_sync.py`：
  - 验证 DeltaKV checkpoint 配置自动同步逻辑。
  - 关注 config.json 与权重形状回退推断。
- `test_quantization_helpers.py`：
  - 验证量化辅助函数构建。
  - 验证跳过量化模块的 dtype 恢复。

---

## 10. 推荐阅读路径（上手顺序）

建议按下面顺序阅读代码：

1. `README.md`：理解项目目标、方法入口和运行参数。
2. `src/deltakv/get_chat_api.py`：理解 HF 与 sparsevllm 两条推理路径。
3. `src/sparsevllm/config.py`：理解方法切换与资源预算参数。
4. `src/sparsevllm/engine/llm_engine.py` + `scheduler.py`：理解运行时主循环与调度。
5. `src/sparsevllm/engine/cache_manager/base.py` + 对应方法 manager：理解 KV 生命周期。
6. `src/sparsevllm/layers/attention.py`：理解 store/compute view 解耦。
7. `src/deltakv/modeling/kv_cache.py`：理解 DeltaKV 残差压缩主逻辑。
8. `benchmark/long_bench/pred.py` 与 `benchmark/scbench/run_scbench.py`：理解评测入口。
9. `scripts/bench_sparse_vllm.py`：理解性能测量口径。

---

## 11. 一句话总结

这个仓库本质上是一个“**方法层（DeltaKV 家族）+ 运行时层（Sparse-vLLM 引擎）+ 评测层（benchmark/scripts）**”三层协同系统：

- `src/deltakv` 负责“怎么压缩/重构 KV”；
- `src/sparsevllm` 负责“怎么把稀疏/压缩在工程上高效跑起来”；
- `benchmark` 与 `scripts` 负责“怎么可复现地比较方法效果与吞吐效率”。
