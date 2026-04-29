# Codex 长期上下文：Sparse-vLLM 与 DeltaKV

> 目的：这份笔记用于后续会话快速恢复项目背景。它来自对 `DeltaKV.pdf`、两份 DeltaKV 笔记、`docs/project-architecture-map_zh.md`、用户三张流程图，以及当前代码主路径的阅读。

## 1. 项目定位

Sparse-vLLM 仓库有两条主线：

- `src/deltakv/`：偏 HF/Transformers 风格的方法实现、压缩器训练、包装模型、评测后端 glue。核心是“怎么压缩/重构 KV”。
- `src/sparsevllm/`：独立 sparse-first 推理引擎。核心是 `sparsevllm.LLM`、调度器、`ModelRunner`、`CacheManager`、`SparseController` 和 Triton kernel。核心是“怎么把稀疏/压缩在真实推理系统里跑起来”。

评测入口：

- `benchmark/long_bench/pred.py` 与 `benchmark/math_bench/pred.py` 都通过 `deltakv.get_chat_api.get_generate_api(...)` 创建生成函数。
- 当 `backend="hf"` 时，使用 `src/deltakv/modeling/` 下的 HF 包装模型或 baseline。
- 当 `backend="sparsevllm"` 时，直接构造 `sparsevllm.LLM(...)`，方法选择完全来自 `infer_config` / `--hyper_param` 中的 `vllm_sparse_method`、`deltakv_path` 等。

## 2. DeltaKV 论文核心

DeltaKV 的问题意识不是“哪些 token 可以删”，而是“KV cache 里大量共享成分能否只存一次，其他 token 只存残差”。

关键观察：

- 长距离 token 相似性：KV 表示的历史最大相似度均值约 0.92，约 59.7% 的最近相似参考距离大于 16，说明不能只依赖局部窗口。
- 高度共享潜在成分：原始 KV 有强各向异性和高范数共享方向。减去相似参考均值后，残差更低能量、更接近零中心、更适合低维压缩和量化。

方法公式：

- 对当前 token 的 pre-RoPE KV 拼接向量 `KV`，从步幅参考集里找 top-k 相似参考，求均值得到 `KV_R`。
- 压缩残差：`z_delta = f_c(KV) - f_c(KV_R)`。
- 重建：`KV_hat = f_d(z_delta) + KV_R`。
- 训练：冻结 LLM，只训练压缩/解压模块，目标是 MSE 重构损失 + NTP 下一词预测损失。

系统含义：

- 参考 token / sink / recent / full-attention 层保留全精度 KV。
- 大多数历史 token 存低维 latent residual。
- 与 OmniKV 这类动态稀疏注意力结合时，只对被选中的重要 token 按需重建，避免“压缩了又全量解压”的收益抵消。

重要边界：

- 29% KV keep ratio 主要来自 DeltaKV + 4-bit residual 设置；普通 DeltaKV 常见 KR 约 43%-48%。
- DeltaKV 不是完全无损：LongBench 很接近完整缓存，但 SCBench 的 R.KV 精确字符串检索和 AIME 复杂推理仍有差距。
- Sparse-vLLM 的吞吐收益依赖 CacheManager、SparseController、间接寻址 kernel、重建 kernel 等系统工程，不只是压缩器本身。

## 3. Sparse-vLLM 初始化流程

入口是 `src/sparsevllm/llm.py` 中的 `LLM`，它只是继承 `LLMEngine`。

`LLMEngine.__init__` 主线：

1. 用 `Config(model, **kwargs)` 解析运行参数、HF config、稀疏方法参数。
2. 若 `tensor_parallel_size > 1`，为 rank 1..N-1 spawn 子进程，每个子进程也创建 `ModelRunner`。
3. rank 0 在主进程创建 `ModelRunner(config, 0, events)`。
4. 加载 tokenizer，设置 `config.eos`。
5. 创建 `Scheduler(config, self.model_runner.cache_manager)`。调度器把 CacheManager 当作 memory oracle，用它判断剩余槽位和准入预算。
6. `_warmup()` 提交一个 dummy prompt，跑完整 prefill/decode，让算子编译和显存分配提前发生。

`ModelRunner.__init__` 主线：

1. 初始化 torch/distributed，绑定当前 rank 的 GPU。
2. 根据 `hf_config.model_type` 创建模型，目前核心路径是 `Qwen2ForCausalLM` / `Qwen3ForCausalLM` / DeepSeek-V2。
3. `load_model(...)` 加载权重分片。
4. 创建 `Sampler()`。
5. `sync_deltakv_config_from_checkpoint(config)` 同步 DeltaKV checkpoint 里的压缩器结构配置。
6. `CacheManager.create(config, rank, world_size)` 根据 `vllm_sparse_method` 创建具体缓存管理器。
7. 创建 `SparseController(config, cache_manager)`，并注入到 Qwen 模型的 `model.layers` 侧。
8. 若使用 DeltaKV，`load_deltakv_compressors()` 将 compressor 权重加载到 CacheManager。

`CacheManager.create(...)` 是稀疏方法分发中心：

- `""` -> `StandardCacheManager`
- `streamingllm` / `attention_sink` -> `StreamingLLMCacheManager`
- `snapkv` / `pyramidkv` -> `SnapKVCacheManager`
- `quest` -> `QuestCacheManager`
- `omnikv` -> `OmniKVCacheManager`
- `deltakv` / `deltakv-triton*` / offload variants -> `deltakv.py` 中的 DeltaKV managers
- `deltakv-standalone` -> `DeltaKVStandaloneCacheManager`
- `deltakv-snapkv` -> `DeltaKVSnapKVCacheManager`

当前约束：

- sparsevllm 里 `qwen3 + deltakv` 被禁用，提示 qk-norm/runtime mismatch；Qwen3 DeltaKV 走 HF backend。
- DeepSeek-V3.2 sparsevllm 路径目前被禁用。
- DeltaKV runtime manager 当前断言 `world_size == 1`，压缩器路径暂不支持 TP。

## 4. 推理主循环

外部使用方式：

- 高层：`llm.generate(prompts, SamplingParams(...))`
- 手动/benchmark：先 `llm.add_request(...)`，再循环 `while not llm.is_finished(): llm.step()`

`LLMEngine.add_request`：

- 字符串 prompt 会先 tokenizer encode。
- 检查 `prompt_len + max_tokens <= max_model_len`。
- 创建 `Sequence(prompt, sampling_params)` 并交给 `Scheduler.add(seq)`。

`LLMEngine.step`：

1. `scheduler.schedule()` 选择本轮序列，返回 `(seqs, is_prefill, preempted_seqs)`。
2. 对被抢占序列调用 `model_runner.call("free_slots", seq_id)` 释放 KV。
3. 若有任务，调用 `model_runner.call("run", seqs, is_prefill)`。TP 下 rank 0 会通过共享内存广播同一方法给其他 rank。
4. `scheduler.postprocess(...)` 更新 prefill 进度、把完成 prefill 的序列转入 decoding、追加新 token、检查 EOS/max_tokens。
5. 对完成序列释放 slots，并返回 `(finished_outputs, num_tokens)`。

`num_tokens` 的符号约定：

- prefill step 返回正数，值为本轮处理的 prompt chunk token 数。
- decode step 返回负数，值为 `-len(seqs)`。

## 5. Scheduler 语义

`Scheduler` 维护两个队列：

- `waiting`：新请求、未完成 chunked prefill 的请求、被抢占后需要重新 prefill 的请求。
- `decoding`：已经完成 prompt prefill、正在逐 token 生成的请求。

调度策略：

- 不混跑 prefill 和 decode。
- 有 waiting 时优先 prefill。
- prefill 会按 `chunk_prefill_size` 分块，`Sequence.current_chunk_size` 记录本轮 chunk 大小。
- decode 只在 waiting 为空时调度。
- CacheManager 提供 `num_free_slots`、`prompt_admission_budgets(...)`、`prompt_admission_costs(...)`、`reserved_prefill_slots(...)` 等，调度器据此判断准入、defer 或 preempt。

`Sequence` 关键状态：

- `token_ids` / `last_token`
- `num_prompt_tokens`
- `num_prefilled_tokens`
- `current_chunk_size`
- `num_completion_tokens`
- `is_last_chunk_prefill`

## 6. Attention 与 SparseController 的交界

Qwen2 模型每层执行：

1. `Qwen2Attention` 做 qkv projection 和 RoPE。
2. 调用通用 `Attention.forward(q, k, v)`。
3. 每层结束后，`Qwen2Model.forward` 调用 `sparse_controller.on_layer_end(i, context)`。

`Attention.forward` 的顺序非常关键：

1. 从 CacheManager 取本层 store view：`get_layer_store_view(layer_idx)`。
2. 用 `store_kvcache(...)` 把当前 token/chunk 的 K/V 写入物理 KV cache。
3. 调用 `cache_manager.on_kv_stored(...)`，给方法特定逻辑留 hook。
4. 调用 `sparse_controller.get_read_view(layer_idx)` 获取本层计算视图：
   - `active_slots`
   - `req_indices`
   - `context_lens`
   - `attn_score`
   - DeltaKV 临时重建 slots
5. prefill 调 `context_attention_fwd(...)`。
6. decode 先可选调用 `cache_manager.build_decode_view(...)`，再跑 flash decode stage1/stage2。
7. 若 DeltaKV 使用了临时重建槽位，finally 中调用 `free_temp_deltakv_full(...)` 回收。

设计原则：

- `attention.py` 尽量保持方法无关。
- 稀疏/压缩方法应通过 CacheManager 的 store/read view 和 SparseController 编排接入。

## 7. SparseController 语义

`prepare_forward(seqs, is_prefill)`：

- 每步前重置每层 `LayerBatchSparseState`。
- 从 CacheManager 的 `LayerBatchStates` 拷贝默认 `context_lens`、`req_indices`。
- 对需要观察注意力分数的层分配 `attn_score` 张量。

`get_read_view(layer_idx)`：

- 全注意力层、vanilla、snapkv、pyramidkv、quest、streamingllm 等默认返回 dense/full req-to-slots 映射。
- `omnikv` 返回 observation layer 计算出的 active slots。
- `deltakv` sparse layer 会调用 `cache_manager.deltakv_reconstruct(...)`，把被选中的 compressed token 重建到临时 full-KV slots，并返回拼好的虚拟 read view。

`on_layer_end(layer_idx, context)`：

- 对 observation layer 的 `attn_score` 做 query 平均和 head max，得到 token score。
- 将当前观察层选出的 top token index 写到后续目标 sparse layers。
- 对 `deltakv`，写入的是 `active_compressed_indices`，后续层读取时再触发按需重建。

`post_forward(seqs, is_prefill)`：

- prefill 每个 chunk 结束后调用 `on_every_chunk_prefill_end(...)`。
- DeltaKV 会在每个 chunk 后尝试增量压缩，避免长 prefill 过程中 full-KV buffer 膨胀。
- decode 后，如果 DeltaKV recent buffer 溢出，也调用 `deltakv_evict(...)`。

## 8. DeltaKV 在 sparsevllm 中的运行时布局

`DeltaKVCacheManager` 有三类主要存储：

- `full_kv_cache`：full-attention layers 的完整 KV。
- `deltakv_full_kv_cache`：sparse layers 中仍保持全精度的 sink/recent/centers/current chunk/temp reconstructed KV。
- `deltakv_latent_cache`：sparse layers 中被压缩 token 的 latent residual。

关键映射：

- `full_layer_slots_map[row, pos]`：full layers 的逻辑位置到物理 slot。
- `sparse_layer_raw_slots_map[row, pos]`：DeltaKV sparse layers 中仍有 full KV 的位置到 slot；压缩后位置为 -1。
- `sparse_layer_latent_slots_map[row, pos]`：压缩 token 的位置到 latent slot。
- `deltakv_latent_to_full_slots[layer, latent_slot, k]`：每个 latent token 的 top-k father/reference full slots。
- `deltakv_slot_to_pos[slot]`：full-KV slot 对应原序列位置，用于 de-RoPE/re-RoPE。
- `row_deltakv_compressed_lens[row]`：sink 之后已经 finalized/compressed 的历史长度。
- `row_deltakv_center_slots[row][layer]`：每层已积累的参考中心 slots。

DeltaKV prefill/decode 准备：

- `_prepare_prefill` 同时给 full layers 和 deltakv sparse layers 分配 full slots，写入两个 batch state。
- `_prepare_decode` 每个序列分配一个新 full-layer slot 和一个 sparse-layer raw slot。
- 当前新 token/chunk 总是先以 full KV 形式写进去，后续由 eviction 把旧 buffer 压成 latent。

DeltaKV 压缩 `deltakv_evict(...)`：

1. 每个序列分开处理。
2. 保留 `sink` 和最近 `recent` 个 token 为 raw/full。
3. 对 `buffer_len > recent` 的旧区间按 `recent` 的倍数 evict。
4. 在 evicted block 内按 `cluster_step = int(1 / cluster_ratio)` 选择新 centers。
5. 每个 sparse layer 对 evicted block 做 de-RoPE，拼接 `[K_unrope, V]`。
6. 从旧 centers + 新 centers 中用 L2/dot/cosine/fastdot 找 top-k fathers，并求 base mean。
7. 对非 center token 写入 `latent = compress_down(kv_block) - compress_down(base_kv)`。
8. 存 latent、father slots，释放非 center token 的 full-KV slots，并把 raw map 置为 -1。

DeltaKV 重建 `deltakv_reconstruct(...)`：

1. `_deltakv_build_view_and_plan_reconstruct(...)` 构造本层 read view。
2. view 结构是 `sink slots + selected top compressed slots + current raw buffer slots`。
3. 对被选中但 raw slot 已不存在的 token，分配 temp full slots，并记录 `recon_pos / recon_latent / recon_out_slot`。
4. 读取 latent，经 `compress_up` 得到 delta。
5. 读取 father full slots，按 father 位置 de-RoPE K，拼接并求均值。
6. `kv_unrope = delta + father_mean`。
7. 对目标 token 位置 re-RoPE K，写回 temp slots。
8. attention kernel 读取虚拟 active slots；attention 结束后临时 slots 回收。

## 9. HF deltakv 路径

HF wrapper 的核心 cache 在 `src/deltakv/modeling/kv_cache.py`：

- `CompressedKVCache`：较早/基础的 chunk mean residual 路径。
- `ClusterCompressedKVCache`：更贴近论文的全局/步幅参考中心路径，维护 `bases_cache`、`token_father_idx`、`comp_kv_cache` 等。

HF Qwen2 DeltaKV 路径：

- `src/deltakv/modeling/qwen2/qwen2_with_compress_inference.py`
- attention 层有 `compress_down` / `compress_up`。
- `past_key_values.update(...)` 返回当前 attention 可见的 key/value/full_idx，同时在 buffer 超阈值时压缩旧 KV。

HF 与 sparsevllm 的主要区别：

- HF 路径把压缩/重建逻辑嵌在模型/cache 对象里，容易验证算法，适合训练和 baseline 对比。
- sparsevllm 路径把状态放进 CacheManager，把跨层选择放进 SparseController，把 attention 层保持为统一 kernel 消费者，更适合系统级吞吐。

## 10. 用户三张图的对齐

核心组件图：

- `LLM` 外壳里包含 `LLMEngine`。
- `LLMEngine` 里有 `Scheduler` 和 rank 0 `ModelRunner`；TP>1 时其他 rank 是独立 `ModelRunner` 子进程。
- `ModelRunner` 内部组件是 model、sampler、cache_manager、sparse_controller。shared memory 只用于 rank 0 向其他 rank 广播方法调用。

初始化图：

- `LLM(args.model_path, **engine_kwargs)` 进入 `LLMEngine.__init__`。
- Config 初始化、TP 进程启动、rank0 ModelRunner、tokenizer、Scheduler、warmup 的顺序与代码一致。
- `ModelRunner` 初始化顺序是分布式/GPU 环境、模型结构和权重、Sampler、CacheManager、SparseController、DeltaKV compressor、共享内存/RPC。

推理图：

- benchmark 或外部先 `llm.add_request(...)`。
- 主循环 `while not llm.is_finished(): llm.step()`。
- `step()` 中 `scheduler.schedule()` 决定 prefill 或 decode。
- `model_runner.call("run", seqs, is_prefill)` 驱动所有 rank 执行。
- `run()` 中先 `cache_manager.prepare_step` 和 `sparse_controller.prepare_forward`，再模型前向、采样、`sparse_controller.post_forward`。
- 模型前向每层由 `Attention.forward` 负责 KV 写入、读视图构造和注意力 kernel；每层结束后 `SparseController.on_layer_end` 可能更新后续层稀疏视图。

## 11. 后续学习入口

建议按这个顺序继续追代码：

1. `src/sparsevllm/engine/llm_engine.py`：请求进入、step 主循环、warmup。
2. `src/sparsevllm/engine/scheduler.py`：waiting/decoding 队列、chunk prefill、准入/抢占。
3. `src/sparsevllm/engine/model_runner.py`：rank 进程、上下文准备、模型前向、采样、稀疏后处理。
4. `src/sparsevllm/layers/attention.py`：KV 写入和 read view 消费。
5. `src/sparsevllm/engine/sparse_controller.py`：观察层、top-k 传播、DeltaKV 重建触发。
6. `src/sparsevllm/engine/cache_manager/base.py` 与 `standard.py`：先理解 vanilla KV slot 模型。
7. `src/sparsevllm/engine/cache_manager/deltakv.py`：再理解 DeltaKV 三池布局、evict、reconstruct。
8. `src/deltakv/modeling/kv_cache.py`：对照 HF cache 版本理解算法原型。
9. `src/deltakv/get_chat_api.py`：理解 benchmark 如何选择 HF/sparsevllm 后端。
10. `scripts/bench_sparse_vllm.py`：理解吞吐测试口径。

## 12. 重要提醒

- 当前仓库 README 关于 `SamplingParams.temperature` 的旧说法可能过时；代码中 `temperature >= 0.0`，所以 greedy `temperature=0.0` 是有效的。
- sparse method 分发必须从 `CacheManager.create(...)` 走，新增一等方法不要塞进模型文件。
- 方法特定持久状态应放在 `src/sparsevllm/engine/cache_manager/`。
- 跨层编排放在 `src/sparsevllm/engine/sparse_controller.py`。
- `src/sparsevllm/layers/attention.py` 应保持通用，优先通过 `get_layer_store_view(...)`、`get_read_view(...)`、`build_decode_view(...)`、`on_kv_stored(...)` 等接口接入。
