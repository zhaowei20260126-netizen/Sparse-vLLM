![logo.png](assets/logo.png)

![sparse_vllm_throughput.png](assets/sparse_vllm_throughput.png)

<p align="center">
  <a href="https://deepwiki.com/CURRENTF/Sparse-vLLM"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
  <a href="https://arxiv.org/abs/2602.08005">
    <img src="https://img.shields.io/badge/arXiv-2602.08005-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://arxiv.org/pdf/2602.08005.pdf">
    <img src="https://img.shields.io/badge/PDF-download-brightgreen.svg" alt="PDF">
  </a>
</p>

本仓库主要是一个**稀疏优先（sparse-first）的推理引擎**（`sparsevllm`）。同时也包含 DeltaKV 压缩器训练与评估工具（`deltakv`）。

*DeltaKV 的模型检查点与数据集即将全部上传。*

## Sparse-vLLM

Sparse-vLLM（实现位于 `src/sparsevllm/`）是一个以**稀疏性为第一设计原则**构建的推理框架。它不是把稀疏方法叠加在传统 KV 缓存之上，而是从缓存布局、控制器流程和内核实现上重新设计，使多种稀疏机制都能干净地接入。

若要查看代码库结构和按文件导航，请使用本页顶部的 DeepWiki 徽章。

概括来说，Sparse-vLLM 支持：

- **物理驱逐**（例如 SnapKV、PyramidKV）：token 会在物理存储中被真正移除或搬移。
- **逻辑掩码**（例如 OmniKV）：token 仍保留在存储中，但在注意力层面被掩码。
- **混合方案**（DeltaKV）：保留一个小型高精度池，同时将更旧的 token 存入压缩池（可选/实验性）。

后续可以持续加入更多稀疏方法。模块化的 `CacheManager` 设计使得在不重写整个引擎的情况下，也能高效集成新方法。

> 如果你希望 Codex 按本仓库架构新增一种稀疏方法，请使用仓库技能
[`$add-sparse-method`](skills/add-sparse-method/SKILL.md)。该技能编码了新增方法的预期结构
（`cache_manager` 优先、通用 `attention.py`、通过 `build_decode_view(...)` 的解码期钩子，以及方法特定状态不放在 `utils/` 中）。

### 安装

```bash
conda create -n svllm python=3.10 -y
conda activate svllm
pip install torch==2.8.0 transformers[torch]==4.53.3 accelerate deepspeed==0.15.4 torchvision datasets==4.1.0
pip install fire matplotlib seaborn wandb loguru ansible
MAX_JOBS=8 pip install flash-attn --no-build-isolation
pip install -e .
```

### 最小使用示例

```python
from sparsevllm import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen2.5-7B-Instruct-1M",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    chunk_prefill_size=4096,
    vllm_sparse_method="omnikv",
    # OmniKV 参数（简化基线；可按需调参）
    full_attn_layers="0,1,2,4,7,14",  # 运行全注意力的层（必须包含第 0 层）
    num_top_tokens=2096,  # 稀疏层保留的 top-K token
    num_top_tokens_in_prefill=8192,  # prefill 阶段使用的 top-K（默认与 num_top_tokens 相同）
    chunk_prefill_accel_omnikv=False,  # 关闭 OmniKV 的 chunk-prefill 加速，便于比较
)

outputs = llm.generate(
    prompts=["Write a short story about sparse attention."],
    sampling_params=SamplingParams(temperature=0.7, max_tokens=128),
)
print(outputs[0]["text"])
llm.exit()
```

### 关键参数

Sparse-vLLM 的运行参数定义在 `src/sparsevllm/config.py` 中，可以作为关键字参数传入 `LLM(...)`。

**通用参数**

- `tensor_parallel_size`：要启动的 GPU rank（进程）数量。
- `gpu_memory_utilization`：用于 KV 缓存的总 GPU 显存占比。
- `max_model_len`：允许的最大 token 数（提示词 + 生成）。
- `chunk_prefill_size`：长提示词 prefill 的分块大小（降低峰值内存并改善长上下文调度）。
- `max_num_batched_tokens`、`max_num_seqs_in_batch`、`max_decoding_seqs`：调度器吞吐/时延约束。

**稀疏参数（依方法而定）**

- `vllm_sparse_method`：方法字符串（见下文）。
- `num_sink_tokens`：开头始终保留的“sink” token 数。
- `num_recent_tokens`：末尾始终保留的“recent” token 数。
- `num_top_tokens`：保留 top-K token（基于重要性选择；OmniKV 和某些混合模式使用）。
- `num_top_tokens_in_prefill`：prefill 阶段使用的 top-K（未设置时默认等于 `num_top_tokens`）。
- `full_attn_layers`：运行全注意力的层索引（逗号分隔字符串或列表）；OmniKV/DeltaKV 将其作为“观察”锚点层。

### 支持的方法

将 `vllm_sparse_method` 设置为以下之一：

- `""`（原生/全注意力）
- `"streamingllm"` / `"attention-sink"`（固定 sink + recent-window 物理驱逐）
- `"snapkv"`、`"pyramidkv"`（物理驱逐）
- `"omnikv"`（逻辑掩码）
- `"quest"`（解码阶段按查询感知做页选择；prefill 仍为全注意力）
- `"deltakv"` / `"deltakv-*"`（混合压缩；可选/实验性，见 [DeltaKV](#deltakv)）

`quest` 运行参数：

- `quest_chunk_size`：QuEST 的页/块大小（token 数，默认 `16`）
- `quest_token_budget`：页对齐前的解码期 token 预算（默认 `1024`）
- `quest_skip_layers`：解码时前 N 层保持稠密（默认 `2`）

## 如何测试

### 吞吐基准测试

使用 `scripts/bench_sparse_vllm.py` 测量 TTFT、prefill 吞吐、decode 吞吐以及 GPU 显存。

说明：

- 推荐使用 `--hyper_params` 传入 Sparse-vLLM 的 `Config` 参数（JSON 对象）。`--gpu_util/--chunk_size/--tp` 这类参数仅为向后兼容保留。
- `--lengths` 测量的是*提示词长度*；脚本内部会设置 `max_model_len = length + output_len + 100`。

基线（vanilla）：

```bash
python scripts/bench_sparse_vllm.py \
  --model_path <PATH_TO_BASE_MODEL> \
  --lengths 512000 \
  --batch_sizes 2 \
  --methods vanilla \
  --hyper_params '{"gpu_memory_utilization": 0.9}'
```

#### 使用 `sparsevllm` 后端运行 MathBench

这些示例适合快速做 GSM8K / AIME 风格对比，同时直接调用 Sparse-vLLM 引擎。数据集细节见 `benchmark/math_bench/README.md`。

全注意力基线：

```bash
python benchmark/math_bench/pred.py \
  --model qwen7b-full \
  --model_path /root/autodl-fs/models/DeepSeek-R1-Distill-Qwen-7B \
  --tokenizer_path /root/autodl-fs/models/DeepSeek-R1-Distill-Qwen-7B \
  --ws 1 \
  --batch_size 30 \
  --backend sparsevllm \
  --task aime2024 \
  --temperature 0.6 \
  --hyper_param '{"chunk_prefill_size": 4096, "vllm_sparse_method": ""}'
```

OmniKV：

```bash
python benchmark/math_bench/pred.py \
  --model qwen7b-omnikv \
  --model_path /root/autodl-fs/models/DeepSeek-R1-Distill-Qwen-7B \
  --tokenizer_path /root/autodl-fs/models/DeepSeek-R1-Distill-Qwen-7B \
  --ws 1 \
  --batch_size 30 \
  --backend sparsevllm \
  --task aime2024 \
  --temperature 0.6 \
  --hyper_param '{"chunk_prefill_size": 4096, "vllm_sparse_method": "omnikv", "chunk_prefill_accel_omnikv": false, "full_attn_layers": "0,1,2,4,7,14", "num_top_tokens": 1024}'
```

DeltaKV：

```bash
python benchmark/math_bench/pred.py \
  --model qwen7b-deltakv \
  --model_path /root/autodl-fs/models/DeepSeek-R1-Distill-Qwen-7B \
  --tokenizer_path /root/autodl-fs/models/DeepSeek-R1-Distill-Qwen-7B \
  --ws 1 \
  --batch_size 30 \
  --backend sparsevllm \
  --task aime2024 \
  --temperature 0.6 \
  --hyper_param '{"chunk_prefill_size": 512, "num_top_tokens_in_prefill": 16384, "max_num_batched_tokens": 8192, "max_num_seqs_in_batch": 30, "vllm_sparse_method": "deltakv-triton-v4", "chunk_prefill_accel_omnikv": true, "full_attn_layers": "0,1,2,4,7,14", "num_top_tokens": 1024, "deltakv_path": "/root/autodl-fs/checkpoints/compressor/<COMPRESSOR_DIR>", "kv_compressed_size": 256}'
```

当使用 `--backend sparsevllm` 时，方法选择完全通过 `--hyper_param` 完成（`vllm_sparse_method`、`deltakv_path` 等）。Sparse-vLLM 后端不会使用 `--model_cls` 和 `--compressor_path`。

#### 使用 `sparsevllm` 后端运行 LongBench

当你希望得到来自真实 Sparse-vLLM 引擎（而非 HF 包装模型）的 LongBench 结果时，请使用此路径。

```bash
python benchmark/long_bench/pred.py \
  --model qwen7b-omnikv \
  --model_path /root/autodl-fs/models/Qwen2.5-7B-Instruct-1M \
  --tokenizer_path /root/autodl-fs/models/Qwen2.5-7B-Instruct-1M \
  --ws 1 \
  --batch_size 1 \
  --backend sparsevllm \
  --task qasper,hotpotqa,multi_news \
  --hyper_param '{"chunk_prefill_size": 4096, "vllm_sparse_method": "omnikv", "chunk_prefill_accel_omnikv": true, "num_top_tokens_in_prefill": 4096, "num_top_tokens": 2048, "full_attn_layers": "0,1,2,4,7,14", "num_recent_tokens": 128, "num_sink_tokens": 8}'
```

若要运行完整 LongBench，只需省略 `--task`。若要切换到 DeltaKV，保持 `--backend sparsevllm`，并将 `--hyper_param` 中方法相关部分替换为 `vllm_sparse_method="deltakv"`（或 `deltakv-triton-v4`）以及 `deltakv_path=...`。

#### 使用 HF 包装器运行 LongBench

当你希望对比 `src/deltakv/` 下实现的 DeltaKV / SnapKV / PyramidKV 包装模型时，请使用 HF 后端。

```bash
python benchmark/long_bench/pred.py \
  --model qwen7b-deltakv \
  --model_path /root/autodl-fs/models/Qwen2.5-7B-Instruct-1M \
  --tokenizer_path /root/autodl-fs/models/Qwen2.5-7B-Instruct-1M \
  --ws 1 \
  --batch_size 1 \
  --backend hf \
  --model_cls deltakv \
  --compressor_path "/root/autodl-fs/checkpoints/compressor/<COMPRESSOR_DIR>" \
  --hyper_param '{"chunk_prefill_size": 2048000, "num_top_tokens_in_prefill": 4096, "chunk_prefill_accel_omnikv": true, "num_top_tokens": 0.11, "full_attn_layers": "0,1,2,4,7,14", "num_recent_tokens": 128, "num_sink_tokens": 8, "use_compression": true, "use_cluster": true, "cluster_ratio": 0.1}'
```

若要对比其他基线，请保持 `--backend hf` 并切换 `--model_cls` / `--hyper_param`，例如：`omnikv` 搭配 `{"chunk_prefill_size": 4096, "num_top_tokens_in_prefill": 4096, "num_top_tokens": 2048, "full_attn_layers": "0,1,2,4,7,14", "num_recent_tokens": 128, "num_sink_tokens": 8}`；`snapkv` 搭配 `{"num_top_tokens": 0.2, "pool_kernel_size": 7}`；`pyramidkv` 搭配类似 token 预算；Llama/Mistral 检查点上的 `kivi` 可用 `{"k_bits": 4, "v_bits": 4, "group_size": 32, "residual_length": 128}`；`kvzip` 可用 `{"ratio": 0.3, "level": "pair", "kv_type": "evict", "prefill_chunk_size": 16000}`。

对于 `kvzip`，仓库内置的 baseline 位于 `baselines/kvzip/`。请先构建其 CUDA 扩展：

```bash
cd baselines/kvzip/csrc
make
```

## DeltaKV

DeltaKV 是一种**压缩 KV 缓存**的方法，用于实现更高效的 Transformer LLM 长上下文推理。
本仓库包含 DeltaKV 压缩器训练代码及部分推理/基准集成，但 DeltaKV 相关的
速度/质量/性能权衡仍在持续迭代中。

### DeltaKV 推理

将 `vllm_sparse_method` 设置为以下之一：

- `"deltakv"`
- `"deltakv-triton"`、`"deltakv-triton-v2"`、`"deltakv-triton-v3"`、`"deltakv-triton-v4"`
- `"deltakv-triton-v3-offload"` / `"deltakv-triton-v3-cuda-offload"`

进行 DeltaKV 推理时，还需要传入 `deltakv_path="/path/to/trained_compressor_dir_or_file"`。

你可能会用到的 DeltaKV 参数：

- `deltakv_path`：训练后压缩器权重路径（可为包含 `*.safetensors` / `*.pt` / `*.bin` 的目录，或单个文件）。
- `kv_compressed_size`：压缩后 KV 的潜在维度。
- `cluster_ratio`、`cluster_metric`：参考选择/聚类行为（不同方法变体的使用方式可能不同）。
- `deltakv_offload_latent`：将潜在缓存卸载到 CPU（`*-offload` 方法会自动启用）。
- `deltakv_offload_cpu_threads`：offload 模式下 CPU gather 线程数。

### 训练压缩器

主要入口为：

- Python：`python src/deltakv/train_compressor.py ...`
- CLI 脚本（安装后）：`deltakv-train ...`

训练脚本期望输入由 Hugging Face `datasets`（`load_from_disk`）保存的**已分词 + 已打包**数据集。

```bash
python src/deltakv/train_compressor.py \
  --model_name_or_path <PATH_TO_BASE_MODEL> \
  --dataset_path <PATH_TO_DATASET_ON_DISK> \
  --output_dir <PATH_TO_OUTPUT_CHECKPOINT_DIR> \
  --kv_compressed_size 512 \
  --seq_chunk_size 4 \
  --layer_chunk_size 1 \
  --batch_size 1 \
  --warmup_ratio 0.02 \
  --max_steps 20000 \
  --learning_rate 2e-4 \
  --use_nonlinear_compressor True \
  --ref_mode avg \
  --collect_kv_before_rope True \
  --model_type cluster_e2e \
  --cluster_soft_assignment False \
  --compressor_down_type mlp_swiglu \
  --compressor_down_intermediate_size 3072 \
  --compressor_up_type linear \
  --compressor_linear_bias False
```

常用参数：

- `--kv_compressed_size`：压缩后 KV 长度（越小表示压缩越强）。
- `--model_type`：`e2e`、`cluster_e2e`、`cluster_e2e_big`（见 `src/deltakv/train_compressor.py`）。
- `--collect_kv_before_rope`：是否在 RoPE 之前收集 KV（与模型相关）。

### 在 LongBench 上评估

`benchmark/long_bench/pred.py` 会运行 LongBench 预测，并将 JSONL 输出写入本地输出目录。

```bash
python benchmark/long_bench/pred.py \
  --model all \
  --model_path <PATH_TO_BASE_MODEL> \
  --tokenizer_path <PATH_TO_TOKENIZER_OR_MODEL> \
  --ws 1 \
  --batch_size 1 \
  --backend hf \
  --model_cls deltakv \
  --compressor_path "<PATH_TO_TRAINED_COMPRESSOR_DIR>" \
  --hyper_param '{"chunk_prefill_size": 2048000, "num_top_tokens_in_prefill": 4096,
  "chunk_prefill_accel_omnikv": true, "num_top_tokens": 0.17, "full_attn_layers": "0,1,2,8,18",
  "num_recent_tokens": 128, "num_sink_tokens": 8, "use_compression": true, "use_cluster": true, "cluster_ratio": 0.1}'
```

说明：

- `--backend` 支持 `hf` 和 `sparsevllm`（见 `benchmark/long_bench/pred.py`）。
- `--hyper_param` 接受 JSON 字符串或 JSON 文件路径。
- `full_attn_layers` 以逗号分隔的层索引字符串传入（示例：`"0,1,2,8,18"`）。

### DeltaKV 检查点

- `deltakv_path` 可指向目录（加载器会先扫描 `*.safetensors`，再扫描 `*.bin` / `*.pt`）或单个检查点文件。
- 分离式 KV 检查点（`k_compress_*` / `v_compress_*`）目前不受 Sparse-vLLM 加载器支持。

### CUDA gather 扩展（仅用于 `*-cuda-offload`）

CUDA 扩展位于 `src/sparsevllm/cuda_kernel/`，仅 `deltakv-triton-v3-cuda-offload` 需要。

```bash
cd src/sparsevllm/cuda_kernel
pip install -e .
```

## 故障排查

### `SamplingParams` 不允许 greedy 解码

`SamplingParams.temperature` 必须 `> 1e-10`（见 `src/sparsevllm/sampling_params.py`）。若想要“近似 greedy”，可使用极小温度（如 `1e-5`）。

### `Mixed long/short batch detected`

Sparse-vLLM 强制每一步只能运行“长文本”批次或“短文本”批次之一，不能混合，以保持内核实现更简单。
如果你遇到此错误，通常表示你绕过了调度器的分离逻辑，或在自定义循环中混合了长度差异很大的请求。

### `Insufficient KV cache slots to admit prompt`

这表示在当前方法和 KV 预算下，引擎无法分配足够的 KV 槽来放置该提示词（或其某个分块）。
可尝试以下一项或多项：

- 增加 `gpu_memory_utilization`。
- 减小 `max_model_len`、`batch_sizes` 或提示词长度。
- 对长上下文方法，减小 `num_recent_tokens` / `num_top_tokens` / `num_sink_tokens`。

## 致谢

本项目受到以下工作的启发和/或参考了其中的思想与实现技术：

- `LightLLM`（`ModelTC/LightLLM`）
- `ShadowKV`（`ByteDance-Seed/ShadowKV`）
- `nano-vllm`（`GeeeekExplorer/nano-vllm`）


# 引用
```text
@article{hao2026deltakv,
  title={DeltaKV: Residual-Based KV Cache Compression via Long-Range Similarity},
  author={Hao, Jitai and Huang, Qiang and Wang, Yaowei and Zhang, Min and Yu, Jun},
  journal={arXiv preprint arXiv:2602.08005},
  year={2026}
}

@inproceedings{hao2025omnikv,
  title={Omnikv: Dynamic context selection for efficient long-context llms},
  author={Hao, Jitai and Zhu, Yuke and Wang, Tian and Yu, Jun and Xin, Xin and Zheng, Bo and Ren, Zhaochun and Guo, Sheng},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
