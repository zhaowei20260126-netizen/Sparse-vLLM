# AttnPredict-Offload 新会话交接 Prompt

本文档供一个全新的 Codex 会话阅读，用来快速建立和当前会话相同的上下文背景。新会话开始后，请先完整阅读本文档，再按“必须先读文件”列表补充代码和报告上下文。

## 0. 新会话角色与回答要求

你是 Sparse-vLLM 项目里的 Codex coding agent。请默认用中文回答。  

涉及英文工程术语、论文术语、缩写或代码变量名时，第一次出现请使用：

```text
英文原词/变量名 + 中文解释 + 一句话含义
```

例如：

```text
packed view（打包读视图，即把每条请求实际要读的 GPU slot 压成二维表）：它是 decode attention kernel 的输入。
```

不要假设用户已经理解 Sparse-vLLM、KV cache、AttentionPredictor、offload、CUDA、kernel、prefetch、hot tokens、packed view、lease、score buffer、stream、event、dirty、residency 等概念。解释时用短句和例子，客观回答，不要附和。

## 1. 当前项目路径与环境

项目路径：

```text
/root/autodl-tmp/Sparse-vLLM
```

必须使用虚拟环境：

```bash
.venv/bin/python ...
```

benchmark 脚本调用 python 时，命令前必须加：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH
```

不要使用系统 Python，不要重新下载依赖，`.venv` 中依赖已齐全。

当前关键模型路径：

```text
MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct
ATTNPREDICT_MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth
```

## 2. 必须遵循的 skill

### 2.1 `$add-sparse-method`

文件：

```text
skills/add-sparse-method/SKILL.md
```

核心要求：

- 保持 cache-manager-first 架构。
- `attnpredict-offload` 的核心状态必须放在：

```text
src/sparsevllm/engine/cache_manager/attnpredict_offload.py
```

- 不要把方法特有逻辑塞进：

```text
src/sparsevllm/layers/attention.py
```

- `attention.py` 只允许调用通用 hook 或共享 kernel，不应出现大量 `attnpredict-offload` 专用分支。
- 跨层调度、score 收集、预测触发等可放在：

```text
src/sparsevllm/engine/sparse_controller.py
```

### 2.2 `python-code-slim`

文件：

```text
skills/python-code-slim/SKILL.md
```

核心要求：

- 只保留行为必要、性能有证据的代码。
- 不新增无必要配置项。
- 不保留无收益实验代码。
- 不添加理论上不会触发的复杂兜底。
- 不把 hot path 张量操作改成 Python loop。
- 注释用中文短注释，只解释方法作用、关键变量、shape、stream、lease、residency 等关键约束。
- 不要为了“看起来完整”保留同步/异步两套路径、废弃 fallback、临时调试分支。

### 2.3 `write-optimization-tech-doc`

文件：

```text
skills/write-optimization-tech-doc/SKILL.md
```

用于写优化技术文档。要求：

- 不编造实验数字。
- 有独立实验就写具体指标。
- 没有独立实验就明确写“暂未单独测量”。
- 不要用内部 profiler range 名称替代读者能懂的指标。
- 不要写“估算累计减少”这类推导结果，除非用户明确要求。

## 3. 新会话必须先读的文件

请先读这些文件建立背景：

```text
README.md
AGENTS.md
skills/add-sparse-method/SKILL.md
skills/python-code-slim/SKILL.md
skills/write-optimization-tech-doc/SKILL.md
attnpredict-offload资料/推理流程与attention调用时序.md
attnpredict-offload资料/2026-05-28_AttnPredictOffload会话上下文.md
attnpredict-offload资料/2026-05-30_nsys最终timeline与优化结论.md
attnpredict-offload资料/优化技术方案.md
attnpredict-offload资料/2026-06-07_优化技术方案_实验结果版.md
src/sparsevllm/engine/cache_manager/attnpredict_offload.py
src/sparsevllm/engine/cache_manager/attnpredict.py
src/sparsevllm/engine/cache_manager/attnpredict_cnn.py
src/sparsevllm/engine/sparse_controller.py
src/sparsevllm/layers/attention.py
scripts/bench_attnpredict_vs_vanilla_128k.sh
scripts/kernel_bench/bench_attnpredict_block_pool.py
```

## 4. 当前工作区状态提醒

当前工作区有 dirty 文件，不能随便回退：

```text
.gitignore
.vscode/launch.json
.vscode/settings.json
AGENTS.md 可能已有用户修改
attnpredict-offload资料/*.md
attnpredict-offload资料/图片/
hfd.sh
profiler_outputs/*
```

尤其不要为了“干净”执行：

```bash
git reset --hard
git checkout -- <file>
```

除非用户明确要求。

`src/sparsevllm/engine/cache_manager/attnpredict_offload.py` 当前应已恢复：

```python
self._layer_reuse_stride = 4
```

最近一次临时实验把它改成 `1`，但已恢复并通过：

```bash
.venv/bin/python -m py_compile src/sparsevllm/engine/cache_manager/attnpredict_offload.py
```

该文件当前仍可能有一处注释 diff：把“每层一个 CUDA stream”改成“单条 CUDA stream 避免多组 predictor/H2D 并发抢占主计算流”。这是合理注释改动，不要回退。

## 5. 当前 attnpredict-offload 稳定配置

当前最终稳定配置：

```text
attnpredict_reuse_steps = 16
attnpredict_max_stale_steps = 16
layer_reuse_stride = 4
num_top_tokens = 4096
num_sink_tokens = 64
num_recent_tokens = 512
attnpredict_history_steps = 64
attnpredict_pooling_block_size = 16
attnpredict_offload_cpu_threads = 8
attnpredict_offload_pin_staging = true
```

关键含义：

- `reuse_steps`（跨步复用步数，即同一份预测结果连续复用多少个 decode step）：减少时间维度的 predictor 刷新频率。
- `max_stale_steps`（最大陈旧步数，即后台预测没完成时旧预测最多继续用多久）：限制旧 lease 继续使用的时间。
- `layer_reuse_stride`（跨层复用跨度，即每隔多少层重新预测一次）：当前每 4 层共享一个 source layer 的预测结果。
- `lease`（预测驻留集合，即当前一段 decode 可复用的 hot positions 和起始位置）：不是单纯“步数”，它包含 `lease_hot_positions` 和 `lease_start_positions` 两类信息。
- `dirty KV`（脏 KV，即 GPU 有但 CPU backing 还没保存的新 token KV）：驱逐前必须写回 CPU backing。
- `residency`（驻留状态，即哪些 token 当前在 GPU active pool）：用于判断哪些 token 要加载、保留或释放。

## 6. 当前核心结果

### 6.1 目标 benchmark：`128k / bs=2 / output_len=64`

稳定结果在 22 vCPU 机器上：

| method | TTFT | PreTP | DecTP | ITL | Mem | Decode speedup |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | 约 42.6s | 约 6000 tok/s | 约 46.6-46.8 tok/s | 约 42.8ms | 约 66.5GB | 1.00x |
| attnpredict-offload | 约 51.4s | 约 4979 tok/s | 约 52.3 tok/s | 约 38.2ms | 约 48.6GB | 约 1.12x |

注意：显存数字不能当作严格公平优势，因为 vanilla/SnapKV/OmniKV 会按 `gpu_memory_utilization` 预分配更多 KV 空间；attnpredict-offload 的显存更低主要来自 active pool + CPU backing 的实现方式。

目标命令模板：

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
ATTNPREDICT_MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth \
LENGTHS=128000 \
BATCH_SIZES=2 \
OUTPUT_LEN=64 \
GPU_MEMORY_UTILIZATION=0.7 \
ATTNPREDICT_REUSE_STEPS=16 \
ATTNPREDICT_MAX_STALE_STEPS=16 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

### 6.2 跨步/跨层复用消融

当前代码消融结果：

| 配置 | DecTP | 结论 |
|---|---:|---|
| vanilla | 46.62-46.73 tok/s | full attention 基线 |
| 不做跨步/跨层复用：`layer_reuse_stride=1 + reuse/max_stale=1/1` | 10.32 tok/s | 每层每步都预测，极慢 |
| 只做跨步复用：`layer_reuse_stride=1 + reuse/max_stale=16/16` | 37.03 tok/s | 每层都预测，仍低于 vanilla |
| 跨步 + 跨层复用：`layer_reuse_stride=4 + reuse/max_stale=16/16` | 约 52.33 tok/s | 超过 vanilla |

日志：

```text
run_stride1_reuse1_ablation_bs2_128k_20260607.log
run_stride1_ablation_r16_s16_bs2_128k_20260607.log
```

结论：当前最终性能主要来自减少 predictor 刷新频率。先用跨步复用减少时间维度刷新次数，再用跨层复用减少层维度刷新次数。

### 6.3 reuse/stale 调参结果

这些结果当时都默认开启 `layer_reuse_stride=4`：

| 配置 | DecTP |
|---|---:|
| `stride=4, reuse=4 / stale=6` | 约 24.45 tok/s |
| `stride=4, reuse=8 / stale=8` | 44.65 tok/s |
| `stride=4, reuse=12 / stale=12` | 47.92 tok/s |
| `stride=4, reuse=16 / stale=16` | 50.50 tok/s |
| 最终同轮 benchmark | 51.21 tok/s |
| 当前同机最新结果 | 52.33 tok/s |

重要说明：`reuse=4 / stale=6` 是“语义正确初始版”或“保守刷新配置”，不是上周错误的 `reuse=10000` 作弊结果。

### 6.4 bs=4 多方法对比

实验配置：`128k / bs=4 / output_len=64`

| method | TTFT | DecTP | AvgBS | Mem | Speedup |
|---|---:|---:|---:|---:|---:|
| vanilla | 64.94s | 45.7 tok/s | 2.0 | 66.87GB | 1.00x |
| SnapKV | 65.81s | 135.2 tok/s | 3.9 | 67.46GB | 2.96x |
| OmniKV | 65.46s | 73.8 tok/s | 2.0 | 66.87GB | 1.61x |
| AP-offload | 78.73s | 73.6 tok/s | 3.9 | 68.67GB | 1.61x |

结论：

- AP-offload 在该组结果中超过 vanilla。
- AP-offload 与 OmniKV 基本持平。
- AP-offload 没有超过 SnapKV。
- `bs=4` 下存在排队和 `AvgBS` 不一致，严格公平性要继续复核。

## 7. 已完成优化点与证据

### 7.1 跨步复用

核心：同一份 predictor 结果连续复用多个 decode step，只在达到刷新步数后重新收集 score 和预测。

证据：`reuse=4/8/12/16` 在 `stride=4` 下 DecTP 从约 `24.45` 提升到 `50.50+ tok/s`。

质量风险：预测新鲜度下降。小样本英文 LongBench 未发现输出变化，但完整质量仍有风险。

### 7.2 跨层复用 `layer_reuse_stride=4`

核心：每 4 层只让复用组首层运行 predictor，其余层复用同一组 hot tokens 和 lease。

证据：当前代码下：

```text
stride=1 + reuse=16 -> 37.03 tok/s
stride=4 + reuse=16 -> 约 52.33 tok/s
```

说明跨层复用是当前超过 vanilla 的关键组成。

### 7.3 CNN predictor 编译

核心：用 `torch.compile(self.cnn, dynamic=True, options={"triton.cudagraphs": False})` 编译 CNN predictor，并在初始化时用 dummy 输入预热。

离线结果：

| 方案 | 耗时 |
|---|---:|
| eager CNN | 约 8.88ms |
| CNN-only CUDA Graph replay | 约 8.63ms |
| `torch.compile` CNN | 约 3.62ms |
| `torch.compile`，关闭 Inductor CUDA Graph | 约 3.67ms |

端到端结果：

```text
compiled CNN 后 AP-offload DecTP 约 34.40 tok/s
此前同配置约 26.23 tok/s
```

结论：采纳。不改变 predictor 结构和输出语义，只减少 predictor 执行成本。

### 7.4 build_decode_view 的 packed view 固定成本优化

核心：缓存 lease 内稳定的 sink/hot slots，recent 部分动态补齐，并复用 packed slots/local req indices buffer。

实验配置：`128k / bs=2 / output_len=16`

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| build_decode_view 平均耗时 | 0.1729 ms/layer | 0.1614 ms/layer |
| DecTP | 50.49 tok/s | 52.53 tok/s |

结论：采纳。不改变 `sink + hot + recent/current` 集合，只改变 view 构造方式。

### 7.5 CPU backing 连续段写入优化

核心：prefill 阶段如果 CPU slots 是连续递增区间，用连续切片 `copy_` 替代 `index_copy_`。

实验配置：`128k / bs=2 / output_len=16`

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| CPU backing 写入平均耗时 | 47.67 ms/次 | 47.05 ms/次 |
| 单次调用平均减少 | - | 0.62 ms/次 |
| TTFT | 53.20s | 52.56s |
| PreTP | 4812.18 tok/s | 4870.42 tok/s |

结论：采纳。它主要改善 prefill/TTFT，不直接减少单个 decode step 时间。

### 7.6 dirty KV 延迟写回与 residency 切换

核心：decode 新生成 KV 先标记 dirty 并留在 GPU，只有即将被驱逐时才写回 CPU backing；普通复用步不做完整 residency diff，lease 切换时再处理。

反向实验：

| 实验 | 结果 |
|---|---:|
| prefill KV 完全延迟写回 | DecTP 降到 5.08 tok/s |
| 当前 decode dirty 延迟写回 | 已采纳 |

注意：

- dirty 主要作用于 decode 新生成 token。
- 新 token 一般在 recent window 中，通常短期不会被驱逐。
- 但若 GPU active pool 紧张、连续 batching、recent 滑窗后移或 lease 切换，dirty token 可能需要驱逐，驱逐前必须 D2H 写回 CPU backing。

## 8. Nsight / profiler 关键结论

### 8.1 当前 overlap 实现

文件：

```text
profiler_outputs/nsys_overlap_attnpredict_r16_s16_bs2_128k_o64.nsys-rep
profiler_outputs/nsys_overlap_attnpredict_r16_s16_bs2_128k_o64.sqlite
profiler_outputs/nsys_overlap_attnpredict_r16_s16_bs2_128k_o64_stats.txt
```

trace 下吞吐：

```text
TTFT 51.59s
DecTP 45.70 tok/s
```

Nsight 会扰动吞吐，该结果只用于 timeline 分析，不作为最终性能。

项目 profiler 摘要：

| 区间 | 调用次数 | 总耗时 |
|---|---:|---:|
| `model_run_model_decode` | 63 | 2.634s |
| `attnpredict_offload_predict_cnn_stream` | 24 | 0.358s |
| `attnpredict_offload_h2d_prefetch_stream` | 68 | 0.0357s |
| `attnpredict_offload_prefetch_wait` | 32 | 0.0021s |

decode step 分类：

| step 类型 | 数量 | 平均 wall time |
|---|---:|---:|
| 普通复用步，无 predictor overlap | 56 | 34.497ms |
| predictor overlap 步 | 3 | 143.394ms |
| 切换新 lease 后清理步 | 3 | 63.622ms |
| 第一个 decode step | 1 | 80.672ms |

结论：

- predictor 通常能在下一次消费前完成。
- `prefetch_wait` 很小，说明不是“下一步等预测完成”主导。
- 真正问题是 predictor 刷新步会拉长当前 decode step。
- 第一个 decode step 和 lease cleanup step 较慢，是因为第一次消费 prefill lease、构造 static view cache、分配 current slots、后续 lease cleanup/residency diff 等固定成本。

图片：

```text
attnpredict-offload资料/图片/attnpredict_offload_decode_stream_overlap_overview.png
attnpredict-offload资料/图片/nsys_overlap_decode_timeline.png
```

### 8.2 no-overlap 对照

文件：

```text
profiler_outputs/nsys_nooverlap_step_boundary_r16_s16_bs2_128k_o64_v2.nsys-rep
profiler_outputs/nsys_nooverlap_step_boundary_r16_s16_bs2_128k_o64_v2.sqlite
profiler_outputs/nsys_nooverlap_step_boundary_r16_s16_bs2_128k_o64_v2_stats.txt
```

实验含义：

`no-overlap`（不重合，即 predictor 不插入当前主计算 step，而是在主计算结束后执行）：用于验证 predictor 是否拖慢主计算流。

结果：

| 模式 | TTFT | DecTP | `model_run_model_decode` 总耗时 | `predict_cnn_stream` 总耗时 | `prefetch_wait` 总耗时 |
|---|---:|---:|---:|---:|---:|
| overlap 当前实现 | 51.59s | 45.70 tok/s | 2.634s | 0.358s | 0.002s |
| no-overlap step 末尾预测 | 65.17s | 43.81 tok/s | 2.358s | 2.354s | 0.323s |

刷新步拆分：

| 模式 | step | 主计算耗时 | predictor 耗时 | step 总观察时间 |
|---|---:|---:|---:|---:|
| overlap | 16 | 182.865ms | 175.667ms | 182.865ms |
| no-overlap | 16 | 41.437ms | 171.227ms | 215.202ms |
| overlap | 32 | 121.877ms | 117.479ms | 121.877ms |
| no-overlap | 32 | 37.037ms | 90.912ms | 129.567ms |
| overlap | 48 | 125.441ms | 120.423ms | 125.441ms |
| no-overlap | 48 | 38.886ms | 90.367ms | 131.041ms |

结论：

- no-overlap 证明 predictor 会干扰主计算：主计算本身从 `121-183ms` 降到约 `37-41ms`。
- 但 predictor 不能被隐藏后，总吞吐下降。
- 所以不采纳 no-overlap；后续应降低 predictor 自身成本，而不是完全取消 overlap。

图片：

```text
attnpredict-offload资料/图片/nsys_no_overlap_decode_timeline.png
```

### 8.3 GPU 计算争夺 vs 带宽争夺

Nsight Systems MemOps 统计：

| 类型 | 次数 | 总时间 | 数据量 |
|---|---:|---:|---:|
| Device-to-Host | 5670 | 15.24s | 约 32.95GiB |
| Host-to-Device | 5492 | 7.49s | 表中未直接列总量 |
| Device-to-Device | 9946 | 0.009s | 很小 |

正常 profiler `128k / bs=2 / output_len=64`：

| 区间 | 调用次数 | 总耗时 |
|---|---:|---:|
| `attnpredict_offload_build_decode_view` | 2016 | 0.348s |
| `attnpredict_offload_predict_cnn_stream` | 24 | 0.340s |
| `attnpredict_offload_topk` | 48 | 0.173s |
| `attnpredict_offload_cnn` | 48 | 0.075s |
| `attnpredict_offload_h2d_prefetch_stream` | 70 | 0.042s |
| `attnpredict_offload_softmax` | 48 | 0.026s |
| `attnpredict_offload_scatter_reduce` | 48 | 0.026s |
| `attnpredict_offload_prefetch_wait` | 32 | 0.0017s |

结论：

- prefill/TTFT 慢主要是 CPU backing/D2H 数据路径问题。
- decode 侧主要是 predictor GPU 计算、top-k/softmax/pooling 和 packed view 构造等固定成本。
- H2D 预取等待不是 decode 主导瓶颈。
- pin staging true/false 对 DecTP 影响在波动范围内，不能证明 H2D 是 decode 主瓶颈。

Nsight Compute / ncu 曾尝试 roofline，但受权限限制：

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

所以没有 SM 利用率、DRAM/L2 throughput 等硬件计数器结论。

## 9. 质量评估结论

### 9.1 小样本 reuse 质量 A/B

LongBench 英文小样本：

| 数据集 | 样本数 | reuse=4 | reuse=12 | reuse=16 | reuse=16 与 reuse=4 完全相同输出 |
|---|---:|---:|---:|---:|---:|
| `hotpotqa` | 20 | F1 45.27 | F1 45.27 | F1 45.27 | 20/20 |
| `passage_retrieval_en` | 20 | retrieval 100.00 | retrieval 100.00 | retrieval 100.00 | 20/20 |

含义：

- `hotpotqa` 是多跳问答任务。
- `passage_retrieval_en` 是段落检索任务。
- 这只是小样本初筛，不能当作完整质量证明。

### 9.2 完整/扩展 LongBench 质量

当前 `reuse=16 / stale=16 / stride=4` 质量结果：

| dataset | metric | n | vanilla | attnpredict r16 | delta | 相同输出 |
|---|---|---:|---:|---:|---:|---:|
| `passage_retrieval_en` | retrieval | 200 | 99.50 | 99.50 | 0.00 | 199/200 |
| `qasper` | QA F1 | 50 | 45.37 | 43.11 | -2.26 | 39/50 |
| `narrativeqa` | QA F1 | 50 | 27.80 | 28.40 | +0.60 | 30/50 |
| `multifieldqa_en` | QA F1 | 50 | 58.25 | 56.90 | -1.35 | 35/50 |
| `gov_report` | ROUGE-L | 20 | 23.27 | 21.89 | -1.38 | 0/20 |
| `multi_news` | ROUGE-L | 20 | 14.80 | 15.24 | +0.44 | 1/20 |
| `qmsum` | ROUGE-L | 20 | 17.86 | 17.07 | -0.79 | 1/20 |

调小跨步复用诊断：

| dataset | vanilla | r16 | r4 | r16 delta | r4 delta |
|---|---:|---:|---:|---:|---:|
| `qasper` | 45.37 | 43.11 | 44.33 | -2.26 | -1.04 |
| `multifieldqa_en` | 58.25 | 56.90 | 57.74 | -1.35 | -0.51 |
| `gov_report` | 23.27 | 21.89 | 21.58 | -1.38 | -1.69 |
| `qmsum` | 17.86 | 17.07 | 18.53 | -0.79 | +0.67 |

结论：

- 检索任务基本不退化。
- QA 和摘要均值没有崩，但存在样本级明显波动。
- 调小 `reuse_steps` 对部分任务有改善，说明质量风险和 stale 有关。
- 当前版本可称为“性能可用，但质量仍需 calibration 或更细粒度复用策略加固”。

## 10. 遇到的重要坑

### 10.1 上周错误结果：`reuse_step` 没传进去

上周有一版结果很好，但后来发现 benchmark 配置有问题：设置的 `reuse_step=4` 没有真正传入，代码仍使用 `10000`，导致 predictor 几乎不刷新，所以开销很小，结果不能作为有效加速。

组会解释方式：

```text
上周的 AP-offload 加速结果来自配置传参问题，predictor 实际几乎不刷新，所以不能作为有效结论。本周修正后重新从语义正确版本开始做消融和优化。
```

### 10.2 22 vCPU 与 25 vCPU 机器结果差异

当前 22 vCPU 机器稳定结果：

```text
attnpredict-offload r16/s16 DecTP 约 52.3 tok/s
vanilla 约 46.6-46.8 tok/s
```

另一台同 GPU、25 vCPU 机器，用户用相同代码和命令跑出：

| method | TTFT | DecTP | Mem |
|---|---:|---:|---:|
| vanilla | 42.71s | 46.17 tok/s | 66.46GB |
| attnpredict-offload | 52.16s | 37.28 tok/s | 49.23GB |

现象：只知道硬件页面显示 CPU 从 22 vCPU 变为 25 vCPU，GPU 都是 RTX PRO 6000 96GB。原因尚未最终确认。

最可信解释方向：

- 云实例 vCPU 数不同可能代表底层 NUMA/CPU 频率/PCIe 拓扑/共享负载不同。
- offload 对 CPU 调度、D2H/H2D、pinned staging、后台线程与主线程 launch 抢占更敏感。
- vanilla 更依赖 GPU 主计算，所以跨机器波动较小。

当前处理：先不把 25 vCPU 结果作为否定当前方案的结论；记录为跨机器敏感性待办，后续需用固定 profiler 和系统信息复核。

### 10.3 显存对比不公平

用户指出：vanilla/SnapKV/OmniKV 等方法会按 `gpu_memory_utilization` 分配剩余可用 KV 空间，即使用不到也会占；attnpredict-offload 只创建 `bs * max_len` 附近的 active pool 和 CPU backing，所以显存显示更低。

结论：显存数字不能直接说 attnpredict-offload “节省了那么多显存”作为公平优势，只能说其当前实现分配方式不同。

### 10.4 `score buffer 数量减少` 的表达不准确

单独“score buffer 数量减少”不是主要成本。更准确说法：

```text
需要写出 attention score 的层数减少；
CNN forward 和 top-k 等 predictor 后处理次数减少；
prefetch worker 任务数量减少。
```

`score buffer` 本身不是大头，真正贵的是 with-score attention kernel 额外写 score、后续 softmax/pooling/CNN/top-k/prefetch 这一整套 predictor 刷新路径。

### 10.5 step 16 比 step 32/48 慢

图中 step 16 是第一次真正的 decode 预测刷新步，叠加了 predictor 路径冷启动、第一次 lease 切换、residency/cache 初始化等成本。

group1 明显比 group2-8 慢，较可能是首个 source layer group 承担了这批首轮初始化成本。不要把它解释成每个 group 固定计算都这么慢。

图中 `182.9ms` 是整个 batch 的一个 decode step 的模型前向墙钟时间，不是单个 seq，也不是单独 attention 时间。

## 11. 已验证无效或回退的方向

不要重复投入，除非有新的证据：

| 方向 | 结果 | 回退理由 |
|---|---:|---|
| logits -> block pooled attention fused runtime kernel | 约 19.97 tok/s 或更低 | 行为正确但比 PyTorch 路径慢 |
| compact pooling 接入 runtime | 12.00 tok/s | CPU/GPU 构造成本吃掉收益 |
| 同层 batch CNN | 24.25 tok/s | 低于稳定版 |
| CPU CNN predictor | 0.43 tok/s | CPU Conv2d 太慢 |
| recent/current GPU append kernel | 20.86 / 21.83 tok/s | 新增 kernel launch 不划算 |
| full-resident fast path | 19.17 tok/s 或 34.55 tok/s | free-list/结束释放成本抵消收益 |
| decode attention BLOCK_SEQ=512/128 | 20.66 / 20.52 tok/s | 低于默认 256 |
| source layer 组级独立 stream | 51.80 tok/s | 低于单 stream 52.33 tok/s |
| no-overlap step-boundary predictor | 43.81 tok/s | 主计算变短但 predictor 变成显式等待 |
| `max_stale=64` 放宽等待 | 52.34 vs 52.25 tok/s | 基本无收益，质量风险更高 |
| `reuse=10, max_stale=4` | 47.31 tok/s | 仅略高于 vanilla，低于 r16/s16 |

## 12. 当前待做清单

### 12.1 质量方向：calibration

原 AttentionPredictor 论文/代码中有 `calibration_step=5` 的想法，即周期性用真实 attention 分数校正 history，避免预测漂移。当前 `attnpredict-offload` 没有完整 calibration。

下一步建议：

1. 先实现诊断版 calibration：每 N 步 source layer 用 full view 收集一次 score 更新 `attn_history`，但先不改变输出 attention。
2. 测 LongBench QA/摘要质量是否恢复。
3. 再测 DecTP 损失。
4. 如果质量无收益，不应默认开启。

注意：保持 cache-manager-first，不要把 calibration 专用逻辑塞进 `attention.py`。

### 12.2 质量-性能曲线

需要系统测试：

```text
reuse_steps = 4 / 8 / 12 / 16
layer_reuse_stride = 1 / 2 / 4
calibration = off / every N steps
```

目标：找到 QA/摘要质量与 DecTP 的折中，而不是只看吞吐。

### 12.3 跨机器敏感性

需要在 22 vCPU 和 25 vCPU 机器上统一采集：

```bash
lscpu
nvidia-smi -q
nvidia-smi topo -m
numactl --hardware
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name())
PY
```

并分别跑同一 benchmark + profiler，比较：

- TTFT / PreTP / DecTP
- CPU backing 写入时间
- predictor stream 时间
- H2D/D2H 时间
- `prefetch_wait`
- CPU thread 数对性能的影响

### 12.4 profiler 继续定位

重点不是继续盲目融合 kernel，而是继续拆：

- predictor 刷新步中 group1 为什么特别慢；
- first decode step 为什么慢；
- lease cleanup step 的 residency diff / dirty writeback / static cache rebuild 占比；
- `build_decode_view()` 是否还有可合并的 Python 调度成本。

### 12.5 文档与 PPT

当前组会版文档：

```text
attnpredict-offload资料/2026-06-07_优化技术方案_实验结果版.md
```

里面只保留有实验结果的优化点，编号已经重新连续：

```text
优化1：跨步复用与 max stale 控制
优化2：跨层复用 stride=4
优化3：build_decode_view 的 packed view 固定成本优化
优化4：CPU backing 连续段写入优化
优化5：dirty KV 延迟写回与 residency 切换
```

PPT 可用图片：

```text
attnpredict-offload资料/图片/attnpredict_offload_decode_stream_overlap_overview.png
attnpredict-offload资料/图片/nsys_overlap_decode_timeline.png
attnpredict-offload资料/图片/nsys_no_overlap_decode_timeline.png
```

## 13. 常用命令

### 13.1 语法检查

```bash
.venv/bin/python -m py_compile src/sparsevllm/engine/cache_manager/attnpredict_offload.py
```

如果改了 controller/attention，也要编译：

```bash
.venv/bin/python -m py_compile \
  src/sparsevllm/engine/cache_manager/attnpredict_offload.py \
  src/sparsevllm/engine/sparse_controller.py \
  src/sparsevllm/layers/attention.py
```

### 13.2 目标 benchmark

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
ATTNPREDICT_MODEL_PATH=/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth \
LENGTHS=128000 \
BATCH_SIZES=2 \
OUTPUT_LEN=64 \
GPU_MEMORY_UTILIZATION=0.7 \
ATTNPREDICT_REUSE_STEPS=16 \
ATTNPREDICT_MAX_STALE_STEPS=16 \
bash scripts/bench_attnpredict_vs_vanilla_128k.sh
```

### 13.3 Nsight Systems overlap trace

```bash
PATH=/root/autodl-tmp/Sparse-vLLM/.venv/bin:$PATH \
PROFILER_SVLLM=1 CUDA_SYNC_SVLLM=0 \
/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --sample=none \
  --force-overwrite=true \
  -o profiler_outputs/nsys_overlap_attnpredict_r16_s16_bs2_128k_o64 \
  .venv/bin/python scripts/bench_sparse_vllm.py \
    --model_path /root/autodl-tmp/Sparse-vLLM/models/llama-3.1-8B-Instruct \
    --methods attnpredict-offload \
    --lengths 128000 \
    --batch_sizes 2 \
    --output_len 64 \
    --hyper_params '{"gpu_memory_utilization":0.7,"chunk_prefill_size":4096,"tensor_parallel_size":1,"attnpredict_model_path":"/root/autodl-tmp/Sparse-vLLM/predictor/CNN_llama3.1_alltask_5case/best_model.pth","attnpredict_history_steps":64,"attnpredict_pooling_block_size":16,"attnpredict_reuse_steps":16,"attnpredict_max_stale_steps":16,"attnpredict_offload_cpu_threads":8,"attnpredict_offload_cpu_slots":-1,"attnpredict_offload_cpu_memory_utilization":0.7,"attnpredict_offload_pin_staging":true,"num_top_tokens":4096,"num_sink_tokens":64,"num_recent_tokens":512,"enable_profiler":true}'
```

导出 sqlite：

```bash
/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys export \
  --type sqlite \
  --force-overwrite=true \
  -o profiler_outputs/nsys_overlap_attnpredict_r16_s16_bs2_128k_o64.sqlite \
  profiler_outputs/nsys_overlap_attnpredict_r16_s16_bs2_128k_o64.nsys-rep
```

## 14. 新会话继续工作时的建议顺序

1. 先读本文档和第 3 节列出的文件。
2. 检查当前 `attnpredict_offload.py` 是否仍为 `self._layer_reuse_stride = 4`。
3. 不要重复已回退实验。
4. 如果用户要写 PPT，优先使用 `2026-06-07_优化技术方案_实验结果版.md` 和图片目录。
5. 如果用户要继续优化性能，先用 profiler 定位，不要直接猜。
6. 如果用户要做质量，优先做 calibration 诊断和 LongBench QA/摘要对比。
7. 每次代码修改后至少运行：

```bash
.venv/bin/python -m py_compile <改动的 Python 文件>
```

8. 任何 benchmark 结果都要记录到：

```text
attnpredict-offload资料/2026-05-30_nsys最终timeline与优化结论.md
```

或者新建当日中文实验报告。

## 15. 给新 Codex 的一句话总结

当前 `attnpredict-offload` 在 22 vCPU 机器上，`128k / bs=2 / output_len=64` 已通过 `reuse=16/stale=16 + layer_reuse_stride=4 + compiled CNN + runtime 固定成本优化` 达到约 `52.3 tok/s`，超过 vanilla 的约 `46.7 tok/s`；但质量在 LongBench QA/摘要上仍有样本级波动，跨机器结果存在敏感性，后续应优先补 calibration 质量实验和跨机器 profiler，而不是继续盲目做 kernel 融合。
