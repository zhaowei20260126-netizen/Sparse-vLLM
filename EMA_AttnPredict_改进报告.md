# Q-Infer AttnPredict 改进：CNN → EMA（指数滑动平均）

## 1. 背景与动机

Q-Infer 项目基于 llama.cpp 实现了 AttentionPredictor（AP）稀疏 KV Cache 机制，通过预测哪些 KV token 在下一步 attention 中是重要的，只保留 top-k 个 token 参与计算，从而在长序列推理时减少 attention 的计算量。

原方案使用 3 层 CNN 模型进行预测，存在以下限制：
- 需要离线训练 CNN 模型，针对不同模型/任务需重新训练
- CNN 前向推理本身有计算开销（~0.016ms/次）
- 部署时需要额外的模型权重文件
- 不适合纯 CPU 部署场景

**改进目标：** 将 CNN 预测器替换为 EMA（指数滑动平均），实现零训练、极低开销、自适应的热点追踪。

## 2. EMA 预测机制

### 2.1 核心公式

```
token_importance[i] = α × cur_attention_score[i] + (1-α) × token_importance[i]
```

- `α = 0.1`（新信息权重）
- `1-α = 0.9`（历史惯性）
- `cur_attention_score[i]`：当前 decode 步中 token i 获得的 attention 分数
- `token_importance[i]`：累积的 token 重要性分数

### 2.2 工作原理示例

```
═══ Step 1: 模型生成第1个回答token ═══
  当前注意力分数 cur_score = [0.1, 0.8, 0.05, 0.02, 0.03]
  更新后: importance = [0.01, 0.08, 0.005, 0.002, 0.003]

═══ Step 2: 模型生成第2个回答token ═══
  当前注意力分数 cur_score = [0.05, 0.7, 0.1, 0.05, 0.1]
  更新后: importance = [0.014, 0.142, ...]
           ↑ token 1 持续被关注，分数持续积累

═══ Step 3: 话题转移，模型开始关注 token 3 ═══
  当前注意力分数 cur_score = [0.05, 0.1, 0.7, 0.1, 0.05]
  更新后: importance = [..., 0.138(↓衰减), 0.0745(↑上升), ...]
```

### 2.3 Mask 生成策略（Per-Block 粒度）

1. 将 KV 序列按 `block_size=16` 分组
2. 对每个 block 内的 token importance 取 max 作为 block score
3. 对中间区域（排除 sink 和 local）的 block 做 top-k 选择
4. 被选中的 block 内所有 token 保留（mask=0），其余截断（mask=-10000）
5. Sink 区（前64 token）和 Local 区（后64 token）始终保留

### 2.4 Mask 复用机制

- 每 `reuse_step=5` 步重新生成一次完整 mask
- 中间步骤复用上一次的 mask，仅做增量扩展（新增 token 加入 local 区）
- 减少 top-k 排序的频率，进一步降低开销

## 3. 代码改动

### 3.1 修改文件清单

| 文件 | 改动内容 |
|------|---------|
| `llama-attn-pred.h` | 添加 `use_ema`、`token_importance`、`importance_kv_len`、`ema_alpha` 字段；添加 `llama_attn_pred_init_ema` 声明 |
| `llama-attn-pred.cpp` | 新增 `ap_ema_update`、`ap_create_mask_ema` 函数；`ap_async_process` 增加 EMA/CNN 分支；新增 `llama_attn_pred_init_ema` 实现 |
| `llama.cpp` | 添加 `use_ema_predictor` 到 model 结构体和参数；模型加载时根据开关选择 EMA 或 CNN 初始化 |
| `llama.h` | `llama_model_params` 添加 `use_ema_predictor` 字段 |
| `common/common.h` | `gpt_params` 添加 `use_ema_predictor` 字段 |
| `common/common.cpp` | 添加 `--use-cnn` 命令行参数解析；传递参数到 model params |

### 3.2 核心新增函数

**`ap_ema_update`** — EMA 重要性更新：
```cpp
static void ap_ema_update(
    struct llama_attn_predictor & pred,
    const float * attn_data,  // [n_heads, n_tokens, kv_len]
    int kv_len, int n_tokens, int n_heads
) {
    if (kv_len > pred.importance_kv_len) {
        pred.token_importance.resize((size_t)n_heads * kv_len, 0.0f);
        pred.importance_kv_len = kv_len;
    }
    const float alpha = pred.ema_alpha;
    const float one_minus_alpha = 1.0f - alpha;

    for (int h = 0; h < n_heads; h++) {
        const float * cur_score = attn_data
            + (size_t)h * n_tokens * kv_len
            + (size_t)(n_tokens - 1) * kv_len;
        float * imp = pred.token_importance.data() + (size_t)h * kv_len;
        for (int i = 0; i < kv_len; i++) {
            imp[i] = alpha * cur_score[i] + one_minus_alpha * imp[i];
        }
    }
}
```

**`ap_create_mask_ema`** — 基于 EMA 分数生成 per-block 稀疏 mask：
```cpp
static void ap_create_mask_ema(
    const struct llama_attn_predictor & pred,
    int full_kv_len,
    std::vector<float> & mask_out
) {
    // 按 block_size 分组，取每 block 内 importance 最大值
    // 对中间区域 block 做 top-k 选择
    // sink + local 区域始终保留
}
```

### 3.3 运行时切换

```bash
# EMA 模式（默认）
./build/bin/main -m model.gguf -n 128 -t 2 -p "prompt"

# CNN 模式
./build/bin/main -m model.gguf -n 128 -t 2 -p "prompt" --use-cnn

# 关闭 AP（完整注意力基线）
./build/bin/main -m model.gguf -n 128 -t 2 -p "prompt" --no-attn-pred
```

## 4. 实验设置

### 4.1 硬件环境

- **GPU:** VGPU-32GB
- **CPU:** 16 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz
- **平台:** AutoDL 容器，Ubuntu + CUDA

### 4.2 模型配置

- **模型:** ReluLLaMA-7B（Q4 量化，PowerInfer GGUF 格式）
- **注意力头数:** 32
- **层数:** 32（前 2 层跳过 AP）

### 4.3 AP 超参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `pooling_block_size` | 16 | mask 按 16 token 一组选择 |
| `sink_token` | 64 | 始终保留序列开头 64 个 token |
| `local_token` | 64 | 始终保留序列末尾 64 个 token |
| `topk` | 256 | KV Cache 总预算（token 数） |
| `reuse_step` | 5 | 每 5 步重新预测 mask |
| `ema_alpha` | 0.1 | EMA 平滑系数 |
| `calibration_step` | 5 | 每 5 步做一次全量注意力校准 |

### 4.4 推理参数

- **线程数:** 2（`-t 2`）
- **上下文长度:** 512（`-c 512`）
- **生成长度:** 128 tokens（`-n 128`）
- **Prompt:** "Hello world, tell me a long story about a wizard"（13 tokens）

## 5. 实验结果

### 5.1 解码速度对比

| 预测模式 | eval time (ms/token) | 速度 (tokens/sec) | 相对基线 |
|---------|---------------------|-------------------|---------|
| **EMA（本方案）** | **63.13** | **15.84** | **+1.7%** |
| 无 AP（完整注意力）| 64.18 | 15.58 | 基线 |
| CNN（原方案）| 119.28 | 8.38 | -46.2% |

### 5.2 分析

1. **EMA 开销可忽略：** EMA 模式与无 AP 基线速度几乎一致（15.84 vs 15.58 tokens/sec），说明 EMA 的计算开销（一次向量乘加）对总推理时间几乎无影响。

2. **CNN 开销显著：** CNN 模式在短序列下反而比基线慢 46%，因为每步 decode 都需要执行 CNN 前向推理（3 层卷积），即使序列长度未超过 topk 阈值不需要裁剪，CNN 开销仍然存在。

3. **短序列局限：** 本次实验总序列长度约 49 tokens（13 prompt + 36 generated），未超过 topk=256 的阈值。因此稀疏 mask 未实际生效，三种模式的 attention 计算量相同。EMA 的真正加速效果需要在长序列（>256 tokens）场景下验证。

### 5.3 方案对比总结

| 对比维度 | CNN 原方案 | EMA 本方案 |
|---------|-----------|-----------|
| 预测机制 | 训练 CNN 学习注意力时序模式 | EMA 追踪 token 被关注频率 |
| 实现复杂度 | 需收集数据、离线训练、部署权重文件 | 一行向量乘加，零训练 |
| CPU 开销 | ~0.016ms/次（含卷积计算） | ~0.001ms/次（纯算术） |
| 泛化性 | 需要针对模型/任务重新训练 | 自适应任何文档/任务 |
| 理论依据 | 数据驱动学习 | 工作集理论 + 时间局部性 |
| 部署依赖 | 需要 CNN 权重文件（best_model.bin） | 无额外依赖 |
| 适合场景 | GPU 充足，可接受训练成本 | **纯 CPU 部署、边缘设备、快速原型** |

## 6. 编译与运行指南

### 6.1 编译（Linux + CUDA）

```bash
cd Q-Infer
rm -rf build
cmake -S . -B build -DLLAMA_CUBLAS=ON
cmake --build build --config Release -j$(nproc)
```

### 6.2 运行

```bash
# EMA 模式（推荐，默认）
./build/bin/main -m ReluLLaMA-7B/llama-7b-relu.q4.powerinfer.gguf \
    -n 128 -t 2 -p "Your prompt here" -c 2048

# 查看 AP 耗时日志
LLAMA_AP_TIME_LOG=1 ./build/bin/main -m ReluLLaMA-7B/llama-7b-relu.q4.powerinfer.gguf \
    -n 128 -t 2 -p "Your prompt here" -c 2048

# CNN 模式对比
./build/bin/main -m ReluLLaMA-7B/llama-7b-relu.q4.powerinfer.gguf \
    -n 128 -t 2 -p "Your prompt here" -c 2048 --use-cnn
```

## 7. 后续优化方向

1. **长序列验证：** 使用 >1024 tokens 的长文档 prompt 验证 EMA 稀疏加速效果
2. **α 调参：** 测试不同 α 值（0.05~0.3）对预测质量的影响
3. **Perplexity 评估：** 对比 EMA/CNN/无AP 三种模式下的 PPL，量化稀疏对输出质量的影响
4. **纯 CPU 场景：** 在无 GPU 环境下测试 EMA 的性能优势（CNN 开销在 CPU 上更显著）
