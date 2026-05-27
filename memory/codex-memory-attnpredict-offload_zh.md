# Codex 会话记忆：AttentionPredictor Offload

本文档用于开启新会话时快速恢复上下文。建议新会话开始后先阅读本文件，再阅读相关代码文件。

## 交互偏好

- 用户要求中文交流，回答要客观，不要附和式确认。
- 用户希望解释底层原因，尤其是 KV cache、slot、prefetch、CUDA stream、Git 状态等概念。
- Sparse 方法相关改动应遵守仓库约束：方法状态放在 `src/sparsevllm/engine/cache_manager/`，`attention.py` 尽量保持通用。

## 本次核心目标

围绕 `attnpredict-offload` 做实现、解释和小范围重构。

目标语义：

- `attnpredict-offload` 是 AttentionPredictor 的 offload 版本。
- GPU 侧是 SnapKV 式 active KV pool。
- CPU 侧保存完整历史 KV，即 CPU full backing，也可称“CPU 端完整 KV 后备存储”或“CPU 全量 KV 备份区”。
- Prefill 保持 full attention，不稀疏化。
- Prefill 后和每步 decode 后，根据 AttentionPredictor 的 mask 预测下一步需要的 token。
- 后台线程和独立 CUDA stream 负责 predictor + CPU->GPU 预取。
- 主线程继续算后续层，只在下一步同层真正消费 KV 时等待该层 prefetch 完成。

## 当前相关文件

重点文件：

- `src/sparsevllm/engine/cache_manager/attnpredict_offload.py`
- `src/sparsevllm/engine/cache_manager/attnpredict.py`
- `src/sparsevllm/engine/cache_manager/base.py`
- `src/sparsevllm/engine/sparse_controller.py`
- `src/sparsevllm/layers/attention.py`
- `src/sparsevllm/config.py`
- `README.md`
- `.vscode/launch.json`

参考文档：

- `docs/AttentionPredictor集成修复说明与推理流程.md`
- `docs/attnpredict_integration_changes.md`
- `docs/attnpredict_offload_todo_zh.md`

## 已完成的实现与重构

### 方法注册与别名

`attnpredict-offload` 已作为一等方法集成。

别名：

- `attentionpredictor-offload`
- `attenpredictor-offload`

代码内部会 canonicalize 成：

```python
attnpredict-offload
```

相关位置：

- `src/sparsevllm/config.py`
- `src/sparsevllm/engine/cache_manager/base.py`

### CPU KV backing 结构

已经把 CPU full backing 从两组 list：

```python
self.cpu_k_cache
self.cpu_v_cache
```

重构为统一张量：

```python
self.cpu_kv_cache = torch.empty(
    2,
    self.num_layers,
    self.cpu_num_slots,
    self.num_kv_heads,
    self.head_dim,
    dtype=self.hf_config.torch_dtype,
    device="cpu",
)
```

含义：

```python
self.cpu_kv_cache[0, layer_idx]  # K
self.cpu_kv_cache[1, layer_idx]  # V
```

CPU slot 不区分 K/V，K/V 由第 0 维区分。

已更新读写点：

- `_copy_cpu_to_gpu()`
- `on_kv_stored()`

静态搜索已确认无旧字段残留：

```bash
rg "cpu_k_cache|cpu_v_cache|mem_available \\* 0\\.70" src/sparsevllm/engine/cache_manager/attnpredict_offload.py
```

### CPU 内存比例参数

新增配置：

```python
attnpredict_offload_cpu_memory_utilization: float = 0.70
```

仅在 `vllm_sparse_method == "attnpredict-offload"` 时校验：

```text
0 < attnpredict_offload_cpu_memory_utilization <= 1
```

作用：

- 当 `attnpredict_offload_cpu_slots=-1` 时，用该比例估算 CPU full-KV slot 容量。
- 默认 `0.70` 保持原行为，只是从硬编码变成参数。

README 已补充说明。

### AttentionPredictor 状态 helper

用户指出 `attnpredict_offload.py` 重复了 `attnpredict.py` 中的 predictor 初始化。

已在 `src/sparsevllm/engine/cache_manager/attnpredict.py` 抽出：

```python
def _init_attnpredictor_state(self, config: Config) -> None:
    ...
```

普通版：

```python
class AttnPredictCacheManager(StandardCacheManager):
    def __init__(...):
        super().__init__(...)
        self._init_attnpredictor_state(config)
```

Offload 版：

```python
class AttnPredictOffloadCacheManager(AttnPredictCacheManager):
    def __init__(...):
        CacheManager.__init__(self, config, rank, world_size)
        ...
        self._init_attnpredictor_state(config)
```

注意：offload 仍继承 `AttnPredictCacheManager`，但不调用它的 `__init__()`，因此不会初始化 `StandardCacheManager`，不会额外分配标准全量 GPU KV cache。

### 编译检查

本次已跑过：

```bash
python -m py_compile \
  src/sparsevllm/config.py \
  src/sparsevllm/engine/cache_manager/attnpredict.py \
  src/sparsevllm/engine/cache_manager/attnpredict_offload.py
```

通过。

## 关键设计解释

### 为什么普通 attnpredict 不需要每层一个 free_slots_stack

普通 `attnpredict` 继承 `StandardCacheManager`。

普通模式中，一个 token 的 slot id 在所有层都一致：

```text
slot 100:
  layer 0 -> kv_cache[:, 0, 100]
  layer 1 -> kv_cache[:, 1, 100]
  layer 2 -> kv_cache[:, 2, 100]
```

即使每层 predictor mask 不同，完整 KV 仍全量常驻 GPU，所以只需要一套：

- `free_slots_stack`
- `buffer_req_to_token_slots`
- `LayerBatchStates`

### 为什么 offload 要每层一套 active pool 元数据

Offload 的 GPU cache 是 per-layer active pool。

同一个逻辑 token 在不同层可能：

```text
layer 0 -> GPU slot 12
layer 1 -> 不在 GPU，slot = -1
layer 2 -> GPU slot 87
```

所以 offload 必须每层维护：

- `free_slots_stack[layer]`
- `buffer_req_to_token_slots[layer]`
- `gpu_req_to_token_slots_cpu[layer]`
- `layer_batch_states[layer]`
- `prefetch_stream[layer]`
- `prefetch_future[layer]`

### slot_mapping 为什么每层能不一样

`slot_mapping` 的含义是“当前这批输入 token 的 K/V 要写到哪个物理 GPU slot”。

普通版本中，slot 是跨层一致的，所以一份 `slot_mapping` 足够。

Offload 中，每层 active pool 的空闲情况不同，同一个当前 decode token 在不同层可能分到不同 slot：

```text
当前 token:
  layer 0 -> slot 12
  layer 1 -> slot 82
  layer 2 -> slot 31
```

因此 offload 的 `LayerBatchStates` 是 per-layer 的。

### resident 的含义

`resident` 表示 token 的 KV 当前驻留在某一层的 GPU active pool 中。

```text
resident     -> row/pos 有有效 gpu_slot
non-resident -> row/pos 在 GPU 映射中为 -1，只在 CPU full backing 中有完整 KV
```

`_ensure_positions_resident()` 的意思是：确保某层下一次 attention 需要的 positions 都已经在 GPU active pool 里。

### `_ensure_positions_resident()` 做什么

输入：

```python
row_positions: dict[int, np.ndarray]
```

含义：

```text
row_idx -> 下一次 attention 需要可见的 token positions
```

核心步骤：

1. 清理不在目标集合中的旧 GPU resident token，释放 GPU active slot。
2. 找出目标集合中当前不在 GPU 的 token。
3. 为这些 missing token 分配 GPU active slot。
4. 从 CPU full backing 把 missing token 的 K/V 拷回 GPU。
5. 更新 GPU 上的 `buffer_req_to_token_slots[layer][row, pos]`。

其中：

- `mirror = self.gpu_req_to_token_slots_cpu[layer_idx]` 是 CPU 侧 GPU residency 镜像。
- `free_rows/free_pos/free_slots` 是待释放映射更新清单，`free_slots` 通常为 `-1`。
- `load_rows/load_pos/load_cpu_slots/load_gpu_slots` 是待从 CPU 拉回 GPU 的清单。

### `_write_gpu_map()` 为什么调用两次

`_write_gpu_map()` 将 CPU 侧 residency 变更同步写到 GPU 映射表：

```python
self.buffer_req_to_token_slots[layer_idx][rows_t, pos_t] = slots_t
```

调用两次是因为语义不同：

```python
self._write_gpu_map(layer_idx, free_rows, free_pos, free_slots, stream=stream)
self._write_gpu_map(layer_idx, load_rows, load_pos, load_gpu_slots, stream=stream)
```

第一次：

- 把释放掉的 token 写成 `-1`
- 表示这些 token 不再驻留 GPU

第二次：

- 把新加载的 token 写成新的 `gpu_slot`
- 表示这些 token 已经驻留 GPU

技术上可以合并，但分开更清楚。

### 异步预取相关变量

在 `AttnPredictOffloadCacheManager.__init__()` 中：

```python
self._prefetch_streams = [torch.cuda.Stream() for _ in range(self.num_layers)]
self._prefetch_futures = [None for _ in range(self.num_layers)]
self._prefetch_errors = [None for _ in range(self.num_layers)]
self._layer_locks = [threading.RLock() for _ in range(self.num_layers)]
self._cnn_lock = threading.RLock()
```

含义：

- `_prefetch_streams`：每层一个 CUDA stream，用于后台 predictor/H2D prefetch。
- `_prefetch_futures`：每层上一轮后台预取任务的 Future 句柄。
- `_prefetch_errors`：每层后台预取异常记录。
- `_layer_locks`：每层 active pool 元数据保护锁。
- `_cnn_lock`：全局 CNN predictor 锁，避免多个后台线程同时跑同一个 `self.cnn`。

## 推理流程概述

### Prefill

1. `_prepare_prefill()` 为当前 chunk 的所有 token 分配 GPU slot 和 CPU slot。
2. 当前 chunk 保持 full attention。
3. `on_kv_stored()` 将每层 K/V 同步保存到 CPU full backing。
4. 只有最后一个 prefill chunk 会用尾部 attention 初始化 predictor history/mask。
5. 最后一个 prefill chunk 的每层 attention 结束后，`on_prefill_layer_end()` 根据 mask 准备首个 decode 的 active set。
6. 后台 stream 可以释放不需要的 GPU active slots，但 CPU full backing 不删。

### Decode

1. `_prepare_decode()` 只分配 CPU full backing slot，不提前给所有层分配 GPU slot。
2. 每层进入 `get_layer_store_view(layer)` 时：
   - 等待该层上一轮 prefetch 完成。
   - 根据上一轮 mask 确保本层需要的历史 positions resident。
   - 给当前 decode token 分配本层 GPU slot。
   - 返回该层的 `slot_mapping` 给 `store_kvcache()`。
3. `on_kv_stored()` 把当前层新 token K/V 保存到 CPU full backing。
4. `build_decode_view()` 打包当前层实际可读的 GPU slots。
5. Attention kernel 计算并写出 `attn_score`。
6. `SparseController.on_attention_end()` 调用 `predict_next_mask()`。
7. `predict_next_mask()` 提交后台任务，后台在独立 stream 中做 predictor + CPU->GPU prefetch。

## 性能观察与判断

用户跑过一次 benchmark，结果大致为：

```text
Len=10000, BS=2, decode_tokens=14

vanilla:
  TTFT 1.95s, Decode 36.1 tok/s, ITL 55.38ms, Mem 25.43GB

snapkv:
  TTFT 2.00s, Decode 30.1 tok/s, ITL 66.54ms, Mem 25.45GB

attnpredict-offload:
  TTFT 3.21s, Decode 3.6 tok/s, ITL 555.03ms, Mem 25.44GB
```

判断：

- 这个结果不一定说明 correctness bug。
- 当前 `attnpredict-offload` 更像 correctness-first 原型，还不是高性能版本。
- BS=2、decode token 很少时，offload 的固定开销会被放大。
- CPU full backing 写入、CPU gather、H2D reload、prefetch wait、CNN predictor 都可能成为瓶颈。
- `Mem(GB)` 几乎一样是正常的，因为 GPU KV pool 仍是预分配；offload 收益主要体现在释放 per-seq inactive slots 后提升并发容量，而不是降低启动时整块 CUDA allocation。

建议后续性能诊断命令：

```bash
PROFILER_SVLLM=1 CUDA_SYNC_SVLLM=1 \
SPARSEVLLM_MASTER_PORT=2345 \
python scripts/bench_sparse_vllm.py \
  --model_path <模型路径> \
  --methods attnpredict,attnpredict-offload \
  --lengths 10000 \
  --batch_sizes 2 \
  --output_len 64 \
  --hyper_params '{"attnpredict_model_path":"<predictor checkpoint>"}'
```

重点看 profiler 项：

- `attnpredict_offload_prefetch_wait`
- `attnpredict_offload_cpu_gather_background`
- `attnpredict_offload_h2d_prefetch_stream`
- `attnpredict_offload_predict_cnn_stream`
- `attnpredict_offload_store_cpu_full_kv`

如果 `prefetch_wait` 高，说明异步预取没有被后续层计算 cover。
如果 `cpu_gather_background` / `h2d_prefetch_stream` 高，说明 CPU reload 是瓶颈。
如果 `predict_cnn_stream` 高，说明 predictor 或 `_cnn_lock` 串行化影响大。

## 已解释过的 Git/环境问题

### 端口占用

报错：

```text
EADDRINUSE, port: 2333, address already in use
```

原因：PyTorch distributed 默认使用 `SPARSEVLLM_MASTER_PORT=2333`，端口被占用。

解决：

```bash
SPARSEVLLM_MASTER_PORT=2345 python scripts/bench_sparse_vllm.py ...
```

或在 VSCode launch 配置中设置：

```json
"env": {
  "SPARSEVLLM_MASTER_PORT": "2345"
}
```

查占用：

```bash
ss -ltnp | grep 2333
lsof -i :2333
```

### 未追踪文件不显示

绿色 `U` 是未追踪文件。推荐本地忽略：

```bash
nano .git/info/exclude
```

加入如：

```gitignore
models/
Research/
future_token_prediction_kvcache*/
kvcache_*/
```

黄色 `M` 是已追踪文件，`.gitignore` 对它无效。如 `.vscode/launch.json` 可本地忽略改动：

```bash
git update-index --skip-worktree .vscode/launch.json
```

恢复：

```bash
git update-index --no-skip-worktree .vscode/launch.json
```

### stash

解释过 `stash` 是“临时保存当前未提交改动”。用户曾要求删除一个 `stash@{0}: On main: wip before sync zz`。

## 当前保留的 TODO / 后续方向

明确未改：

- TODO 1：`world_size > 1` 仍不支持。当前有：

  ```python
  assert world_size == 1
  ```

- TODO 3：CPU full backing 容量仍按 `max_model_len * max_num_seqs_in_batch` 估算，没有改成基于单次 forward token 或 GPU 利用率估算。

可考虑的性能优化：

- 增加 reload token count 统计，记录每层每步从 CPU 拉回多少 token。
- 复用 pinned staging buffer，避免每次 `_copy_cpu_to_gpu()` 临时分配 pinned tensor。
- 减少 Python set/list/numpy 操作在每层每步的开销。
- 对比 `attnpredict` 和 `attnpredict-offload`，先分离 predictor 开销和 CPU reload 开销。
- 如果 `_cnn_lock` 成为瓶颈，可考虑每层/每 worker 独立 CNN 实例，但会增加显存。
- 研究更低开销的 CPU gather/H2D 路径。

## 新会话建议第一步

如果用户继续问代码问题，优先打开：

```bash
sed -n '1,220p' src/sparsevllm/engine/cache_manager/attnpredict_offload.py
sed -n '1,330p' src/sparsevllm/engine/cache_manager/attnpredict.py
sed -n '260,350p' src/sparsevllm/engine/sparse_controller.py
sed -n '160,320p' src/sparsevllm/layers/attention.py
```

如果用户问性能，先让他贴 profiler 输出，或建议开启：

```bash
PROFILER_SVLLM=1 CUDA_SYNC_SVLLM=1
```

如果用户问是否该重构继承关系，当前结论是：

- 现在保持 `class AttnPredictOffloadCacheManager(AttnPredictCacheManager)` 可以接受。
- 关键是 offload 不调用 `super().__init__()`，而是调用 `CacheManager.__init__()`，因此不会初始化 `StandardCacheManager`。
- 长期更干净的结构是抽 `AttnPredictorMixin`，但用户最后决定暂时保持现状。
