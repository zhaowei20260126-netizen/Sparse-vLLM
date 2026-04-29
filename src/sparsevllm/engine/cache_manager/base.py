from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.constant import REDUNDANCY_BATCH_SIZE_FACTOR
from sparsevllm.utils.log import logger, log_level


@dataclass
class LayerBatchStates:
    """存储当前 Batch 在特定层的前向计算状态。

    仅包含与物理存储和基本前向元数据相关的字段。
    """

    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    req_indices: torch.Tensor | None = None


class CacheManager(ABC):
    """每个 Rank 只有一个 CacheManager，内部管理所有层的物理槽位和 KV Cache。"""

    def __init__(self, config: Config, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.hf_config = config.hf_config
        self.num_layers = self.hf_config.num_hidden_layers

        self.num_kv_heads = self.hf_config.num_key_value_heads // world_size
        self.head_dim = getattr(
            self.hf_config,
            "head_dim",
            self.hf_config.hidden_size // self.hf_config.num_attention_heads,
        )

        self.max_model_len = config.max_model_len
        
        # 初始化 Cache Manager 内部用于管理序列（Sequence）的元数据缓冲区行数上限。
        # 每处理一个新序列，在元数据映射表（如 req_to_token_slots）中就需要占据一行（Row）来专门记录该序列包含的 token 存放在哪些物理插槽中。
        # 为什么不直接等于单次 batch 的最大并发数 (max_num_seqs_in_batch)，而是要乘上一个冗余系数 (REDUNDANCY_BATCH_SIZE_FACTOR)？
        # 1. 调度队列排队：很多序列可能只是处于被抢占（Swapped）或等待状态，当下并没有参与这一个 Batch 的前向传播，但它们的历史 KV cache 映射关系不能丢，这就需要一直占据一行元数据。
        # 2. 复杂的生成策略：例如 Beam Search 中，一个请求会随解码步骤不断分叉出多个候选路线，每个独立路线都需要占用一行。
        # 3. 延迟释放：序列运行结束后，底层可能出于性能考虑做延迟或批量释放，冗余量保证高吞吐时元数据表不会枯竭。
        self.max_buffer_rows = config.max_num_seqs_in_batch * REDUNDANCY_BATCH_SIZE_FACTOR

        self.kv_cache = None

    @staticmethod
    def create(config: Config, rank: int, world_size: int) -> "CacheManager":
        sparse_method = config.vllm_sparse_method
        model_type = getattr(getattr(config, "hf_config", None), "model_type", "") or ""

        # DeepSeek MLA cache layout (required for DeepSeek-V2 / V3.2 model wiring).
        if model_type == "deepseek_v32":
            raise NotImplementedError(
                "DeepSeek-V3.2 sparsevllm support is disabled. "
                "Use DeepSeek-V2 or another backend for now."
            )
        if model_type == "deepseek_v2":
            if sparse_method not in ("", "dsa"):
                raise ValueError(
                    f"DeepSeek MLA cache does not support vllm_sparse_method={sparse_method!r}. "
                    "Use vanilla mode (empty string) or `dsa` where supported."
                )
            if sparse_method == "dsa" and model_type != "deepseek_v32":
                raise ValueError(
                    "vllm_sparse_method='dsa' is currently only supported for DeepSeek-V3.2 "
                    f"(model_type={model_type!r})."
                )
            from .deepseek_mla import DeepSeekMLACacheManager

            return DeepSeekMLACacheManager(config, rank, world_size)
        if sparse_method == "dsa":
            raise ValueError(
                "vllm_sparse_method='dsa' is currently only supported for DeepSeek-V3.2 "
                f"(model_type={model_type!r})."
            )
        if model_type == "qwen3" and isinstance(sparse_method, str) and sparse_method.startswith("deltakv"):
            raise NotImplementedError(
                "sparsevllm qwen3 + deltakv is disabled for now due to qk-norm/runtime mismatch. "
                "Use the HF backend for qwen3 DeltaKV inference."
            )

        if sparse_method == "deltakv":
            from .deltakv import DeltaKVCacheManager

            return DeltaKVCacheManager(config, rank, world_size)
        if sparse_method == "deltakv-triton":
            # Run DeltaKV logic, but use Triton for reconstruction.
            config.vllm_sparse_method = "deltakv"
            from .deltakv import DeltaKVCacheTritonManager

            return DeltaKVCacheTritonManager(config, rank, world_size)
        if sparse_method == "deltakv-triton-v2":
            # Run DeltaKV logic, but use Triton for reconstruction + eviction.
            config.vllm_sparse_method = "deltakv"
            from .deltakv import DeltaKVCacheTritonManagerV2

            return DeltaKVCacheTritonManagerV2(config, rank, world_size)
        if sparse_method == "deltakv-triton-v3":
            # Run DeltaKV logic, with Triton for reconstruction + eviction + blockwise L2-topk.
            config.vllm_sparse_method = "deltakv"
            from .deltakv import DeltaKVCacheTritonManagerV3

            return DeltaKVCacheTritonManagerV3(config, rank, world_size)
        if sparse_method == "deltakv-triton-v4":
            # Run DeltaKV logic, with Triton for reconstruction + eviction + blockwise L2-topk,
            # and extra kernel fusions (grouped-head reconstruction, fused gather+mean for clustering).
            config.vllm_sparse_method = "deltakv"
            from .deltakv import DeltaKVCacheTritonManagerV4

            return DeltaKVCacheTritonManagerV4(config, rank, world_size)
        if sparse_method == "deltakv-standalone":
            from .deltakv_standalone import DeltaKVStandaloneCacheManager

            return DeltaKVStandaloneCacheManager(config, rank, world_size)
        if sparse_method == "deltakv-snapkv":
            from .deltakv_snapkv import DeltaKVSnapKVCacheManager

            return DeltaKVSnapKVCacheManager(config, rank, world_size)
        if sparse_method in ("deltakv-triton-v3-offload", "deltakv-triton-v3-with-offload"):
            # Run DeltaKV logic, with Triton for reconstruction + eviction + blockwise L2-topk,
            # and offload latent cache to CPU.
            config.vllm_sparse_method = "deltakv"
            config.deltakv_offload_latent = True
            from .deltakv import DeltaKVCacheTritonManagerV3WithOffload

            return DeltaKVCacheTritonManagerV3WithOffload(config, rank, world_size)
        if sparse_method in ("deltakv-triton-v3-cuda-offload", "deltakv-triton-v3-with-cuda-offload"):
            # Run DeltaKV logic, and offload latents to CPU, but gather them back to GPU via
            # the custom CUDA kernel under `sparsevllm/cuda_kernel` (ShadowKV-style).
            config.vllm_sparse_method = "deltakv"
            config.deltakv_offload_latent = True
            from .deltakv import DeltaKVCacheTritonManagerV3WithCUDAOffload

            return DeltaKVCacheTritonManagerV3WithCUDAOffload(config, rank, world_size)
        if sparse_method in ("streamingllm", "attention-sink", "attention_sink"):
            from .streamingllm import StreamingLLMCacheManager

            return StreamingLLMCacheManager(config, rank, world_size)
        if sparse_method in ("snapkv", "pyramidkv"):
            from .snapkv import SnapKVCacheManager

            return SnapKVCacheManager(config, rank, world_size)
        if sparse_method == "quest":
            from .quest import QuestCacheManager

            return QuestCacheManager(config, rank, world_size)
        if sparse_method == "omnikv":
            from .omnikv import OmniKVCacheManager

            return OmniKVCacheManager(config, rank, world_size)

        from .standard import StandardCacheManager # 注意力计算时访问全部历史 token（因此叫 "vanilla"）

        return StandardCacheManager(config, rank, world_size)

    def _get_available_slots_info(self) -> tuple[int, int]:
        """返回 (可用显存字节数, 每层每 token 的字节数)"""
        config = self.config
        hf_config = config.hf_config
        # 获取当前 GPU 物理层面上真实的空闲显存和总显存
        free, total = torch.cuda.mem_get_info()

        # =========================================================================
        # 阶段 1：动态估计并修正 max_num_batched_tokens，防止前向计算中激活显存击穿
        # =========================================================================

        # gpu_memory_utilization 默认通常是 0.8，意味着 20% 预留给前向传播等动态开销。
        reserved_mem = total * (1 - config.gpu_memory_utilization)
        
        # 获取 MLP 层的中间维度大小。MLP 层通常是模型前向传播中产生最大激活张量（Activation）的地方。
        # 如果模型配置没写，默认认为是 hidden_size * 4 (LLaMA 等模型的常见比例)
        intermediate_size = getattr(hf_config, "intermediate_size", hf_config.hidden_size * 4)
        
        # 计算单个元素的字节数，例如 float16 或 bfloat16 是 2 字节
        dtype_size = torch.tensor([], dtype=hf_config.torch_dtype).element_size()

        # 启发式算法：估计 `reserved_mem` 能支撑的最大单次前向传播的 token 数量。
        # 为什么要乘 10？
        # - 一个 MLP 中间激活大约占 intermediate_size * dtype_size 个字节。
        # - 前向传播时不仅有 MLP，还有 Attention 的 Q、K、V、Softmax 结果等多个大张量同时存在。
        # - 此外还要为 PyTorch 内存池的碎片化（Fragmentation）留出大量冗余。
        # 因此这是一个保守的估计公式，用于推算最大的 batch tokens。
        estimated_max_tokens = int(reserved_mem / (intermediate_size * dtype_size * 10))
        
        # 检查配置的 chunk_prefill_size（单次处理的分块大小）是否过于激进。
        # 至少要保证估计的最大容量是单次 chunk 大小的 2 倍以上，否则很容易 OOM。
        assert 2 * config.chunk_prefill_size < estimated_max_tokens, (
            f"{2 * config.chunk_prefill_size} >= {estimated_max_tokens}"
        )

        # 如果用户配的批处理上限大于我们保守估计的最大上限，强行下调它，防止后续 OOM
        if estimated_max_tokens < config.max_num_batched_tokens:
            logger.warning(
                f"Estimated max_num_batched_tokens ({estimated_max_tokens}) is smaller than config "
                f"({config.max_num_batched_tokens}). Updating to avoid OOM."
            )
            config.max_num_batched_tokens = estimated_max_tokens

        logger.info(f"Set dynamically max_num_batched_tokens = {config.max_num_batched_tokens}")

        # =========================================================================
        # 阶段 2：计算还能用来分配 KV Cache 的真实可用显存
        # =========================================================================

        # 当前 GPU 物理层面已用的总显存（包含了 PyTorch 缓存的和非 PyTorch 使用的）
        used = total - free
        
        # peak: 历史显存占用最高水位线。
        '''
        从进程启动到当前为止，PyTorch CUDA allocator 见过的最高已分配水位。它主要来自：
            模型结构创建时的 GPU 参数分配
            load_model() 把权重 copy 到 GPU 参数
            初始化过程中产生过的临时 CUDA 张量/缓冲
            可能还有一些库初始化产生的 PyTorch CUDA 分配
        '''
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        # 临时张量释放后:
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # 计算剩余可用给 KV Cache 的理论显存。
        # 公式解析：
        # - 总预算: total * config.gpu_memory_utilization
        # - 减去物理已用: - used (此时排除了模型权重和常驻库占用的真实物理内存)
        # - 减去历史上出现过的额外临时显存涨幅: - (peak - current) 
        available_memory = int(total * config.gpu_memory_utilization - used - (peak - current))

        # 计算存储 1 个 Token 在单层中的 KV Cache 所需要的字节数。
        # 包含 2 部分(Key 和 Value)，每部分有 num_kv_heads 个头，每个维度为 head_dim，然后乘单个元素的字节数。
        slot_bytes_per_layer = 2 * self.num_kv_heads * self.head_dim * dtype_size

        if log_level == "DEBUG":
            logger.debug(
                f"[DEBUG] Available Memory: {available_memory / 1024**3:.2f} GB, "
                f"Slot Bytes Per Layer: {slot_bytes_per_layer / 1024**2:.4f} MB"
            )

        return available_memory, slot_bytes_per_layer

    # =========================================================================
    # 阶段 A: 推理准备 (Prepare Step)
    # =========================================================================

    def prepare_step(self, seqs: list[Sequence], is_prefill: bool):
        """单步推理前的数据准备入口。根据阶段转发到 _prepare_prefill 或 _prepare_decode。

        由 ModelRunner.prepare_step() 调用，返回本轮所需的 input_ids、positions 和 cu_seqlens_q。
        内部逻辑由子类实现（vanilla/deltakv/snapkv 等各有不同的 slot 分配策略）。

        返回值:
            - prefill: (input_ids, positions, cu_seqlens_q)
            - decode:  (input_ids, positions, None)
        """
        if is_prefill:
            return self._prepare_prefill(seqs)
        return self._prepare_decode(seqs)

    # =========================================================================
    # 阶段 B: 显存分配 (Cache Allocation)
    # =========================================================================

    @abstractmethod
    def allocate_kv_cache(self):
        """自动计算并物理分配 KV Cache 张量。

        在子类构造函数末尾调用。核心逻辑：
        1. 调用 _get_available_slots_info() 获取 (可用显存字节数, 每层每 token 的字节数)
        2. 根据稀疏方法决定 KV cache 池的划分策略
           - StandardCacheManager: 单一全量池
           - DeltaKVCacheManager: 三池划分 (full / deltakv_full / deltakv_latent)
        3. 分配 torch.Tensor 并赋值给 self.kv_cache 或各子池
        4. 初始化 slot 管理元数据 (free_slots 队列、slot_maps 等)

        在 __init__ 中调用，分配后 kv_cache 大小不再变化。
        """
        raise NotImplementedError

    # =========================================================================
    # 阶段 C: 层状态与张量获取 (Layer State & Tensor Access)
    #     这些方法被 Attention.forward 和 SparseController 频繁调用，
    #     每层每 step 都会执行。是 KV Cache 消费侧的核心接口。
    # =========================================================================

    @abstractmethod
    def get_layer_batch_states(self, layer_idx: int) -> LayerBatchStates:
        """获取指定层的 LayerBatchStates。

        包含三个字段:
          - slot_mapping: (total_tokens,) → 每个 token 被写入的物理 slot ID (-1 表示不写入)
          - context_lens: (batch_size,) → 每个序列当前的 KV 可见长度
          - req_indices:  (batch_size,) → 每个序列在元数据表中的行号

        被 SparseController.prepare_forward() 调用，用于初始化每层的稀疏状态。
        也被 Scheduler 用于准入判断时获取上下文长度。
        """
        raise NotImplementedError

    @abstractmethod
    def get_layer_kv_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """获取指定层的 K/V 物理缓存张量。

        返回 (k_cache, v_cache)，形状通常为 (num_slots, num_kv_heads, head_dim)。
        被 Attention.forward 调用，用于从物理缓存中 gather KV 数据。
        对于标准方法，返回的是单一全量池中的该层张量；
        对于 DeltaKV，full-attn layers 返回 full_kv_cache，sparse layers 返回 deltakv_full_kv_cache。
        """
        raise NotImplementedError

    @abstractmethod
    def get_layer_store_view(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取指定层的 KV "写入视图"。

        返回 (k_cache, v_cache, slot_mapping):
          - k_cache / v_cache: 要写入的物理缓存张量引用
          - slot_mapping: 每个 token 对应的写入目的 slot ID

        被 Attention.forward 的第一步调用。Attention 层用此视图将当前 token/chunk 的
        K/V 写入物理缓存。无论什么稀疏方法，所有新 token 总是先以全精度写入物理缓存，
        后续的稀疏/压缩由 eviction 和 read view 控制。

        设计意图: "写"和"读"分离——写总是全量写入，读按稀疏视图选择性读取。
        """
        raise NotImplementedError

    @abstractmethod
    def get_layer_compute_tensors(self, layer_idx: int, sparse_controller):
        """获取指定层的"计算用"K/V 张量。

        与 get_layer_kv_cache 的区别：某些稀疏方法可能需要在计算时提供不同的
        张量视图（例如经过特殊布局转换后的张量）。大多数子类直接返回 get_layer_kv_cache。

        参数:
          - layer_idx: 层索引
          - sparse_controller: SparseController 实例，方法可据此查询当前的稀疏状态
        """
        raise NotImplementedError

    @abstractmethod
    def get_layer_buffer_req_to_token_slots(self, layer_idx: int) -> torch.Tensor:
        """获取指定层的 req→token→slot 完整映射表。

        返回形状 (max_buffer_rows, max_model_len) 的张量，其中 [row, pos] 表示某序列
        第 pos 个 token 存放在哪个物理 slot 中（-1 表示不存在/已驱逐/已压缩）。

        被 SparseController.get_read_view() 中非动态稀疏方法使用（vanilla, snapkv,
        streamingllm 等），返回全量 slots 供 attention kernel 使用。
        也被 OmniKV 的 build_omnikv_keep_and_slots 用于根据逻辑位置查找物理 slot。
        """
        raise NotImplementedError

    # =========================================================================
    # 阶段 D: 方法特定 Hook (Method-Specific Hooks)
    #     提供可选的方法特定行为注入点，默认实现为空操作。
    #     子类可覆写以接入自定义逻辑。
    # =========================================================================

    def on_kv_stored(self, layer_idx: int, k: torch.Tensor, slot_mapping: torch.Tensor):
        """KV 写入物理缓存后的方法特定 Hook。

        在 Attention.forward 中 store_kvcache() 执行完毕后立即调用。
        默认空操作。某些方法可能在此处做额外的元数据记录或状态更新。

        参数:
          - layer_idx: 当前层索引
          - k: 刚写入的 K 张量 (total_tokens, num_kv_heads, head_dim)
          - slot_mapping: 写入的目标 slot ID (total_tokens,)
        """
        return None

    def build_decode_view(
        self,
        layer_idx: int,
        q: torch.Tensor,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        *,
        num_heads: int,
        num_kv_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode 阶段的逻辑视图构建器。

        在 Attention.forward 的 decode 分支中，flash decode kernel 执行前调用。
        允许方法在 decode 时对 active_slots / req_indices / context_lens 做最终调整。

        参数:
          - layer_idx: 当前层索引
          - q: query 张量 (batch_size, num_heads, head_dim)
          - active_slots: SparseController.get_read_view() 返回的活跃 slot 索引
          - req_indices: 序列→行号映射
          - context_lens: 每个序列的 KV 可见长度
          - num_heads: Q 的头数
          - num_kv_heads: KV 的头数

        返回:
          (adjusted_active_slots, adjusted_req_indices, adjusted_context_lens)

        默认实现: 原样返回，不做调整。
        子类（如 DeltaKV）可能覆写以在 decode 时做特殊的 slot 重排或长度调整。
        """
        return active_slots, req_indices, context_lens

    # =========================================================================
    # 阶段 E: 空闲 Slot 查询 (Free Slot Queries)
    #     被 Scheduler 用于准入判断和抢占决策。
    #     不同方法有不同的"空闲"定义（vanilla 看全量池，deltakv 看多池）。
    # =========================================================================

    @property
    @abstractmethod
    def num_free_slots(self) -> int:
        """当前空闲的物理 slot 总数。

        这是 Scheduler 做准入判断的核心指标。每个子类的计算方式不同：
          - StandardCacheManager: 单一全量池中未分配的 slot 数
          - DeltaKVCacheManager: 综合考虑 full_pool + raw_pool + latent_pool 的余量
          - StreamingLLM/SnapKV: 同 vanilla（物理驱逐后 slot 直接回收）

        注意: 每个 slot 对应一层的一个 KV 位置。例如 num_free_slots=1000
        表示每层都有 1000 个空位可分配给新 token。
        """
        raise NotImplementedError

    def num_free_slots_full_layers(self) -> int:
        """Full-attention 层池中的空闲 slot 数。

        默认行为: 将 num_free_slots 视为唯一的池容量。
        DeltaKV 覆写此方法，暴露 full-attention pool 的独立容量——
        这个值决定了在不触发 thrashing 的前提下能准入多少长 prompt。
        因为 DeltaKV 中 sparse layer 可以压缩到 latent，其容量远大于 full layer，
        所以 full layer 池的容量才是真正的瓶颈。
        """
        return self.num_free_slots

    # =========================================================================
    # 阶段 F: Scheduler 准入控制接口 (Admission Control Interface)
    #     Scheduler 通过这些方法查询"能否准入新序列"和"需要多少资源"。
    #     不同稀疏方法可覆写这些方法以实现方法特定的准入策略。
    #     (例如 DeltaKV 为 full-attn layers 和 sparse layers 设置不同的预算)
    # =========================================================================

    def prefill_batched_tokens_margin(self) -> int:
        """Prefill 阶段的 token 数量安全裕度。

        Scheduler 在判断 batch 是否已满时，会在 max_num_batched_tokens 基础上
        减去此裕度值。防止 prefill batch 完全填满后导致下一轮无 token 预算处理 decode。

        默认返回 0（无额外裕度）。某些方法可能需要预留 token 预算给
        内部元数据处理（如 DeltaKV 的增量压缩需要额外 GPU 内存）。
        """
        return 0

    def remaining_prefill_tokens(self, seq: Sequence) -> int:
        """返回某序列还剩多少个 token 未 prefill。

        计算公式: num_prompt_tokens - num_prefilled_tokens
        被 Scheduler.schedule() 用于计算本轮 chunk 大小时的上限。

        某些方法可能覆写此方法以返回方法特定的"有效剩余长度"（例如
        考虑到压缩后 slot 复用可减少实际需要的 prefill 长度）。
        """
        return int(seq.num_prompt_tokens - seq.num_prefilled_tokens)

    def reserved_prefill_slots(self, waiting_seqs: deque[Sequence], chunk_prefill_size: int) -> int:
        """为 waiting 队列中正在进行 chunked prefill 的序列预留 slot 数。

        设计意图:
          如果一个长序列（比如 10k token）被切成了多个 chunk 处理，系统不能等它处理了一半，
          显存却被其他正在生成的请求（Decode）占满了。所以对于正在半路上的预填充请求，
          必须提前把它们剩下的 (总长度 - 已处理长度) 的槽位在逻辑账本上扣除（reserve），
          保证它能安稳毕业。

        计算逻辑:
          遍历 waiting 中所有序列，对满足 0 < num_prefilled_tokens < num_prompt_tokens
          (即正在进行 chunked prefill 但尚未完成) 的序列，累加其剩余 token 数。

        被 prompt_admission_budgets() 调用，用于扣减可用预算。
        也被 Scheduler 直接调用以计算 logical_free_count。
        """
        reserved = 0
        for seq in waiting_seqs:
            # 条件: 0 < 已处理token数 < 总提示词token数
            # 意味着序列既不是崭新的(=0, 还没开始 prefill)，也不是完全出炉的(>=总长, 已进入 decode)
            # 正处于被分块处理 (Chunked Prefill) 的中间态
            if 0 < seq.num_prefilled_tokens < seq.num_prompt_tokens:
                reserved += int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
        return reserved

    def prompt_admission_free_slots(self) -> int:
        """用于准入判断的空闲 slot 池大小。

        默认返回 num_free_slots。
        DeltaKV 覆写此方法，返回综合考虑 full-attn layer 容量后的可用 slot 数，
        因为 full-attn layer 的 KV 不可压缩，其容量才是真正的瓶颈。
        """
        return int(self.num_free_slots)

    def prompt_admission_cost(self, seq: Sequence) -> int:
        """新序列准入所需的 slot 成本。

        默认返回 seq.num_prompt_tokens（整个 prompt 的长度）。
        DeltaKV 等压缩方法可能覆写为更小的值，因为大部分 prompt token
        会被压缩为 latent，实际占用的 full slot 远小于 prompt 长度。
        """
        return int(seq.num_prompt_tokens)

    def prompt_logical_reservation_cost(self, seq: Sequence) -> int:
        """新序列准入后在逻辑账本上预留的 slot 数。

        与 prompt_admission_cost 的区别：admission_cost 用于 budgets 检查
        (方法特定预算)，logical_reservation_cost 用于 logical_free_count 扣减
        (通用 slot 计数)。大多数情况下两者相同，但 DeltaKV 等可能有差异——
        方法特定预算可能需要更复杂的计算（如区分 full-layer slots 和 latent slots）。
        """
        return int(seq.num_prompt_tokens)

    def prompt_admission_failure_action(self) -> str:
        """准入失败时的处理方式。

        - "defer": 将序列放回 waiting 末尾，等待下一轮调度（默认行为）
        - "raise": 直接抛 RuntimeError 终止推理

        "defer" 适用于 KV cache 可动态回收的场景（如 DeltaKV 压缩后释放空间），
        暂时无法准入不代表永远不行。
        "raise" 适用于显存确定不足的场景（如 prompt 本身就超过最大 KV cache 容量）。
        """
        return "defer"

    def prompt_admission_budgets(
        self,
        waiting_seqs: deque[Sequence],
        chunk_prefill_size: int,
    ) -> dict[str, int]:
        """返回 Scheduler 用于新序列准入的预算字典。（也是逻辑可用的,等同于logical_free_count）

        设计意图:
          为不同的资源维度提供独立的预算额度。默认只有一个维度 "slots"，但 DeltaKV
          等方法可能增加更多维度（如 "full_slots" 和 "latent_slots"），Scheduler 逐维度
          检查是否能容纳新序列。

        默认实现:
          budgets["slots"] = max(0, 当前空闲 slot - 为进行中 prefill 预留的 slot)
          即将预留后的余量作为新序列可用的准入预算。

        键值约定:
          "slots": 可分配给新序列的 slot 数量
        """
        reserved = int(self.reserved_prefill_slots(waiting_seqs, chunk_prefill_size))
        free_slots = int(self.prompt_admission_free_slots())
        return {"slots": max(0, free_slots - reserved)}

    def prompt_admission_costs(self, seq: Sequence) -> dict[str, int]:
        """返回新序列准入所需的各项成本。

        与 prompt_admission_budgets 对应，返回相同维度的成本字典。
        Scheduler 逐维度比较 costs[name] <= budgets[name] 来判断能否准入。

        默认: costs["slots"] = seq 的 prompt token 数
        """
        return {"slots": int(self.prompt_admission_cost(seq))}

    def on_prompt_admitted(self, seq: Sequence, costs: dict[str, int]):
        """新序列准入成功后的回调 Hook。

        Scheduler 在确认准入并扣减 budgets 后调用此方法。
        允许 CacheManager 做一些方法特定的初始化工作，例如：
          - DeltaKV: 为新序列预分配 center slots、初始化压缩器状态
          - 记录该序列的准入时间戳用于 TTFT 统计

        参数:
          - seq: 已准入的序列
          - costs: 准入时消耗的各项成本（与 prompt_admission_costs() 返回值一致）
        """
        return

    # =========================================================================
    # 阶段 G: 调试与诊断接口 (Debug & Diagnostics)
    # =========================================================================

    def free_slot_stats(self) -> dict[str, int]:
        """返回空闲 slot 的简要统计，用于日志和调试。

        被 ModelRunner.free_slots() 和 Scheduler 抢占日志使用。
        默认只返回 {"free_slots": num_free_slots}。
        DeltaKV 可能覆写为返回各分池的详细信息。
        """
        return {"free_slots": int(self.num_free_slots)}

    def debug_live_seq_slots(self) -> dict[int, int]:
        """返回当前活跃序列及其占用的 slot 数，用于 debug。

        返回 {seq_id: occupied_slot_count}。
        默认返回空字典（子类可覆写以提供实际数据）。
        在 Scheduler 准入失败 debug 日志中调用，帮助排查"哪些序列占着 slot 不放"。
        """
        return {}

    # =========================================================================
    # 阶段 H: 资源释放 (Resource Freeing)
    # =========================================================================

    @abstractmethod
    def free_seq(self, seq_id: int):
        """释放指定序列占用的所有 KV Cache 资源。

        遍历所有层，将该序列分配的所有物理 slot 标记为空闲，归还到 free_slots 队列。
        被 ModelRunner.free_slots() 调用，触发时机：
          - 序列完成 (FINISHED)：释放全部资源
          - 序列被抢占 (Preempted)：释放全部资源，序列将从头重新 prefill

        对于 DeltaKV：除了回收 full slots，还要回收 latent slots 和相关的 center 引用。
        """
        raise NotImplementedError

    @abstractmethod
    def free_part_slots(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        """部分释放某序列在指定层的 KV slot。

        与 free_seq (全量释放) 不同，此方法只释放不被 keep_indices 包含的 token 的 slot。
        用于 SnapKV / PyramidKV / StreamingLLM 的 eviction 操作：
          - 根据注意力分数选出 keep_indices (需要保留的 token 位置)
          - 对不在 keep_indices 中的 token，释放其物理 slot
          - 更新 slot map，将释放的位置设为 -1

        参数:
          - layer_idx: 层索引
          - seq: 目标序列
          - keep_indices: 要保留的 token 逻辑位置 (long tensor)
        """
        raise NotImplementedError

    # =========================================================================
    # 阶段 I: 内部准备方法 (Internal Prepare Methods)
    #     由 prepare_step() 转发，是子类必须实现的核心抽象。
    #     不同稀疏方法的 slot 分配策略差异主要体现在这里。
    # =========================================================================

    @abstractmethod
    def _prepare_prefill(self, seqs: list[Sequence]):
        """为 Prefill 阶段准备输入数据和 KV slot 分配。

        子类实现的核心逻辑:
          1. 遍历 seqs，为每个序列的当前 chunk 分配物理 slot
          2. 构造 slot_mapping: 每个 prefill token → 物理 slot ID
          3. 构造 input_ids: 当前 chunk 的 prompt token（展平为一维）
          4. 构造 positions: 对应的位置编码
          5. 构造 cu_seqlens_q: batch 各序列累积 token 数（用于 NestedTensor 布局）
          6. 更新序列的 metadata (req_to_token_slots)

        对于 DeltaKV:
          - 同时为 full-attn layers 和 sparse layers 分配 slot
          - sparse layer 的 slot 从 raw buffer 池中分配
          - 可能触发增量压缩 (如果 raw buffer 已满)

        返回: (input_ids, positions, cu_seqlens_q)
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_decode(self, seqs: list[Sequence]):
        """为 Decode 阶段准备输入数据和 KV slot 分配。

        子类实现的核心逻辑:
          1. 为每个序列分配 1 个新的物理 slot（存放新生成的 token 的 KV）
          2. 构造 slot_mapping: 每个 decode token → 物理 slot ID
          3. input_ids = [seq.last_token for seq in seqs]
          4. positions = [seq.num_tokens - 1 for seq in seqs] (当前绝对位置)
          5. 更新序列的 metadata

        与 _prepare_prefill 的主要区别:
          - 每个序列只分配 1 个 slot（而非 chunk_size 个）
          - 不需要 cu_seqlens_q（decode 是单 token attention，非 NestedTensor）
          - input_ids 来自 last_token（上一轮生成的 token）

        对于 DeltaKV:
          - decode 时仍需为 full layers 和 sparse layers 各分配 1 个 slot
          - 新 token 先以全精度写入 raw buffer，后续 buffer 溢出时增量压缩

        返回: (input_ids, positions, None)
        """
        raise NotImplementedError

    def get_compressed_lens(self, req_indices):
        """获取指定序列的已压缩历史长度。

        用于 OmniKV/DeltaKV 中计算观察层的有效搜索范围。
        DeltaKV 覆写此方法，返回 row_deltakv_compressed_lens[row]，
        即 sink 之后已经被压缩为 latent 的历史 token 数。
        观察层只需要从这个范围中选 top-k（未被压缩的 sink+recent+buffer 不需要重新选择）。

        默认抛出 NotImplementedError（非压缩方法不需要此功能）。
        """
        raise NotImplementedError
