import os
from collections import deque

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence, SequenceStatus
from sparsevllm.engine.cache_manager import CacheManager
from sparsevllm.utils.log import logger


class Scheduler:
    """
    请求调度器，负责管理待处理 (waiting) 和正在运行 (running) 的序列。
    主要职责：
    1. 决定每一轮 (step) GPU 应该处理哪些序列。
    2. 实现分块 Prefill (Chunked Prefill) 以处理长序列。
    3. 管理逻辑显存额度，并在显存不足时触发抢占 (Preemption/Eviction)。
    """

    # 初始化调度器：配置参数、显存管理器、waiting/decoding 双队列
    def __init__(self, config: Config, memory_oracle: CacheManager):
        # ============ 调度策略参数 ============
        self.config = config
        
        # 单个 batch 的容量限制
        self.max_num_seqs_in_batch = config.max_num_seqs_in_batch  #  batch 最多容纳的序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens  #  batch 最多容纳的 token 总数
        logger.debug(f'set max_num_batched_tokens = {config.max_num_batched_tokens} in Scheduler')
        self.max_decoding_seqs = config.max_decoding_seqs  # 同时进行 decode 的最多序列数（decoding 队列最大长度）
        
        # Prefill 分块参数：长序列会分成多个 chunk 逐步处理，每个 chunk 最大大小
        self.chunk_prefill_size = config.chunk_prefill_size  # 默认 4096，用于分块 prefill
        
        # 生成终止条件
        self.eos = config.eos  # End-Of-Sequence token ID，生成到此 token 时停止
        
        # 稀疏方法的阈值参数（用于判断"长文本"和"短文本"的分界线）
        self.num_sink_tokens = config.num_sink_tokens  # StreamingLLM 风格保留的开头 token 数
        self.num_recent_tokens = config.num_recent_tokens  # 保留的最近 token 数
        self.num_top_tokens = config.num_top_tokens  # 保留的 attention 权重最高 token 数

        # ============ 显存管理（memory_oracle）============
        # memory_oracle 引用 Rank 0 的 CacheManager，作为全局显存余量参考。
        # 主要职责：
        #   - 提供剩余槽位数（num_free_slots）
        #   - 估算新序列的显存成本（prompt_admission_costs）
        #   - 决策是否可以准入新序列
        self.memory_oracle = memory_oracle
        
        # ============ 序列队列管理 ============
        # waiting 队列：存放等待 Prefill 的序列
        # 包括：
        #   1. 新加入系统的序列（通过 add_request() 提交）
        #   2. Prefill 未完成的序列（分块 Prefill，下一个 chunk 在下一轮调度）
        #   3. 被驱逐的序列（显存不足时被强制停止，重置为 WAITING 等待重新 Prefill）
        self.waiting: deque[Sequence] = deque()
        
        # decoding 队列：存放正在进行增量生成（Decode）的序列
        # 包括：
        #   1. 已完成全部 Prefill，进入生成阶段的序列
        #   2. 等待被调度进行下一步 Decode 的序列
        # 当序列生成到 EOS 或达到 max_tokens 时，会被标记为 FINISHED 并从此队列移除
        self.decoding: deque[Sequence] = deque()
        
        # 用于避免重复警告：记录已经延迟准入的序列 ID，避免日志爆炸
        self._admission_defer_warned_seq_ids: set[int] = set()

    # 根据稀疏方法和阶段（prefill/decode）计算区分长短文本的长度阈值
    def _long_text_threshold(self, is_prefill: bool) -> int:
        """
        根据稀疏方法和处理阶段，计算区分"长文本"和"短文本"的阈值。
        
        参数：
            is_prefill: True 表示 Prefill 阶段，False 表示 Decode 阶段
        
        返回值：序列长度的阈值，超过此值被视为"长文本"
        
        设计理念：
        - 不同的稀疏方法（vanilla、snapkv、deltakv）保留的 token 数不同
        - Prefill 需要额外加上 chunk_prefill_size（分块大小），因为长 prompt 会分块处理
        - Decode 不需要加 chunk_prefill_size，只看已生成的 token 数
        """
        if self.config.vllm_sparse_method == "deltakv-snapkv":
            base = (
                self.num_sink_tokens
                + self.num_recent_tokens
                + self.config.snapkv_window_size
            )
        elif self.config.vllm_sparse_method == "deltakv-standalone":
            base = self.num_sink_tokens + self.num_recent_tokens
        elif self.config.vllm_sparse_method in ("streamingllm", "attention-sink", "attention_sink"):
            base = self.num_sink_tokens + self.num_recent_tokens
        else:
            base = self.num_sink_tokens + self.num_top_tokens + self.num_recent_tokens
        return base + (self.chunk_prefill_size if is_prefill else 0)

    # 判断序列是否超过长短文本分界阈值
    def _is_long_text(self, seq: Sequence, is_prefill: bool) -> bool:
        """判断一个序列是否属于"长文本"分类"""
        threshold = self._long_text_threshold(is_prefill)
        seq_len = seq.num_prompt_tokens if is_prefill else seq.num_tokens
        return int(seq_len) > int(threshold)

    # 从 waiting 队列指定位置弹出序列，0 为头部
    def _pop_waiting_at(self, idx: int) -> Sequence:
        """从 waiting 队列的任意位置弹出一个序列（idx 表示位置，0 表示头部）"""
        if idx == 0:
            return self.waiting.popleft()
        self.waiting.rotate(-idx)
        seq = self.waiting.popleft()
        self.waiting.rotate(idx)
        return seq

    # 按长短分类从 waiting 队列中取出下一个 prefill 序列
    def _pop_next_prefill_seq(self, target_is_long: bool) -> Sequence | None:
        """从 waiting 队列中寻找符合长度分类（长或短）的下一个 prefill 序列"""
        if not self.waiting:
            return None
        for idx, seq in enumerate(self.waiting):
            if self._is_long_text(seq, is_prefill=True) == target_is_long:
                return self._pop_waiting_at(idx)
        return None

    # 从 decoding 队列指定位置弹出序列
    def _pop_decoding_at(self, idx: int) -> Sequence:
        """从 decoding 队列的任意位置弹出一个序列"""
        if idx == 0:
            return self.decoding.popleft()
        self.decoding.rotate(-idx)
        seq = self.decoding.popleft()
        self.decoding.rotate(idx)
        return seq

    # 按长短分类从 decoding 队列中取出下一个 decode 序列
    def _pop_next_decoding_seq(self, target_is_long: bool) -> Sequence | None:
        """从 decoding 队列中寻找符合长度分类（长或短）的下一个 decode 序列"""
        if not self.decoding:
            return None
        for idx, seq in enumerate(self.decoding):
            if self._is_long_text(seq, is_prefill=False) == target_is_long:
                return self._pop_decoding_at(idx)
        return None

    # 判断 waiting 和 decoding 是否均为空（所有请求处理完毕）
    def is_finished(self):
        """判断所有请求是否已处理完成（waiting 和 decoding 队列都为空）"""
        return len(self.waiting) == 0 and len(self.decoding) == 0

    # 将一个新序列加入 waiting 队列等待调度
    def add(self, seq: Sequence):
        """将新请求加入等待队列"""
        self.waiting.append(seq)
    
    # 查询 memory_oracle 为 waiting 中所有 chunked prefill 预留的槽位数
    def _reserved_prefill_tokens(self) -> int:
        """计算为 waiting 队列中所有序列的 prefill 预留的槽位数"""
        return int(self.memory_oracle.reserved_prefill_slots(self.waiting, self.chunk_prefill_size))

    # 显存不足时抛出 RuntimeError，附带详细的预算诊断信息
    def _raise_prompt_admission_failure(
        self,
        seq: Sequence,
        failed_budget: str,
        need: int,
        free: int,
        *,
        physical_free_count: int,
        reserved_prefill: int,
        logical_free_count: int,
        admission_budgets: dict[str, int],
    ):
        """
        抛出异常：新序列的 prompt 无法准入（显存不足）。
        用于记录详细的显存不足信息，便于调试。
        """
        raise RuntimeError(
            "Insufficient KV cache slots to admit prompt. "
            f"cache_manager={type(self.memory_oracle).__name__} prompt_len={seq.num_prompt_tokens} "
            f"failed_budget={failed_budget} need={need} free={free} budgets={admission_budgets} "
            f"free_slots={physical_free_count} reserved_prefill={reserved_prefill} "
            f"logical_free={logical_free_count}"
        )

    # 核心调度：prefill 优先，decode 备选，显存不足时触发驱逐
    def schedule(self) -> tuple[list[Sequence], bool, list[Sequence]]:
        """
        核心调度逻辑。每一轮 step 调用一次
        返回：(本次要运行的序列列表, 是否是 Prefill 阶段, 本次被抢占的序列列表)
        
        注意：目前为了简化算子实现，单次 step 不支持 Prefill 和 Decode 混合。
        
        核心决策流程：
        1. 阶段1（Prefill 优先）：如果 waiting 队列有序列，优先调度 Prefill
        2. 阶段2（Decode 备选）：仅当 waiting 为空时，调度 Decode 序列
        3. 显存管理：通过 memory_oracle 判断是否有显存容纳新序列
        4. 驱逐策略：显存不足时驱逐序列，释放其占用的槽位
        """
        # 本轮调度结果
        scheduled_seqs = []  # 本轮要执行的序列列表
        preempted_seqs = []  # 本轮被驱逐的序列列表
        
        # 本轮 batch 填充程度的计数器
        num_batched_seqs = 0  # 当前已选中的序列数
        num_batched_tokens = 0  # 当前已选中的 token 总数（仅用于 prefill）
        
        # ============ 显存预算估算（逻辑空间） ============
        # 调度器需要估算：本轮能否容纳更多序列或 token，而不超过显存限制
        
        # 物理显存槽位数（硬指标）
        physical_free_count = self.memory_oracle.num_free_slots  # GPU 上还有多少个空槽位(每一层都有那么多)
        
        # 逻辑显存预留（软指标）：为待处理的 prefill 请求预留一些槽位
        reserved_prefill = self._reserved_prefill_tokens() 
        
        # 本轮实际可用的逻辑显存
        logical_free_count = max(0, physical_free_count - reserved_prefill)  # 扣除预留后的可用显存
        
        # 初始值：本轮 prefill 还能消耗多少物理槽位
        step_free_count = physical_free_count
        
        # 每种预算类别的可用额度（比如"所有 token"、"某个长度段"等）
        admission_budgets = dict(
            self.memory_oracle.prompt_admission_budgets(self.waiting, self.chunk_prefill_size)
        )
        
        # prefill 阶段的 token 数量裕度（为了避免 batched_tokens 完全填满），默认为0
        margin_batched_tokens = self.memory_oracle.prefill_batched_tokens_margin()
        
        # 如果所有 prefill 都被延迟，记录第一个被延迟的序列（用于死锁检测）
        deferred_prompt_failure: tuple[Sequence, str, int, int] | None = None

        # --- 阶段 1: Prefill 调度 ---
        # 只要 waiting 队列有活，就优先处理 Prefill，因为它是计算密集型的。
        # 按照"长文本优先"或"短文本优先"的顺序扫描 waiting 队列
        prefill_bucket_order: list[bool] = []
        if self.waiting:
            # 确定第一个桶的类型（长文本或短文本），然后交替扫描两个桶
            first_bucket = self._is_long_text(self.waiting[0], is_prefill=True)
            prefill_bucket_order.append(first_bucket)
            prefill_bucket_order.append(not first_bucket)

        for target_is_long in prefill_bucket_order:
            if scheduled_seqs:
                # 一旦某个桶调中了序列，立刻停止扫描
                break
            bucket_scan_budget = len(self.waiting)  # 单个桶内最多扫描多少个序列（防止无限扫描）
            while (
                self.waiting  # 还有序列等待 prefill
                and bucket_scan_budget > 0  # 还有扫描预算
                and step_free_count > 0  # 还有物理显存
                and num_batched_tokens <= self.max_num_batched_tokens - margin_batched_tokens  #  单 batch 的 已调token 总量不超过 max_num_batched_tokens (减去 margin 留余地)
                and num_batched_seqs < self.max_num_seqs_in_batch  # 单 batch 的序列数不超过 max_num_seqs_in_batch
                and len(self.decoding) < self.max_decoding_seqs  # decode 队列未满
            ):
                # 从 waiting 队列中弹出下一个符合长度分类的序列
                seq = self._pop_next_prefill_seq(target_is_long)
                if seq is None:
                    # 当前桶中没有更多符合条件的序列
                    break
                bucket_scan_budget -= 1
                
                # 计算该序列还剩多少 prefill token 未处理
                remaining_prefill_tokens = self.memory_oracle.remaining_prefill_tokens(seq) # seq.num_prompt_tokens - seq.num_prefilled_tokens

                # 异常处理：如果由于某种原因已经 prefill 完却还在 waiting 队列
                if remaining_prefill_tokens <= 0:
                    raise ValueError('BUG：理论上不应该在 waiting 里')

                # 确定本次 Chunk 的大小：取多个限制条件的最小值
                can_prefill_tokens = min(
                    remaining_prefill_tokens,  # 该序列剩余的 prefill token 数
                    self.chunk_prefill_size,  # prefill 每次最多处理的 chunk 大小
                    self.max_num_batched_tokens - num_batched_tokens,  # 本轮 batch 还能容纳的 token 数
                    step_free_count,  # 物理显存还剩多少
                )

                if can_prefill_tokens <= 0:
                    # 无法再调度 prefill，将序列重新放回 waiting 队列头部
                    logger.debug(f'{can_prefill_tokens=} 结束 schedule prefill 请求')
                    self.waiting.appendleft(seq)
                    break

                # ============ 逻辑显存分配检查（新序列准入） ============
                # 如果是新序列的起始（num_prefilled_tokens == 0），需要预先检查是否能容纳整个 prompt
                # 策略：保守预留整个 prompt 长度的显存，即使后续可能会有稀疏逐出
                if seq.num_prefilled_tokens == 0:
                    # 向 memory_oracle 查询新序列需要的各类显存成本（slots:num_prompt_tokens）
                    costs = self.memory_oracle.prompt_admission_costs(seq)
                    failed = None
                    for name, need in costs.items():
                        # 检查该预算类别是否还有足够空间
                        free = int(admission_budgets.get(name, 0) or 0)
                        if free < int(need):
                            # 该预算类别不足，记录失败原因
                            failed = (name, int(need), free)
                            break
                    if failed is not None:
                        # 显存不足，根据配置决定是"延迟"还是"拒绝"
                        action = self.memory_oracle.prompt_admission_failure_action()
                        name, need, free = failed
                        if action == "defer":
                            # 延迟此序列的准入，继续处理下一个
                            if deferred_prompt_failure is None:
                                deferred_prompt_failure = (seq, name, need, free) # 如(seq_B, "slots",   8000, 7000) 
                            # 打印 WARNING 日志（每个 seq_id 只打印一次）
                            if seq.seq_id not in self._admission_defer_warned_seq_ids:
                                logger.warning(
                                    "Prompt admission deferred because the current batch/KV budget is saturated. "
                                    f"seq_id={seq.seq_id} prompt_len={seq.num_prompt_tokens} "
                                    f"failed_budget={name} need={need} free={free} "
                                    f"waiting={len(self.waiting) + 1} decoding={len(self.decoding)} "
                                    f"scheduled_prefill={len(scheduled_seqs)} free_slots={physical_free_count} "
                                    f"reserved_prefill={reserved_prefill}. "
                                    "This usually means batch size is too large for the current KV budget."
                                )
                                if os.getenv("SPARSEVLLM_DEBUG_SLOTS", "0") == "1" and len(self.decoding) == 0:
                                    live_seq_slots = self.memory_oracle.debug_live_seq_slots()
                                    live_seq_items = sorted(
                                        ((int(seq_id), int(n_slots)) for seq_id, n_slots in live_seq_slots.items()),
                                        key=lambda x: (-x[1], x[0]),
                                    )[:16]
                                    waiting_seq_ids_all = [int(s.seq_id) for s in self.waiting]
                                    decoding_seq_ids_all = [int(s.seq_id) for s in self.decoding]
                                    scheduled_seq_ids_all = [int(s.seq_id) for s in scheduled_seqs]
                                    known_seq_ids = (
                                        set(waiting_seq_ids_all)
                                        | set(decoding_seq_ids_all)
                                        | set(scheduled_seq_ids_all)
                                    )
                                    zombie_seq_ids = sorted(
                                        int(seq_id)
                                        for seq_id in live_seq_slots
                                        if int(seq_id) not in known_seq_ids
                                    )[:16]
                                    logger.info(
                                        "defer_with_no_decoding seq_id={} need={} free={} free_slots={} reserved_prefill={} "
                                        "scheduled_prefill={} waiting_seq_ids={} scheduled_seq_ids={} zombie_seq_ids={} "
                                        "live_seq_slots={}",
                                        seq.seq_id,
                                        int(need),
                                        int(free),
                                        int(physical_free_count),
                                        int(reserved_prefill),
                                        len(scheduled_seqs),
                                        waiting_seq_ids_all[:16],
                                        scheduled_seq_ids_all[:16],
                                        zombie_seq_ids,
                                        live_seq_items,
                                    )
                                self._admission_defer_warned_seq_ids.add(seq.seq_id)
                            # 将序列重新加入 waiting 队列末尾，继续下一个序列
                            self.waiting.append(seq)
                            continue
                        # 否则拒绝准入，抛出异常
                        self._raise_prompt_admission_failure(
                            seq,
                            name,
                            need,
                            free,
                            physical_free_count=physical_free_count,
                            reserved_prefill=reserved_prefill,
                            logical_free_count=logical_free_count,
                            admission_budgets=admission_budgets,
                        )
                    
                    # 准入成功：更新预算
                    self._admission_defer_warned_seq_ids.discard(seq.seq_id)
                    for name, need in costs.items():
                        admission_budgets[name] = int(admission_budgets.get(name, 0) or 0) - int(need)
                    # 通知 memory_oracle 该序列已准入
                    self.memory_oracle.on_prompt_admitted(seq, costs) # 新序列准入成功后的回调 Hook(标准版cache，无)
                    # 检查逻辑显存是否真的足够
                    logical_need = self.memory_oracle.prompt_logical_reservation_cost(seq)
                    if logical_free_count < logical_need:
                        # Fail fast: admission budgets should already account for reserved prefill headroom.
                        # Reaching this branch usually means a cache-manager-specific budget mismatch.
                        raise RuntimeError(
                            "Prompt admission budget mismatch after reservation check. "
                            f"cache_manager={type(self.memory_oracle).__name__} prompt_len={seq.num_prompt_tokens} "
                            f"logical_need={logical_need} logical_free={logical_free_count} "
                            f"budgets={admission_budgets} costs={costs} "
                            f"free_slots={physical_free_count} reserved_prefill={reserved_prefill}"
                        )
                    logical_free_count -= int(logical_need)

                # 成功调度这个 prefill 块
                logger.debug(f'Add chunk prefill with {can_prefill_tokens} tokens.')
                seq.current_chunk_size = can_prefill_tokens  # 记录本轮 prefill 的 chunk 大小
                num_batched_seqs += 1
                num_batched_tokens += can_prefill_tokens
                step_free_count -= can_prefill_tokens # 真实的剩余物理显存
                seq.status = SequenceStatus.RUNNING  # 标记为运行状态
                scheduled_seqs.append(seq)

        # 如果有 Prefill 请求被选中，直接返回，本次 step 只跑 Prefill。
        if scheduled_seqs:
            return scheduled_seqs, True, []

        # --- 阶段 2: Decode 调度 ---
        # 只有在没有 Prefill 任务时才处理增量生成任务。
        # Decode 优先短序列：如果当前 decoding 队列里存在 short，则本轮只调度 short；
        # 仅当全部都是 long 时，才调度 long。（避免 short 被 long 淹没）
        
        # 判断 decoding 队列中是否有"短序列"
        if self.decoding:
            decode_threshold = self._long_text_threshold(is_prefill=False)
            has_short_decode = any(int(seq.num_tokens) <= int(decode_threshold) for seq in self.decoding) # 短序列优先
            target_is_long_decode = not has_short_decode  # 有短序列则只调度短序列，否则调度长序列
        else:
            target_is_long_decode = False
        
        # 从 decoding 队列中逐个选择序列进行增量生成
        while self.decoding and num_batched_seqs < self.max_num_seqs_in_batch:
            seq = self._pop_next_decoding_seq(target_is_long_decode)
            if seq is None:
                # 没有符合条件的序列
                break

            # 检查逻辑空间是否够塞下一个新 Token (Decode step)
            if logical_free_count < 1:
                # 显存耗尽，触发驱逐/抢占逻辑
                # 策略：牺牲当前 seq，并立刻返回，让上层先释放槽位再进入下一轮调度。
                # 这样可以避免在一次 schedule() 调用中反复驱逐多个请求造成抖动。
                victim = seq
                debug_slots = os.getenv("SPARSEVLLM_DEBUG_SLOTS", "0") == "1"
                if debug_slots:
                    logger.info(
                        "preempt seq_id={} prompt_len={} num_tokens={} prefetched={} free_slots_before={} waiting_before={} decoding_before={}",
                        victim.seq_id,
                        int(victim.num_prompt_tokens),
                        int(victim.num_tokens),
                        int(victim.num_prefilled_tokens),
                        int(self.memory_oracle.num_free_slots),
                        len(self.waiting),
                        len(self.decoding),
                    )
                # 将被害者标记为 WAITING，重置其 prefill 进度，让其重新开始 prefill
                victim.status = SequenceStatus.WAITING
                victim.num_prefilled_tokens = 0  # 重置进度，下次回来重新跑 Prefill
                # Requeue to the tail instead of the head. Otherwise a long sequence can
                # be immediately re-admitted after preemption and thrash in a tight
                # prefill->decode->preempt loop while other waiting prompts never drain.
                self.waiting.append(victim)  # 加入 waiting 队列末尾
                # Any decode sequences already popped into `scheduled_seqs` in this round
                # have not been executed yet. Put them back before returning, otherwise
                # they disappear from scheduler queues while still occupying KV slots.
                if scheduled_seqs:
                    self.decoding.extendleft(reversed(scheduled_seqs))  # 驱逐发生时，把"本轮已经从队列里取出、但还没来得及执行"的 decode 序列放回去，防止它们丢失。这是一个回滚操作。
                    scheduled_seqs.clear()
                preempted_seqs.append(victim)
                logger.warning(f'驱逐请求 id = {victim.seq_id} | slots={self.memory_oracle.free_slot_stats()}')
                return [], False, preempted_seqs  # 返回驱逐信息，让 LLMEngine 处理
            else:
                # 逻辑空间充足，继续调度该序列
                logical_free_count -= 1  # 为下一个 decode token 预留 1 个位子
                num_batched_seqs += 1
                scheduled_seqs.append(seq)
                # logger.debug('Add a decode req.')
        
        # 死锁检测，无调中 + 全部defer + 无decode → RuntimeError
        if not scheduled_seqs:
            # 检查是否存在"所有 prefill 都被延迟导致死锁"的情况
            if deferred_prompt_failure is not None and not self.decoding:
                seq, name, need, free = deferred_prompt_failure
                if os.getenv("SPARSEVLLM_DEBUG_SLOTS", "0") == "1":
                    waiting_seq_ids_all = [int(s.seq_id) for s in self.waiting]
                    decoding_seq_ids_all = [int(s.seq_id) for s in self.decoding]
                    scheduled_seq_ids_all = [int(s.seq_id) for s in scheduled_seqs]
                    live_seq_slots = self.memory_oracle.debug_live_seq_slots()
                    live_seq_items = sorted(
                        ((int(seq_id), int(n_slots)) for seq_id, n_slots in live_seq_slots.items()),
                        key=lambda x: (-x[1], x[0]),
                    )[:16]
                    waiting_prompt_lens = [int(s.num_prompt_tokens) for s in list(self.waiting)[:8]]
                    known_seq_ids = (
                        set(waiting_seq_ids_all)
                        | set(decoding_seq_ids_all)
                        | set(scheduled_seq_ids_all)
                    )
                    zombie_seq_ids = sorted(
                        int(seq_id)
                        for seq_id in live_seq_slots
                        if int(seq_id) not in known_seq_ids
                    )[:16]
                    logger.info(
                        "deferred_deadlock seq_id={} failed_budget={} need={} free={} free_slots={} reserved_prefill={} "
                        "waiting_prompt_lens={} waiting_seq_ids={} decoding_seq_ids={} scheduled_seq_ids={} "
                        "zombie_seq_ids={} live_seq_slots={}",
                        seq.seq_id,
                        name,
                        int(need),
                        int(free),
                        int(physical_free_count),
                        int(reserved_prefill),
                        waiting_prompt_lens,
                        waiting_seq_ids_all[:16],
                        decoding_seq_ids_all[:16],
                        scheduled_seq_ids_all[:16],
                        zombie_seq_ids,
                        live_seq_items,
                    )
                raise RuntimeError(
                    "All prompt admissions were deferred and no runnable work remains. "
                    f"cache_manager={type(self.memory_oracle).__name__} seq_id={seq.seq_id} "
                    f"prompt_len={seq.num_prompt_tokens} failed_budget={name} need={need} free={free} "
                    f"free_slots={physical_free_count} reserved_prefill={reserved_prefill} "
                    f"waiting={len(self.waiting)} decoding={len(self.decoding)}. "
                    "Reduce batch size/max_num_seqs_in_batch/max_num_batched_tokens, "
                    "or shorten the prompt / generation budget."
                )
            return [], False, preempted_seqs
            
        # 将被选中的 Decode 序列放回 running 队列以保持顺序
        self.decoding.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False, preempted_seqs

    # 模型执行后的后处理：更新 token 序列、prefill 进度、在队列间转移或标记完成
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """
        模型运行后的后处理工作。
        主要职责：
        1. 更新序列的 Token 序列（completion_token_ids）
        2. 更新 Prefill 进度（num_prefilled_tokens）
        3. 处理序列完成状态（检查 EOS 或 max_tokens）
        4. 在 waiting/decoding 队列间转移序列
        """
        if is_prefill:
            # ============ Prefill 后处理 ============
            for seq, token_id in zip(seqs, token_ids):
                # 更新该序列的 prefill 进度
                seq.num_prefilled_tokens += seq.current_chunk_size
                
                # 检查 Chunked Prefill 是否完成
                if seq.num_prefilled_tokens < seq.num_prompt_tokens:
                    # 这个序列的 prefill 还未完成，需要继续下一个 chunk
                    # 将其放回 waiting 队列头部，下一轮调度会继续处理这个序列
                    seq.status = SequenceStatus.WAITING
                    self.waiting.appendleft(seq)
                else:
                    # Prefill 彻底结束，该序列可以进入 Decode 阶段了
                    seq.status = SequenceStatus.RUNNING
                    self.decoding.append(seq)
                    # 记录模型生成的第一个 Token（Prefill 阶段产生的首 token）
                    seq.append_token(token_id)
                    # 检查是否已经命中结束条件
                    if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                        # 如果首个 token 就是 EOS 或已达 max_tokens，标记为完成
                        seq.status = SequenceStatus.FINISHED
                        self.decoding.remove(seq)  # 立即从 decoding 移除
            return

        # ============ Decode 后处理 ============
        # 处理 Decode 步骤的输出
        for seq, token_id in zip(seqs, token_ids):
            # 将新生成的 token 追加到序列
            seq.append_token(token_id)
            
            # 检查是否达成生成终止条件
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                # 序列已完成
                seq.status = SequenceStatus.FINISHED
                if seq in self.decoding:
                    self.decoding.remove(seq)  # 从 decoding 队列移除
