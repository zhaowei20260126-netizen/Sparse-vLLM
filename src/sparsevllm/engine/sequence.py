import torch
from copy import copy
from enum import Enum, auto
from itertools import count

from sparsevllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto() # 生成永不重复的数字
    RUNNING = auto()
    FINISHED = auto()


# 一部分属性会在 scheduler 动态修改
class Sequence:
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)  # 全局唯一的序列ID
        self.status = SequenceStatus.WAITING  # 初始状态设为等待调度（还未扔进GPU）
        self.token_ids = copy(token_ids)  # 所有 token（prompt + 已生成的 completion）
        self.last_token = token_ids[-1] if token_ids else None  # 记录当前最后一个token
        self.num_tokens = len(self.token_ids)  # 当前序列包含的总token数（随着生成会不断增加）
        self.num_prompt_tokens = len(self.token_ids)  # 初始提示词(Prompt)的长度，这是一个固定的快照值
        self.num_prefilled_tokens = 0  # 已经过GPU计算(prefill)的token数，用于统计分块(Chunk)预填充的进度
        self.current_chunk_size = None  # 本次调度中，喂给GPU的prefill分块(chunk)的具体大小

        self.temperature = sampling_params.temperature  # 采样温度，控制生成文本的随机性
        self.max_tokens = sampling_params.max_tokens  # 允许该序列生成的最大token数，超出则截断
        self.ignore_eos = sampling_params.ignore_eos  # 是否忽略文本结束符(EOS)，直到跑到max_tokens才停止

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, i):
        return self.token_ids[i]

    @property
    def kv_change_state(self):
        if self.num_prefilled_tokens == 0:
            return 'first_prefill' # 序列的首次 prefill
        elif self.num_prefilled_tokens < self.num_prompt_tokens:
            return 'prefill' # 后续 chunk prefill
        elif self.num_prefilled_tokens == self.num_prompt_tokens:
            return 'decode' # 正在逐 token 解码

        raise ValueError

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def is_last_chunk_prefill(self):
        '''本轮是否是 prefill 的最后一个 chunk。Scheduler 调度时先设 `current_chunk_size`，然后 SparseController 据此判断是否触发最终驱逐'''
        return (self.num_prefilled_tokens + self.current_chunk_size) >= self.num_prompt_tokens

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        # 优化 IPC：不发送 slot_mapping，只发送元数据和必要的 token
        if self.num_completion_tokens == 0:
            chunk_size = self.current_chunk_size if self.current_chunk_size is not None else self.num_prompt_tokens
            data = self.token_ids[self.num_prefilled_tokens : self.num_prefilled_tokens + chunk_size]
        else:
            data = self.last_token

        return (
            self.seq_id,
            self.status,
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_prefilled_tokens,
            self.current_chunk_size,
            self.temperature,
            self.max_tokens,
            self.ignore_eos,
            data,
        )

    def __setstate__(self, state):
        (self.seq_id, self.status, self.num_tokens, self.num_prompt_tokens,
         self.num_prefilled_tokens, self.current_chunk_size, self.temperature,
         self.max_tokens, self.ignore_eos, data) = state

        if self.num_completion_tokens == 0:
            self.token_ids = data
            self.last_token = self.token_ids[-1] if self.token_ids else None
        else:
            self.last_token = data
            self.token_ids = []
