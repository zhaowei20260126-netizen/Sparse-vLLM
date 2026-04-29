from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64 # 限制模型最多生成多少个token
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature >= 0.0, "temperature must be non-negative"
