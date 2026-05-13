import math
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def reverse_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """对已应用 RoPE 的向量执行逆操作，恢复到位置无关状态。
    
    RoPE 公式:     y1 = x1*cos - x2*sin,  y2 = x2*cos + x1*sin
    De-RoPE 公式:  x1 = y1*cos + y2*sin,  x2 = y2*cos - y1*sin
    """
    y1, y2 = torch.chunk(x.float(), 2, dim=-1)
    x1 = y1 * cos + y2 * sin
    x2 = y2 * cos - y1 * sin
    return torch.cat((x1, x2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = _compute_inv_freq(rotary_dim, base, rope_scaling)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


def _compute_inv_freq(
    rotary_dim: int,
    base: float,
    rope_scaling: dict | None,
) -> torch.Tensor:
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
    if rope_scaling is None:
        return inv_freq

    rope_type = str(
        rope_scaling.get("rope_type", rope_scaling.get("type", ""))
    ).lower()
    if rope_type in ("", "default", "none"):
        return inv_freq

    factor = float(rope_scaling.get("factor", 1.0))
    if rope_type == "linear":
        return inv_freq / factor

    if rope_type == "llama3":
        low_freq_factor = float(rope_scaling.get("low_freq_factor", 1.0))
        high_freq_factor = float(rope_scaling.get("high_freq_factor", 4.0))
        old_context_len = float(
            rope_scaling.get("original_max_position_embeddings", 8192)
        )
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        wavelen = 2 * math.pi / inv_freq

        scaled_inv_freq = torch.where(
            wavelen > low_freq_wavelen,
            inv_freq / factor,
            inv_freq,
        )
        smooth = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (1 - smooth) * (inv_freq / factor) + smooth * inv_freq
        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        return torch.where(is_medium_freq, smoothed_inv_freq, scaled_inv_freq)

    raise NotImplementedError(f"Unsupported rope_scaling type: {rope_type}")


_ROPE_CACHE = {}


def _freeze_rope_scaling(rope_scaling: dict | None):
    if rope_scaling is None:
        return None
    return tuple(sorted((k, str(v)) for k, v in rope_scaling.items()))


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    key = (head_size, rotary_dim, max_position, float(base), _freeze_rope_scaling(rope_scaling))
    rotary_emb = _ROPE_CACHE.get(key)
    if rotary_emb is None:
        rotary_emb = RotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            rope_scaling=rope_scaling,
        )
        _ROPE_CACHE[key] = rotary_emb
    return rotary_emb
