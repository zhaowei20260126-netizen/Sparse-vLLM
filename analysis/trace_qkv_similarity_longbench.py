#!/usr/bin/env python3
"""Trace Q/K/V similarity on LongBench-style prompts.

这个版本刻意写得接近 `src/deltakv/analysis` 里的分析脚本风格：

1. `QKVCollector` 只负责在 forward 过程中收集当前 step 的 Q/K/V。
2. `patch_model_attention()` 给每层 `self_attn.forward` 套一个轻量 wrapper。
3. `consume_forward()` 在每次 prefill/decode 后立刻统计相似性。
4. `write_outputs()` 把 summary/csv/heatmap 写到输出目录。

核心 shape 约定：

    hidden_states: [B, T, hidden_size]
    q:             [B, q_heads,  T, head_dim]
    k/v:           [B, kv_heads, T, head_dim]
    recorded:      [heads, sampled_T, head_dim]

脚本只用于离线 Stage-0 现象验证，不修改 Sparse-vLLM 推理主路径。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOT = True
except Exception:
    HAS_PLOT = False


KINDS = ("q", "k", "v")
VIEWS = ("pre_rope", "post_rope")
DEFAULT_DISTANCE_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str | None, *, positive: bool = False, sorted_unique: bool = False) -> list[int]:
    if not value:
        return []
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if positive:
        values = [v for v in values if v > 0]
    if sorted_unique:
        values = sorted(set(values))
    return values


def safe_name(path_or_name: str) -> str:
    text = Path(path_or_name).name or path_or_name
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def model_input_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1)


def cosine_last_dim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """只在最后一维算 cosine，其它维度必须已对齐。

    例子:
        a/b: [heads, tokens, head_dim] -> [heads, tokens]
        a/b: [heads, head_dim] -> [heads]
    """
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reshape_rope_tensor(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """把 cos/sin reshape 成可以 broadcast 到 [B, H, T, D] 的形状。"""
    if x.dim() == 2:
        # [T, D] -> [1, 1, T, D]
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3:
        # [B, T, D] -> [B, 1, T, D]
        return x.unsqueeze(1)
    if x.dim() == 4:
        return x
    raise ValueError(f"Unsupported RoPE tensor shape {tuple(x.shape)} for target {tuple(target.shape)}")


def apply_rope(q: torch.Tensor, k: torch.Tensor, position_embeddings: Any) -> tuple[torch.Tensor, torch.Tensor, str]:
    """返回 post-RoPE Q/K，以及本次 RoPE 状态。

    输入:
        q: [B, q_heads, T, D]
        k: [B, kv_heads, T, D]
        position_embeddings: HF attention forward 传入的 (cos, sin)

    输出:
        q_post/k_post: shape 不变
        status: applied / missing_position_embeddings / fallback_error:...
    """
    if position_embeddings is None:
        return q, k, "missing_position_embeddings"
    try:
        cos, sin = position_embeddings
        cos_q = reshape_rope_tensor(cos, q)
        sin_q = reshape_rope_tensor(sin, q)
        cos_k = reshape_rope_tensor(cos, k)
        sin_k = reshape_rope_tensor(sin, k)
        q_post = (q * cos_q) + (rotate_half(q) * sin_q)
        k_post = (k * cos_k) + (rotate_half(k) * sin_k)
        return q_post, k_post, "applied"
    except Exception as exc:
        return q, k, f"fallback_error:{exc}"


def sample_positions(seq_len: int, max_samples: int) -> torch.Tensor:
    """从 prefill 的 T 个 token 中抽 sampled_T 个位置。

    输入:
        seq_len: T
        max_samples: sampled_T 上限

    输出:
        pos: [sampled_T]
    """
    max_samples = max(1, int(max_samples))
    if seq_len <= max_samples:
        return torch.arange(seq_len, dtype=torch.long)
    return torch.linspace(0, seq_len - 1, steps=max_samples).round().unique().long()


def build_prompt(record: dict[str, Any]) -> str:
    context = record.get("context") or record.get("document") or record.get("passage") or ""
    question = record.get("input") or record.get("question") or record.get("query") or ""
    if context and question:
        return f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    if context:
        return f"Context:\n{context}\n\nAnswer:"
    if question:
        return f"Question:\n{question}\n\nAnswer:"
    return json.dumps(record, ensure_ascii=False)


def resolve_task_files(data_path: str | None, tasks: list[str]) -> list[Path]:
    if not data_path:
        return []
    root = Path(data_path)
    if root.is_file():
        return [root]

    files: list[Path] = []
    if tasks:
        for task in tasks:
            for base in (root, root / "data"):
                path = base / f"{task}.jsonl"
                if path.exists():
                    files.append(path)
                    break
        return files

    for base in (root, root / "data"):
        if base.exists():
            files = sorted(base.glob("*.jsonl"))
            if files:
                return files
    return []


def load_prompts(args: argparse.Namespace) -> tuple[str, list[str]]:
    if args.smoke:
        prompt = (
            "You are analyzing a long document. "
            + "Sparse attention methods try to reuse stable key value memories across decoding steps. " * 180
            + "\nQuestion: Summarize the main idea in one sentence.\nAnswer:"
        )
        return "smoke", [prompt for _ in range(args.sample_num)]

    task_files = resolve_task_files(args.data_path, parse_csv_list(args.tasks))
    if not task_files:
        raise FileNotFoundError("No LongBench jsonl files found. Use --smoke or pass --data_path.")

    prompts: list[str] = []
    names: list[str] = []
    per_file_limit = max(1, math.ceil(args.sample_num / max(1, len(task_files))))
    for path in task_files:
        names.append(path.stem)
        used = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(prompts) >= args.sample_num or used >= per_file_limit:
                    break
                line = line.strip()
                if not line:
                    continue
                prompts.append(build_prompt(json.loads(line)))
                used += 1
        if len(prompts) >= args.sample_num:
            break
    return "+".join(names) or "longbench", prompts


class QKVCollector:
    """收集当前一次 forward 的 per-layer Q/K/V。

    current[layer_idx][view][kind] -> Tensor[heads, sampled_T, head_dim]
    positions[layer_idx] -> Tensor[sampled_T]
    """

    def __init__(self, prefill_sample_tokens: int):
        self.prefill_sample_tokens = prefill_sample_tokens  # prefill 阶段从长序列中采样的 token 数量上限
        self.phase = "" # 当前阶段 "prefill" | "decode" | ""
        self.sample_idx = -1 # 当前是第几个 prompt 样本
        self.step_idx = -1 # 当前是第几步（prefill=0，decode 从 1 递增）
        self.current: dict[int, dict[str, dict[str, torch.Tensor]]] = {} # # layer_idx -> {"pre_rope"|"post_rope"} -> {"q"|"k"|"v"} -> Tensor[heads, sampled_T, head_dim]
        self.positions: dict[int, torch.Tensor] = {} # layer_idx -> 被采样 token 在原序列中的绝对位置
        self.errors: list[str] = []# 采集过程中的异常信息
        self.rope_status_counts: dict[str, int] = defaultdict(int) # RoPE 应用状态统计
        self.rope_status_examples: list[str] = []# RoPE 状态异常的样本示例

    # 清空 current 和 positions，记录当前阶段元信息。每次 prefill/decode forward 之前调用。
    def start(self, phase: str, sample_idx: int, step_idx: int) -> None:
        self.phase = phase
        self.sample_idx = sample_idx
        self.step_idx = step_idx
        self.current = {}
        self.positions = {}
    # 核心方法，被 patch_model_attention 注入的 wrapper 调用
    def record(self, layer_idx: int, attn: torch.nn.Module, call_args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if not self.phase:
            return
        try:
            hidden_states = kwargs.get("hidden_states", call_args[0] if call_args else None)
            if hidden_states is None:
                return

            position_embeddings = kwargs.get("position_embeddings", call_args[1] if len(call_args) > 1 else None)
            head_dim = int(getattr(attn, "head_dim"))
            hidden_shape = (*hidden_states.shape[:-1], -1, head_dim)

            # [B, T, hidden] -> [B, H, T, D]
            q = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            k = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            v = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            if hasattr(attn, "q_norm"):
                q = attn.q_norm(q)
            if hasattr(attn, "k_norm"):
                k = attn.k_norm(k)

            q_post, k_post, rope_status = apply_rope(q, k, position_embeddings)
            self.rope_status_counts[rope_status] += 1
            if rope_status != "applied" and len(self.rope_status_examples) < 10:
                self.rope_status_examples.append(
                    f"sample={self.sample_idx} step={self.step_idx} layer={layer_idx} "
                    f"phase={self.phase} status={rope_status}"
                )

            seq_len = int(q.shape[2])
            if self.phase == "prefill":
                pos = sample_positions(seq_len, self.prefill_sample_tokens).to(q.device)
            else:
                pos = torch.arange(seq_len, dtype=torch.long, device=q.device)

            pre = {"q": q, "k": k, "v": v}
            post = {"q": q_post, "k": k_post, "v": v}
            self.current[layer_idx] = {"pre_rope": {}, "post_rope": {}}
            for view_name, view_states in (("pre_rope", pre), ("post_rope", post)):
                for kind, tensor in view_states.items():
                    # 只支持 batch=1 的离线 trace，保存到 CPU 后再统计。
                    self.current[layer_idx][view_name][kind] = tensor[0, :, pos, :].detach().float().cpu()
            self.positions[layer_idx] = pos.detach().cpu()
        except Exception as exc:
            if len(self.errors) < 20:
                self.errors.append(f"sample={self.sample_idx} step={self.step_idx} layer={layer_idx}: {exc}")

# 非侵入式的模型 hook 注入器，给每一层 attention 模块的 forward 套一层 wrapper，在不改变模型行为的前提下旁路收集 Q/K/V
def patch_model_attention(model: torch.nn.Module, collector: QKVCollector) -> list[tuple[torch.nn.Module, Any]]:
    patched: list[tuple[torch.nn.Module, Any]] = []
    # 第 1 步：定位 layers
    # HuggingFace decoder-only 模型的约定路径是 model.model.layers（例如 Qwen2ForCausalLM.model.layers、LlamaForCausalLM.model.layers）
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("Expected a decoder-only HF model with model.layers")

    # 第 2 步：遍历每层，验证 self_attn 存在，给每层的 self_attn.forward 套 wrapper
    # 验证该层有 q_proj/k_proj/v_proj 三个投影层，缺任意一个就跳过（兼容某些非标准模型）
    for layer_idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj")):
            continue
        original_forward = attn.forward

        # 第 3 步：创建 wrapper 替换 forward
        def wrapped_forward(self, *args, __orig=original_forward, __layer_idx=layer_idx, **kwargs):
            collector.record(__layer_idx, self, args, kwargs)
            return __orig(*args, **kwargs)

        attn.forward = types.MethodType(wrapped_forward, attn)
        patched.append((attn, original_forward))

    if not patched:
        raise ValueError("No attention modules with q_proj/k_proj/v_proj were found")
    return patched


def restore_patches(patched: list[tuple[torch.nn.Module, Any]]) -> None:
    for module, original_forward in patched:
        module.forward = original_forward


def new_stats() -> dict[str, Any]:
    return {
        "layer_mats": {},
        "head_mats": {},
        "scalars": defaultdict(lambda: [0.0, 0]),
        "temporal": defaultdict(lambda: [0.0, 0]),
        "prefill_token": defaultdict(lambda: [0.0, 0]),
        "history": [],
        "num_forward_records": 0,
    }


def add_scalar(table: dict[tuple[Any, ...], list[Any]], key: tuple[Any, ...], value: float, count: int = 1) -> None:
    if math.isfinite(float(value)):
        table[key][0] += float(value) * int(count)
        table[key][1] += int(count)

# add_matrix 是逐元素累积平均的累加器，用于把多次 forward 产生的同类型矩阵（如层间相似矩阵）叠加起来，最后统一除以次数得到平均
def add_matrix(mats: dict[tuple[Any, ...], dict[str, torch.Tensor]], key: tuple[Any, ...], matrix: torch.Tensor) -> None:
    matrix = matrix.detach().float().cpu() # 切断计算图（不干扰推理显存），转 float32，移出 GPU。持久化统计，不和模型抢显存
    if key not in mats:
        mats[key] = {"sum": torch.zeros_like(matrix), "count": torch.zeros_like(matrix)}
    valid = torch.isfinite(matrix)
    mats[key]["sum"][valid] += matrix[valid]
    mats[key]["count"][valid] += 1


def avg_matrix(mats: dict[tuple[Any, ...], dict[str, torch.Tensor]], key: tuple[Any, ...]) -> torch.Tensor:
    item = mats[key]
    count = item["count"].clamp_min(1)
    out = item["sum"] / count
    out[item["count"] == 0] = float("nan")
    return out


def scalar_rows(table: dict[tuple[Any, ...], list[Any]]) -> list[tuple[tuple[Any, ...], float, int]]:
    rows = []
    for key in sorted(table):
        total, count = table[key]
        rows.append((key, total / max(1, count), count))
    return rows


def compute_layer_matrix(states: dict[int, Any], layers: list[int], view: str, kind: str) -> torch.Tensor:
    """layer i 和 layer j 比较相同 head/token 下标的 cosine。"""
    n_layers = max(layers) + 1
    matrix = torch.full((n_layers, n_layers), float("nan"))
    for i in layers:
        a = states[i][view][kind]
        for j in layers:
            b = states[j][view][kind]
            heads = min(a.shape[0], b.shape[0])
            tokens = min(a.shape[1], b.shape[1])
            if heads > 0 and tokens > 0:
                matrix[i, j] = cosine_last_dim(a[:heads, :tokens], b[:heads, :tokens]).mean()
    return matrix


def compute_head_matrix(x: torch.Tensor) -> torch.Tensor:
    """[H, T, D] -> [H, H]，比较同层不同 head 的整体相似性。"""
    heads = int(x.shape[0])
    flat = normalize(x.reshape(heads, -1))
    return flat @ flat.T


def consume_head_similarity(stats: dict[str, Any], states: dict[int, Any], layers: list[int], view: str, kind: str) -> None:
    for layer_idx in layers:
        x = states[layer_idx][view][kind]
        if x.shape[0] <= 0:
            continue
        matrix = compute_head_matrix(x)
        add_matrix(stats["head_mats"], (view, kind, f"head_pair_layer_{layer_idx}"), matrix)

        heads = matrix.shape[0]
        offdiag = matrix[~torch.eye(heads, dtype=torch.bool)]
        if offdiag.numel() > 0:
            add_scalar(stats["scalars"], (view, kind, "head_offdiag_mean", layer_idx, -1), float(offdiag.mean()), offdiag.numel())


def consume_temporal(stats: dict[str, Any], states: dict[int, Any], args: argparse.Namespace) -> None:
    """decode 第 t 步和 t-lag 步比较，每个 layer/head 单独记录。"""
    lags = parse_int_list(args.temporal_lags, positive=True, sorted_unique=True) or [1, 2, 4, 8]
    step_index = len(stats["history"])
    for lag in lags:
        prev_index = step_index - lag
        if prev_index < 0:
            continue
        prev = stats["history"][prev_index]
        for layer_idx in sorted(set(states).intersection(prev)):
            for view in VIEWS:
                for kind in KINDS:
                    cur = states[layer_idx][view][kind][:, -1, :]
                    old = prev[layer_idx][view][kind][:, -1, :]
                    heads = min(cur.shape[0], old.shape[0])
                    if heads <= 0:
                        continue
                    sims = cosine_last_dim(cur[:heads], old[:heads])
                    for head_idx, value in enumerate(sims.tolist()):
                        add_scalar(stats["temporal"], (view, kind, lag, layer_idx, head_idx), float(value))
                    add_scalar(stats["temporal"], (view, kind, lag, layer_idx, -1), float(sims.mean()), heads)


def consume_prefill_token_similarity(
    stats: dict[str, Any],
    states: dict[int, Any],
    positions: dict[int, torch.Tensor],
    layers: list[int],
    view: str,
    kind: str,
    args: argparse.Namespace,
) -> None:
    """prefill 中每层每头做 token-token cosine，统计距离桶和 top-k 邻居。"""
    buckets = parse_int_list(args.distance_buckets, positive=True, sorted_unique=True) or list(DEFAULT_DISTANCE_BUCKETS)
    topks = parse_int_list(args.topk_values, positive=True, sorted_unique=True) or [16, 32, 64]

    for layer_idx in layers:
        x = states[layer_idx][view][kind]
        heads, tokens, _ = x.shape
        if tokens < 2:
            continue

        pos = positions.get(layer_idx, torch.arange(tokens))
        if int(pos.numel()) != tokens:
            pos = torch.arange(tokens)
        diff = (pos[None, :] - pos[:, None]).abs()

        for head_idx in range(heads):
            sim = normalize(x[head_idx]) @ normalize(x[head_idx]).T
            sim.fill_diagonal_(float("-inf"))
            finite = torch.isfinite(sim)

            for i, bucket in enumerate(buckets):
                if i == len(buckets) - 1:
                    mask = (diff >= bucket) & finite
                else:
                    mask = (diff >= bucket) & (diff < buckets[i + 1]) & finite
                if mask.any():
                    add_scalar(
                        stats["prefill_token"],
                        (view, kind, "distance_bucket", layer_idx, head_idx, bucket),
                        float(sim[mask].mean()),
                        int(mask.sum()),
                    )

            for topk in topks:
                k_eff = min(topk, tokens - 1)
                if k_eff > 0:
                    vals = torch.topk(sim, k=k_eff, dim=-1).values
                    add_scalar(
                        stats["prefill_token"],
                        (view, kind, "topk_neighbor", layer_idx, head_idx, topk),
                        float(vals.mean()),
                        int(vals.numel()),
                    )


def consume_forward(stats: dict[str, Any], collector: QKVCollector, phase: str, args: argparse.Namespace) -> None:
    states = collector.current
    if not states:
        return
    stats["num_forward_records"] += 1
    layers = sorted(states)

    for view in VIEWS:
        for kind in KINDS:
            layer_matrix = compute_layer_matrix(states, layers, view, kind)
            add_matrix(stats["layer_mats"], (view, kind, "layer_pair"), layer_matrix) # 多prompt的同类型矩阵累积到一起，最后平均
            consume_head_similarity(stats, states, layers, view, kind)
            if phase == "prefill":
                consume_prefill_token_similarity(stats, states, collector.positions, layers, view, kind, args)

    if phase == "decode":
        consume_temporal(stats, states, args)
    stats["history"].append(states)


def run_one_prompt(
    prompt: str,
    sample_idx: int,
    model: torch.nn.Module,
    tokenizer: Any,
    collector: QKVCollector,
    stats: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    '''prompt 的完整推理 + 采集循环，分为 prefill 和 decode 两个阶段，
    每一步都先通知 collector 再执行 forward，让被 hook 的 attention 模块把 Q/K/V 抓下来。'''

    # 第 1 步：Tokenize
    # 把文本转成 token ids，截断到 max_prefill_tokens（默认 2048），送入 GPU。少于 2 个 token 的 prompt 直接跳过（无法做有意义的相似性分析）
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_prefill_tokens).input_ids
    input_ids = input_ids.to(model_input_device(model))
    if input_ids.shape[1] < 2:
        return 0

    # 第 2 步：Init
    # generated 统计实际生成了多少个 token（提前遇到 EOS 会提前终止），stats["history"] 清空——每个 prompt 独立统计，用于 temporal（时序）分析。
    generated = 0
    stats["history"] = []

    with torch.no_grad():
        # 第 3 步：Prefill 阶段
        collector.start("prefill", sample_idx, 0) # 告诉 collector "接下来是一次 prefill forward"，清空上一步的旧数据。
        outputs = model(input_ids=input_ids, use_cache=True) # 模型吃进完整 prompt → 每层 self_attn 的 wrapper 在 forward 之前被触发 → 调用 collector.record() → 把该层 Q/K/V（pre_rope + post_rope）存入 collector.current。use_cache=True 保证返回 past_key_values 用于后续 decode
        consume_forward(stats, collector, "prefill", args) # 立刻消费 collector.current 里的 Q/K/V，计算 layer-layer 相似矩阵、head-head 相似矩阵、prefill token 距离桶统计，然后把 current 存入 stats["history"]。
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True) # 贪心解码：取 logits 最后一个位置的 argmax 作为下一个 token
        past_key_values = outputs.past_key_values
        if tokenizer.eos_token_id is not None and int(next_token.item()) == int(tokenizer.eos_token_id):
            return generated
        # 第 4 步：Decode 循环
        for step_idx in range(1, args.max_new_tokens + 1):
            # 和 prefill 完全一样的模式：start → forward（collector 自动抓）→ consume_forward。
            # 区别在于 consume_forward 的 "decode" 分支额外调用 consume_temporal——把第 t 步的 Q/K/V 和第 t-1、t-2、t-4、t-8 步比较，
            # 统计每个 layer/head 随时间的相似性衰退。
            collector.start("decode", sample_idx, step_idx)
            outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            consume_forward(stats, collector, "decode", args)
            generated += 1

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values
            if tokenizer.eos_token_id is not None and int(next_token.item()) == int(tokenizer.eos_token_id):
                break
    return generated


def write_scalar_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_temporal_csv(output_dir: Path, stats: dict[str, Any]) -> None:
    rows = []
    for key, value, count in scalar_rows(stats["temporal"]):
        view, kind, lag, layer_idx, head_idx = key
        rows.append([view, kind, lag, layer_idx, head_idx, value, count])
    write_scalar_csv(output_dir / "temporal_similarity.csv", ["view", "kind", "lag", "layer", "head", "value", "count"], rows)


def write_layer_head_csv(output_dir: Path, stats: dict[str, Any]) -> None:
    rows = []
    for key, value, count in scalar_rows(stats["scalars"]):
        view, kind, metric, layer_idx, head_idx = key
        rows.append([view, kind, metric, layer_idx, head_idx, value, count])
    for key, value, count in scalar_rows(stats["prefill_token"]):
        view, kind, metric, layer_idx, head_idx, bucket = key
        rows.append([view, kind, f"{metric}_{bucket}", layer_idx, head_idx, value, count])
    write_scalar_csv(output_dir / "per_layer_head.csv", ["view", "kind", "metric", "layer", "head", "value", "count"], rows)


def plot_heatmap(matrix: torch.Tensor, title: str, path: Path) -> None:
    if not HAS_PLOT:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix.numpy(), cmap="viridis", annot=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_temporal_heatmaps(output_dir: Path, stats: dict[str, Any]) -> None:
    """画 AttentionPredictor 论文同口径的 temporal heatmap。

    横坐标: head index
    纵坐标: layer index
    单元格: 同一 layer/head 的 Q/K/V 在 t 和 t-lag 之间的 cosine similarity

    注意: 这和 layer_heatmaps 不同。layer_heatmaps 是 layer-vs-layer；
    temporal_heatmaps 是 layer-vs-head，才对应 AttentionPredictor 的 Query
    self-similarity 图。
    """
    if not HAS_PLOT:
        return

    grouped: dict[tuple[str, str, int], list[tuple[int, int, float]]] = defaultdict(list)
    for key, value, _count in scalar_rows(stats["temporal"]):
        view, kind, lag, layer_idx, head_idx = key
        if head_idx >= 0:
            grouped[(view, kind, lag)].append((int(layer_idx), int(head_idx), float(value)))

    for (view, kind, lag), rows in grouped.items():
        if not rows:
            continue
        max_layer = max(layer for layer, _head, _value in rows)
        max_head = max(head for _layer, head, _value in rows)
        matrix = torch.full((max_layer + 1, max_head + 1), float("nan"))
        for layer, head, value in rows:
            matrix[layer, head] = value

        path = output_dir / "temporal_heatmaps" / f"{view}_{kind}_lag_{lag}_temporal_similarity.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(max(9, 0.35 * (max_head + 1)), max(7, 0.28 * (max_layer + 1))))
        sns.heatmap(
            matrix.numpy(),
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            annot=False,
            cbar_kws={"label": "cosine similarity"},
        )
        plt.title(f"{view} {kind.upper()} temporal similarity lag={lag}")
        plt.xlabel("Head Index")
        plt.ylabel("Layer Index")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()


def matrix_summary(matrix: torch.Tensor) -> dict[str, float | None]:
    mean = float(torch.nanmean(matrix)) if torch.isfinite(matrix).any() else None
    adjacent = None
    offdiag = None
    if matrix.shape[0] > 1:
        diag1 = torch.diagonal(matrix, offset=1)
        if torch.isfinite(diag1).any():
            adjacent = float(torch.nanmean(diag1))
        mask = ~torch.eye(matrix.shape[0], dtype=torch.bool)
        vals = matrix[mask]
        if torch.isfinite(vals).any():
            offdiag = float(torch.nanmean(vals))
    return {"mean_including_diag": mean, "offdiag_mean": offdiag, "adjacent_mean": adjacent}


def write_outputs(
    output_dir: Path,
    model_path: str,
    dataset_name: str,
    prompts_used: int,
    generated_tokens: int,
    stats: dict[str, Any],
    collector: QKVCollector,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "layer_heatmaps").mkdir(exist_ok=True)
    (output_dir / "head_heatmaps").mkdir(exist_ok=True)
    (output_dir / "temporal_heatmaps").mkdir(exist_ok=True)

    write_temporal_csv(output_dir, stats)
    write_layer_head_csv(output_dir, stats)
    plot_temporal_heatmaps(output_dir, stats)

    layer_summary = {}
    for key in stats["layer_mats"]:
        view, kind, metric = key
        avg = avg_matrix(stats["layer_mats"], key)
        layer_summary[f"{view}/{kind}/{metric}"] = matrix_summary(avg)
        plot_heatmap(avg, f"{view} {kind.upper()} layer similarity", output_dir / "layer_heatmaps" / f"{view}_{kind}_layer_similarity.png")

    selected_layers = set(parse_int_list(args.plot_layers, positive=False, sorted_unique=True))
    if not selected_layers and stats["layer_mats"]:
        first_key = next(iter(stats["layer_mats"]))
        n_layers = stats["layer_mats"][first_key]["sum"].shape[0]
        selected_layers = {0, n_layers // 2, n_layers - 1}

    for key in stats["head_mats"]:
        view, kind, metric = key
        layer_idx = int(metric.rsplit("_", 1)[-1])
        if layer_idx not in selected_layers:
            continue
        avg = avg_matrix(stats["head_mats"], key)
        plot_heatmap(avg, f"{view} {kind.upper()} head similarity layer {layer_idx}", output_dir / "head_heatmaps" / f"{view}_{kind}_layer_{layer_idx}_head_similarity.png")

    temporal_layer_averages = {
        "/".join(map(str, key)): {"value": value, "count": count}
        for key, value, count in scalar_rows(stats["temporal"])
        if key[-1] == -1
    }
    summary = {
        "model_path": model_path,
        "dataset": dataset_name,
        "prompts_used": prompts_used,
        "generated_tokens": generated_tokens,
        "num_forward_records": stats["num_forward_records"],
        "args": vars(args),
        "layer_summary": layer_summary,
        "temporal_layer_averages": temporal_layer_averages,
        "rope_status_counts": dict(sorted(collector.rope_status_counts.items())),
        "rope_status_examples": collector.rope_status_examples,
        "record_errors": collector.errors,
        "plotting_enabled": HAS_PLOT,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def load_model_and_tokenizer(model_path: str, args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.dtype == "bf16" else (torch.float16 if args.dtype == "fp16" else "auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    return model, tokenizer


def run_for_model(model_path: str, dataset_name: str, prompts: list[str], args: argparse.Namespace) -> Path:
    model_name = safe_name(model_path)
    output_dir = Path(args.output_dir) / model_name / dataset_name

    collector = QKVCollector(args.prefill_sample_tokens) # 是一个纯数据容器，负责在整个推理过程中收集每层每步的 Q/K/V 张量（pre_rope + post_rope 两个视图）
    stats = new_stats()
    model, tokenizer = load_model_and_tokenizer(model_path, args) # 加载 HF 模型到 GPU，设置 eval 模式
    patched = patch_model_attention(model, collector) # 遍历 model.layers[*].self_attn，把每个 attention 模块的 forward 替换成一个 wrapper，wrapper 的核心逻辑是：在原 forward 执行之前，手动调用 q_proj/k_proj/v_proj 算出 Q/K/V，调用 QKVCollector.record() 保存下来，然后再调用原始 forward。这样不改变模型输出，只是旁路收集数据。

    generated_total = 0
    try: # 遍历 prompts 执行推理
        for sample_idx, prompt in enumerate(tqdm(prompts, desc=f"trace {model_name}")):
            generated_total += run_one_prompt(prompt, sample_idx, model, tokenizer, collector, stats, args)
    finally:
        restore_patches(patched) # 把每个 attention 模块的 forward 恢复为原始版本
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # 释放 GPU 显存，避免多个模型连续分析时 OOM

    if stats["num_forward_records"] == 0: # 先做安全检查——如果一次 forward 都没记录到（可能是模型结构不匹配、attention 实现方式不同导致 hook 失败），直接报错。然后调用 write_outputs 写出 summary.json、csv 文件和 heatmap 图片
        raise RuntimeError("No Q/K/V records were collected. Check model architecture and attention implementation.")
    if collector.errors:
        print(f"[Warning] Collector recorded {len(collector.errors)} errors. See summary.json for examples.")
    write_outputs(output_dir, model_path, dataset_name, len(prompts), generated_total, stats, collector, args)
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trace Q/K/V similarity by layer, head, and decode step.")
    parser.add_argument("--model_paths", nargs="+", default=["models/Qwen2.5-0.5B-Instruct"])
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--max_prefill_tokens", type=int, default=2048)
    parser.add_argument("--prefill_sample_tokens", type=int, default=256)
    parser.add_argument("--temporal_lags", default="1,2,4,8")
    parser.add_argument("--distance_buckets", default=",".join(map(str, DEFAULT_DISTANCE_BUCKETS)))
    parser.add_argument("--topk_values", default="16,32,64")
    parser.add_argument("--plot_layers", default="")
    parser.add_argument("--output_dir", default="outputs/qkv_similarity_trace")
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument(
        "--attn_implementation",
        default="eager",
        help="HF attention implementation. eager is preferred because the forward inputs are easiest to inspect.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.smoke and not args.data_path:
        raise ValueError("Pass --smoke or --data_path.")

    dataset_name, prompts = load_prompts(args)
    prompts = prompts[: args.sample_num]
    if not prompts:
        raise ValueError("No prompts loaded.")

    output_dirs = []
    for model_path in args.model_paths:
        output_dirs.append(str(run_for_model(model_path, dataset_name, prompts, args)))

    print("Trace completed. Outputs:")
    for path in output_dirs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
