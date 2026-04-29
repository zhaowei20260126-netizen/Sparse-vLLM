import json
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from sparsevllm.utils.log import logger


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _translate_deepseek_weight_name(model: nn.Module, weight_name: str) -> str | None:
    if getattr(model, "hf_model_type", "") not in ("deepseek_v2", "deepseek_v32"):
        return None

    if weight_name == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    if weight_name == "model.norm.weight":
        return "norm.weight"
    if weight_name == "lm_head.weight":
        return "output.weight"

    if not weight_name.startswith("model.layers."):
        return None

    parts = weight_name.split(".")
    if len(parts) < 5:
        return None
    layer_idx = parts[2]
    tail = parts[3:]
    prefix = f"layers.{layer_idx}."

    if tail == ["input_layernorm", "weight"]:
        return prefix + "attn_norm.weight"
    if tail == ["post_attention_layernorm", "weight"]:
        return prefix + "ffn_norm.weight"

    if tail[:2] == ["self_attn", "q_proj"] and getattr(getattr(model, "args", None), "q_lora_rank", 0) <= 0:
        return prefix + "attn.wq.weight"
    if tail[:2] == ["self_attn", "kv_a_proj_with_mqa"]:
        return prefix + "attn.wkv_a.weight"
    if tail[:2] == ["self_attn", "kv_a_layernorm"]:
        return prefix + "attn.kv_norm.weight"
    if tail[:2] == ["self_attn", "kv_b_proj"]:
        return prefix + "attn.wkv_b.weight"
    if tail[:2] == ["self_attn", "o_proj"]:
        return prefix + "attn.wo.weight"

    if tail[:2] == ["mlp", "gate_proj"]:
        return prefix + "ffn.w1.weight"
    if tail[:2] == ["mlp", "up_proj"]:
        return prefix + "ffn.w3.weight"
    if tail[:2] == ["mlp", "down_proj"]:
        return prefix + "ffn.w2.weight"
    if tail[:2] == ["mlp", "gate"] and len(tail) == 3 and tail[2] == "weight":
        return prefix + "ffn.gate.weight"
    if tail[:2] == ["mlp", "shared_experts"] and len(tail) == 4:
        proj = tail[2]
        if proj == "gate_proj":
            return prefix + "ffn.shared_experts.w1.weight"
        if proj == "up_proj":
            return prefix + "ffn.shared_experts.w3.weight"
        if proj == "down_proj":
            return prefix + "ffn.shared_experts.w2.weight"
    if tail[:2] == ["mlp", "experts"] and len(tail) == 5:
        expert_idx = tail[2]
        proj = tail[3]
        if proj == "gate_proj":
            return prefix + f"ffn.experts.{expert_idx}.w1.weight"
        if proj == "up_proj":
            return prefix + f"ffn.experts.{expert_idx}.w3.weight"
        if proj == "down_proj":
            return prefix + f"ffn.experts.{expert_idx}.w2.weight"

    return None


def _iter_deltakv_compressor_items(state_dict: dict[str, torch.Tensor]):
    """
    从 state_dict 中迭代所有压缩器相关的权重项。
    
    支持两种格式：
    1) HF 风格按层："...layers.{i}....compress_down...."
    2) 共享压缩器："compress_down...."（广播到所有层）
    """
    for key, weight in state_dict.items():
        parts = key.split(".")
        if "compress_down" in parts:
            yield key, "compress_down", parts, weight
        elif "compress_up" in parts:
            yield key, "compress_up", parts, weight


def _infer_single_compressor_spec(state_dict: dict[str, torch.Tensor], comp_name: str):
    """
    从权重文件推断压缩器的架构规格。
    
    返回：(种类, 中间维度, 是否有偏置, 数据类型)
    - 种类："linear", "mlp_gelu", "mlp_swiglu" 等
    - 中间维度：MLP 的隐藏层大小（Linear 为 None）
    """
    bias = False
    inferred_kind = None
    inferred_intermediate = None
    inferred_dtype = None

    for key, name, parts, weight in _iter_deltakv_compressor_items(state_dict):
        if name != comp_name:
            continue
        comp_token_idx = parts.index(comp_name)
        sub_parts = parts[comp_token_idx + 1:]
        if not sub_parts:
            continue

        inferred_dtype = inferred_dtype or weight.dtype
        if sub_parts[-1] == "bias":
            bias = True

        head = sub_parts[0]
        # nn.Linear: 直接的权重矩阵
        if head in ("weight", "bias"):
            inferred_kind = inferred_kind or "linear"
            continue
        # nn.Sequential 的 MLP（GELU激活）
        if head.isdigit():
            inferred_kind = inferred_kind or "mlp_gelu"
            if head == "0" and sub_parts[-1] == "weight":
                inferred_intermediate = inferred_intermediate or int(weight.shape[0])
            continue
        # SwiGLU 风格的 MLP
        if head == "w12":
            inferred_kind = inferred_kind or "mlp_swiglu"
            if sub_parts[-1] == "weight":
                inferred_intermediate = inferred_intermediate or int(weight.shape[0] // 2)
            continue

    return inferred_kind, inferred_intermediate, bias, inferred_dtype


def _infer_kv_compressed_size(state_dict: dict[str, torch.Tensor]) -> int | None:
    """
    从权重文件推断 KV 缓存的压缩大小。
    
    通过查看压缩器的输出维度来推断（compress_down 的输出 = compress_up 的输入）。
    """
    for _, name, parts, weight in _iter_deltakv_compressor_items(state_dict):
        comp_token_idx = parts.index(name)
        sub_parts = parts[comp_token_idx + 1:]
        if not sub_parts or sub_parts[-1] != "weight":
            continue

        head = sub_parts[0]
        if name == "compress_down":
            # compress_down 的输出维度
            if head in ("weight", "2", "w3"):
                return int(weight.shape[0])
        else:
            # compress_up 的输入维度（应与 compress_down 输出相同）
            if head in ("weight", "0", "w12"):
                return int(weight.shape[1])
    return None


def _resolve_deltakv_checkpoint_files(path: str) -> tuple[str, list[str], bool]:
    """
    定位 DeltaKV checkpoint 文件。
    
    支持目录（查找所有权重文件）或单个文件。
    返回：(checkpoint 目录, 权重文件列表, 是否是 safetensors 格式)
    """
    if os.path.isdir(path):
        # 优先查找 safetensors 格式（更快），其次是 .bin 或 .pt
        files = sorted(glob(os.path.join(path, "*.safetensors")))
        is_safetensors = len(files) > 0
        if not is_safetensors:
            files = sorted(glob(os.path.join(path, "*.bin")) + glob(os.path.join(path, "*.pt")))
        ckpt_dir = path
    elif os.path.isfile(path):
        # 单个文件情况
        files = [path]
        is_safetensors = path.endswith(".safetensors")
        ckpt_dir = os.path.dirname(path)
    else:
        raise FileNotFoundError(f"No compressor weights found in {path}")

    if not files:
        raise FileNotFoundError(f"No compressor weights found in {path}")
    return ckpt_dir, files, is_safetensors


def _load_deltakv_state_dict(file: str, *, is_safetensors: bool) -> dict[str, torch.Tensor]:
    """
    加载 DeltaKV checkpoint 的权重字典。
    
    支持 safetensors（更快、更安全）和 PyTorch .pt/.bin 格式。
    """
    if is_safetensors:
        with safe_open(file, "pt", "cpu") as f:
            return {k: f.get_tensor(k) for k in f.keys()}
    return torch.load(file, map_location="cpu")


def sync_deltakv_config_from_checkpoint(config) -> bool:
    """
    从 DeltaKV checkpoint 中同步压缩器配置到运行时配置。
    
    流程：
    1. 优先从 checkpoint 目录的 config.json 读取配置
    2. 如果缺少某些配置参数，从权重文件推断（推断压缩器的架构类型、大小等）
    3. 将新配置应用到 config 对象中
    
    返回：True 表示有配置被修改，False 表示无变化
    """
    # 检查是否配置了 DeltaKV checkpoint 和方法
    path = getattr(config, "deltakv_path", None)
    method = str(getattr(config, "vllm_sparse_method", "") or "")
    if not path or not method.startswith("deltakv"):
        return False

    # 定位 checkpoint 文件
    ckpt_dir, files, is_safetensors = _resolve_deltakv_checkpoint_files(path)
    updates: dict[str, object] = {}
    
    # ==================== 第1步：从 config.json 读取配置 ====================
    config_json = os.path.join(ckpt_dir, "config.json")
    if os.path.isfile(config_json):
        with open(config_json, "r", encoding="utf-8") as f:
            ckpt_cfg = json.load(f)

        # 提取压缩器相关的配置参数
        for key in (
            "kv_compressed_size",
            "use_nonlinear_compressor",
            "compressor_intermediate_size",
            "compressor_linear_bias",
            "compressor_down_type",
            "compressor_up_type",
            "compressor_down_intermediate_size",
            "compressor_up_intermediate_size",
        ):
            if key in ckpt_cfg:
                updates[key] = ckpt_cfg[key]

        # 检查是否使用了不支持的 split_kv 模式（K 和 V 分别压缩）
        if ckpt_cfg.get("split_kv", False):
            raise NotImplementedError(
                "Detected split_kv DeltaKV checkpoint from config.json. "
                "sparsevllm DeltaKVCacheManager currently expects unified compress_down/compress_up."
            )

    # ==================== 第2步：从权重文件推断缺失的配置 ====================
    # 如果 config.json 中缺少某些必要配置，则从权重文件推断
    missing_shape_keys = {
        "kv_compressed_size",
        "compressor_down_type",
        "compressor_up_type",
        "compressor_down_intermediate_size",
        "compressor_up_intermediate_size",
        "compressor_linear_bias",
        "use_nonlinear_compressor",
    }
    need_weight_inference = bool(missing_shape_keys.difference(updates.keys()))
    state_dict = None
    
    if need_weight_inference:
        # 加载权重文件以推断压缩器架构
        for file in files:
            state_dict = _load_deltakv_state_dict(file, is_safetensors=is_safetensors)
            if any("compress_down" in key or "compress_up" in key for key in state_dict):
                break
            state_dict = None

        if state_dict is not None:
            # 从权重文件推断压缩器的配置参数
            kv_compressed_size = _infer_kv_compressed_size(state_dict)
            down_kind, down_inter, down_bias, _ = _infer_single_compressor_spec(state_dict, "compress_down")
            up_kind, up_inter, up_bias, _ = _infer_single_compressor_spec(state_dict, "compress_up")
            
            # 填充缺失的配置项
            if kv_compressed_size is not None and "kv_compressed_size" not in updates:
                updates["kv_compressed_size"] = kv_compressed_size
            if down_kind is not None and "compressor_down_type" not in updates:
                updates["compressor_down_type"] = down_kind
            if up_kind is not None and "compressor_up_type" not in updates:
                updates["compressor_up_type"] = up_kind
            if down_kind is not None and "compressor_down_intermediate_size" not in updates:
                updates["compressor_down_intermediate_size"] = -1 if down_kind == "linear" else int(down_inter or -1)
            if up_kind is not None and "compressor_up_intermediate_size" not in updates:
                updates["compressor_up_intermediate_size"] = -1 if up_kind == "linear" else int(up_inter or -1)
            if down_kind is not None and up_kind is not None and "use_nonlinear_compressor" not in updates:
                updates["use_nonlinear_compressor"] = (down_kind != "linear" or up_kind != "linear")
            if down_bias == up_bias and "compressor_linear_bias" not in updates:
                updates["compressor_linear_bias"] = bool(down_bias)

            # 检查是否使用了不支持的 split_kv 格式
            for key in state_dict.keys():
                if ".k_compress_down." in key or ".v_compress_down." in key or ".k_compress_up." in key or ".v_compress_up." in key:
                    raise NotImplementedError(
                        "Detected split_kv compressor checkpoint (k_compress_*/v_compress_*). "
                        "sparsevllm DeltaKVCacheManager currently expects unified compress_down/compress_up."
                    )

    # ==================== 第3步：应用配置更新 ====================
    changed: dict[str, tuple[object, object]] = {}
    for key, value in updates.items():
        if value is None or not hasattr(config, key):
            continue
        current = getattr(config, key)
        if isinstance(value, str):
            value = value.strip().lower() or "auto"
        # 仅在配置值发生变化时才更新
        if current != value:
            setattr(config, key, value)
            changed[key] = (current, value)

    # 记录日志
    if changed:
        changes_str = ", ".join(f"{key}: {old} -> {new}" for key, (old, new) in changed.items())
        logger.info(f"Synced DeltaKV config from checkpoint {ckpt_dir}: {changes_str}")
    return bool(changed)


def _compressor_signature(mod: nn.Module):
    if isinstance(mod, nn.Linear):
        return "linear", None, (mod.bias is not None)
    if hasattr(mod, "w12") and hasattr(mod, "w3"):
        w3 = getattr(mod, "w3")
        inter = int(w3.in_features) if isinstance(w3, nn.Linear) else None
        has_bias = bool(getattr(getattr(mod, "w12", None), "bias", None) is not None) and bool(
            getattr(getattr(mod, "w3", None), "bias", None) is not None
        )
        return "mlp_swiglu", inter, has_bias
    if isinstance(mod, nn.Sequential) and len(mod) >= 3 and isinstance(mod[0], nn.Linear) and isinstance(mod[1], nn.GELU) and isinstance(mod[2], nn.Linear):
        return "mlp_gelu", int(mod[0].out_features), (mod[0].bias is not None and mod[2].bias is not None)
    return mod.__class__.__name__, None, None


def _maybe_rebuild_cache_manager_compressors(cache_manager, state_dict: dict[str, torch.Tensor]):
    config = getattr(cache_manager, "config", None)
    if config is None:
        return

    down_kind, down_inter, down_bias, down_dtype = _infer_single_compressor_spec(state_dict, "compress_down")
    up_kind, up_inter, up_bias, up_dtype = _infer_single_compressor_spec(state_dict, "compress_up")
    if down_kind is None or up_kind is None:
        return

    current_down = cache_manager.compress_down[0] if getattr(cache_manager, "compress_down", None) else None
    current_up = cache_manager.compress_up[0] if getattr(cache_manager, "compress_up", None) else None
    cur_down_kind, cur_down_inter, cur_down_bias = _compressor_signature(current_down) if current_down is not None else (None, None, None)
    cur_up_kind, cur_up_inter, cur_up_bias = _compressor_signature(current_up) if current_up is not None else (None, None, None)

    need_rebuild = (
        cur_down_kind != down_kind
        or cur_up_kind != up_kind
        or (down_kind != "linear" and cur_down_inter != down_inter)
        or (up_kind != "linear" and cur_up_inter != up_inter)
        or cur_down_bias != down_bias
        or cur_up_bias != up_bias
    )
    target_dtype = down_dtype or up_dtype
    if not need_rebuild:
        # Even if the structure matches, ensure dtype matches checkpoint weights (avoid bf16->fp32 silent casts).
        if target_dtype is not None and current_down is not None:
            try:
                cur_dtype = next(current_down.parameters()).dtype
            except StopIteration:
                cur_dtype = None
            if cur_dtype is not None and cur_dtype != target_dtype:
                logger.info(f"Casting sparsevllm compressors to {target_dtype} to match checkpoint weights.")
                cache_manager.compress_down = [m.to(dtype=target_dtype) for m in cache_manager.compress_down]
                cache_manager.compress_up = [m.to(dtype=target_dtype) for m in cache_manager.compress_up]
        return

    logger.info(
        f"Rebuilding sparsevllm compressors to match checkpoint: "
        f"down={down_kind}(inter={down_inter},bias={down_bias}) "
        f"up={up_kind}(inter={up_inter},bias={up_bias})"
    )

    # Update config so downstream logs/debug reflect the effective compressor architecture.
    if hasattr(config, "compressor_linear_bias"):
        if down_bias == up_bias:
            config.compressor_linear_bias = bool(down_bias)
        else:
            logger.warning(
                "Checkpoint has asymmetric bias setting between compress_down and compress_up; "
                "sparsevllm config only has a single `compressor_linear_bias`, keeping current value."
            )
    if hasattr(config, "use_nonlinear_compressor"):
        config.use_nonlinear_compressor = (down_kind != "linear" or up_kind != "linear")
    if hasattr(config, "compressor_down_type"):
        config.compressor_down_type = down_kind
    if hasattr(config, "compressor_up_type"):
        config.compressor_up_type = up_kind
    if hasattr(config, "compressor_down_intermediate_size"):
        config.compressor_down_intermediate_size = -1 if down_kind == "linear" else int(down_inter or -1)
    if hasattr(config, "compressor_up_intermediate_size"):
        config.compressor_up_intermediate_size = -1 if up_kind == "linear" else int(up_inter or -1)

    from sparsevllm.utils.compressor import create_compressor

    num_layers = len(cache_manager.compress_down)
    cache_manager.compress_down = [
        create_compressor(is_down=True, config=config, bias_override=down_bias).to(device="cuda", dtype=target_dtype)
        for _ in range(num_layers)
    ]
    cache_manager.compress_up = [
        create_compressor(is_down=False, config=config, bias_override=up_bias).to(device="cuda", dtype=target_dtype)
        for _ in range(num_layers)
    ]


def load_deltakv_compressors_to_cache_manager(cache_manager, path: str):
    """
    Load DeltaKV compressor weights into cache manager compressor modules.
    """
    _, files, is_safetensors = _resolve_deltakv_checkpoint_files(path)

    loaded_count = 0
    for file in files:
        state_dict = _load_deltakv_state_dict(file, is_safetensors=is_safetensors)

        # Detect unsupported split_kv checkpoints early (k_compress_down/v_compress_down).
        for key in state_dict.keys():
            if ".k_compress_down." in key or ".v_compress_down." in key or ".k_compress_up." in key or ".v_compress_up." in key:
                raise NotImplementedError(
                    "Detected split_kv compressor checkpoint (k_compress_*/v_compress_*). "
                    "sparsevllm DeltaKVCacheManager currently expects unified compress_down/compress_up."
                )

        if loaded_count == 0:
            _maybe_rebuild_cache_manager_compressors(cache_manager, state_dict)

        for key, weight in state_dict.items():
            parts = key.split('.')
            if "compress_down" not in parts and "compress_up" not in parts:
                continue

            # Two formats:
            # 1) HF-style per-layer: "...layers.{i}....compress_down...."
            # 2) Shared compressors: "compress_down...." (broadcast to all DeltaKV layers)
            is_shared = ("layers" not in parts)
            layer_idx = None
            if not is_shared:
                try:
                    layer_token_idx = parts.index("layers")
                    layer_idx = int(parts[layer_token_idx + 1])
                except (ValueError, IndexError):
                    raise ValueError(f"无法从权重键名中解析层索引: {key}")

            if "compress_down" in parts:
                comp_name = "compress_down"
            else:
                comp_name = "compress_up"

            comp_token_idx = parts.index(comp_name)
            sub_key = ".".join(parts[comp_token_idx + 1:])

            if is_shared:
                target_layer_indices = list(range(len(cache_manager.compress_down)))
            else:
                if layer_idx not in cache_manager.deltakv_layer_to_idx:
                    logger.debug(f"权重 {key} 对应的层索引 {layer_idx} 不在 deltakv_layer_to_idx 中，跳过")
                    continue
                target_layer_indices = [cache_manager.deltakv_layer_to_idx[layer_idx]]

            for l_idx in target_layer_indices:
                compressor = cache_manager.compress_down[l_idx] if comp_name == "compress_down" else cache_manager.compress_up[l_idx]
                try:
                    # 尝试获取对应的参数
                    if '.' in sub_key:
                        prefix, name = sub_key.rsplit('.', 1)
                        param = compressor.get_submodule(prefix).get_parameter(name)
                    else:
                        param = compressor.get_parameter(sub_key)

                    if param.shape != weight.shape:
                        raise ValueError(f"权重 {key} 形状不匹配: 预期 {param.shape}, 实际 {weight.shape}")

                    param.data.copy_(weight)
                    loaded_count += 1
                except Exception as e:
                    # 尝试直接访问属性作为备选方案
                    try:
                        target = getattr(compressor, sub_key)
                        if isinstance(target, nn.Parameter):
                            if target.shape != weight.shape:
                                raise ValueError(f"权重 {key} 形状不匹配: 预期 {target.shape}, 实际 {weight.shape}")
                            target.data.copy_(weight)
                            loaded_count += 1
                            continue
                    except Exception:
                        pass

                    raise RuntimeError(f"未能将权重 {key} 加载到压缩器模块 (layer {layer_idx}): {e}")

    assert loaded_count > 0, f"No DeltaKV compressor weights were loaded into cache manager from {path}"
    print(f"Successfully loaded {loaded_count} DeltaKV compressor weights into cache manager from {path}")


def load_model(model: nn.Module, path: str, *, rank: int | None = None, world_size: int | None = None):
    """
    从 safetensors 权重文件加载模型权重。
    
    功能：
    - 支持多进程张量并行（Tensor Parallel）分片加载
    - 支持模型架构转换（DeepSeek HF 格式 → sparsevllm 格式）
    - 支持融合投影层合并（如 q/k/v_proj → qkv_proj）
    
    参数：
    - model: 目标 PyTorch 模型
    - path: 权重文件所在目录
    - rank: 当前 GPU 进程编号（TP 场景下用于分片加载）
    - world_size: 总进程数（TP 场景下用于分片加载）
    """
    # ==================== 第1步：获取打包模块映射 ====================
    # 打包模块映射规则用于将多个独立权重（如 q_proj, k_proj, v_proj）
    # 合并到单个融合权重（如 qkv_proj）以加速推理
    # 例如 Qwen2 的规则：{"q_proj": ("qkv_proj", 0), "k_proj": ("qkv_proj", 1), ...}
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # 发现权重目录中的所有 .safetensors 文件
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    assert len(files) > 0, f"No safetensors found in {path}"

    # ==================== 第2步：处理张量并行（TP）分片加载 ====================
    # 如果是分布式推理（rank 和 world_size 都提供），则按 Rank 加载对应的权重分片
    # 
    # DeepSeek 官方转换器的分片格式：model{rank}-mp{world_size}.safetensors
    # 例如 TP=4 时的文件：
    #   model0-mp4.safetensors (Rank 0 的权重分片)
    #   model1-mp4.safetensors (Rank 1 的权重分片)
    #   model2-mp4.safetensors (Rank 2 的权重分片)
    #   model3-mp4.safetensors (Rank 3 的权重分片)
    if rank is not None and world_size is not None:
        # 构造该 Rank 对应的权重分片文件名
        shard = os.path.join(path, f"model{rank}-mp{world_size}.safetensors")
        if os.path.isfile(shard):
            # ✓ 找到了该 Rank 的分片，只加载这个文件
            # 这样可以节省内存（只加载 1/world_size 的权重）
            files = [shard]
        else:
            # 检查是否存在其他 Rank 的分片（表示确实是分片模式）
            mp_files = sorted(glob(os.path.join(path, f"model*-mp{world_size}.safetensors")))
            if mp_files:
                # ✗ 发现了分片模式但缺少当前 Rank 的文件，这是致命错误
                # （可能是用户传入了错误的权重目录或 Rank/world_size 参数）
                raise FileNotFoundError(
                    "Detected per-rank weight shards but missing expected shard for this rank. "
                    f"expected={shard} available={mp_files}"
                )
            # 如果既没有该 Rank 的分片也没有其他 Rank 的分片，
            # 则按默认模式加载（使用上面发现的全量权重文件）
    
    # ==================== 第3步：核心加载循环 ====================
    # 逐个加载权重文件中的所有参数
    loaded_count = 0
    for file in files:
        # 使用 safetensors 库开启文件句柄（加载到 CPU，后续模型会移到 GPU）
        with safe_open(file, "pt", "cpu") as f:
            for source_weight_name in f.keys():
                # ——————— 第3a步：权重名称转换 ———————
                # 将 HuggingFace 标准命名转换为 sparsevllm 内部命名（仅针对 DeepSeek 模型）
                # 
                # 转换示例：
                #   model.embed_tokens.weight → tok_embeddings.weight
                #   model.norm.weight → norm.weight
                #   model.lm_head.weight → output.weight
                #   model.layers.0.self_attn.q_proj.weight → layers.0.attn.wq.weight
                #   model.layers.0.mlp.gate_proj.weight → layers.0.ffn.w1.weight
                #
                # 对于非 DeepSeek 模型，_translate_deepseek_weight_name() 返回 None，
                # 此时 `or source_weight_name` 会使用原始权重名
                param_name = _translate_deepseek_weight_name(model, source_weight_name) or source_weight_name
                
                # ——————— 第3b步：检查打包模块处理 ———————
                # 如果该权重匹配到打包规则，则需要特殊处理（权重合并）
                packed_found = False
                for k in packed_modules_mapping:
                    if k in param_name:
                        # 匹配到打包规则
                        # 例如 param_name="layers.0.self_attn.q_proj.weight"，k="q_proj"
                        # packed_modules_mapping["q_proj"] = ("qkv_proj", 0)
                        # → v="qkv_proj", shard_id=0
                        v, shard_id = packed_modules_mapping[k]
                        
                        # 替换参数名：q_proj → qkv_proj
                        packed_param_name = param_name.replace(k, v)
                        # 例如：layers.0.self_attn.q_proj.weight → layers.0.self_attn.qkv_proj.weight
                        
                        # 获取融合后的目标参数（它比单个权重大，包含多个分量）
                        param = model.get_parameter(packed_param_name)
                        
                        # 调用该参数的自定义 weight_loader 方法
                        # weight_loader 负责将当前权重（q_proj）切片插入到融合权重的对应位置
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(source_weight_name), shard_id)
                        
                        loaded_count += 1
                        packed_found = True
                        break
                
                if packed_found:
                    # 已处理过打包逻辑，继续加载下一个权重
                    continue
                
                # ——————— 第3c步：常规加载（未匹配到打包规则） ———————
                # 直接获取参数并加载权重（不需要特殊处理）
                param = model.get_parameter(param_name)
                
                # 优先使用参数的自定义 weight_loader（某些特殊权重可能需要自定义处理逻辑）
                # 其次使用默认的复制逻辑（default_weight_loader = param.data.copy_(loaded_weight)）
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, f.get_tensor(source_weight_name))
                
                loaded_count += 1
    
    # ==================== 第4步：验证和反馈 ====================
    # 确保至少加载了一个权重
    # （防止路径错误、权重文件损坏或权重文件被删除导致的静默失败）
    assert loaded_count > 0, f"No weights were loaded from {path}"
    
    # 打印加载统计信息用于调试和日志
    # 便于用户确认权重加载的成功与否以及加载的权重数量
    print(f"Successfully loaded {loaded_count} weights from {path}")
