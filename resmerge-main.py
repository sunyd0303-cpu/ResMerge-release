import argparse
import gc
import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


MERGE_DTYPE = torch.float32


def dtype_from_str(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def dtype_to_config_str(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    return str(dtype).replace("torch.", "")


def parse_gpu_ids(value: str) -> List[int]:
    ids = [int(x.strip()) for x in value.split(",") if x.strip()]
    return ids or [0]


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(idx)
                torch.cuda.empty_cache()
            except Exception:
                pass


def state_dict_summary(state_dict: Dict[str, torch.Tensor], name: str) -> None:
    total_params = 0
    total_bytes = 0
    for value in state_dict.values():
        if torch.is_tensor(value):
            total_params += value.numel()
            total_bytes += value.numel() * value.element_size()
    print(
        f"[Summary:{name}] tensors={len(state_dict)}, "
        f"params={total_params:,}, size={total_bytes / 1024**3:.2f} GB"
    )


def resolve_safetensors_index(model_dir: str) -> Tuple[Dict[str, str], Dict[str, List[int]]]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")
    key_to_file: Dict[str, str] = {}
    key_to_shape: Dict[str, List[int]] = {}

    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        for key, rel_path in index["weight_map"].items():
            key_to_file[key] = os.path.join(model_dir, rel_path)
    elif os.path.exists(single_path):
        with safe_open(single_path, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                key_to_file[key] = single_path
    else:
        raise FileNotFoundError(
            f"No safetensors checkpoint found under {model_dir}. "
            "Expected model.safetensors or model.safetensors.index.json."
        )

    shard_to_keys: Dict[str, List[str]] = {}
    for key, path in key_to_file.items():
        shard_to_keys.setdefault(path, []).append(key)

    for path, keys in shard_to_keys.items():
        with safe_open(path, framework="pt", device="cpu") as sf:
            for key in keys:
                try:
                    key_to_shape[key] = list(sf.get_slice(key).get_shape())
                except Exception:
                    tensor = sf.get_tensor(key)
                    key_to_shape[key] = list(tensor.shape)
                    del tensor

    return key_to_file, key_to_shape


def load_tensor(key_to_file: Dict[str, str], name: str) -> torch.Tensor:
    with safe_open(key_to_file[name], framework="pt", device="cpu") as sf:
        return sf.get_tensor(name).contiguous()


def resolve_and_validate_checkpoints(
    base_path: str,
    task_paths: List[str],
) -> Tuple[Dict[str, str], Dict[str, List[int]], List[Dict[str, str]]]:
    base_key_to_file, base_key_to_shape = resolve_safetensors_index(base_path)
    base_keys = list(base_key_to_file.keys())
    base_key_set = set(base_keys)
    task_key_to_files: List[Dict[str, str]] = []

    for path in task_paths:
        task_key_to_file, _ = resolve_safetensors_index(path)
        missing = [key for key in base_keys if key not in task_key_to_file]
        if missing:
            raise RuntimeError(f"Task model {path} is missing keys. Example: {missing[:5]}")
        extra = [key for key in task_key_to_file.keys() if key not in base_key_set]
        if extra:
            print(f"[Warn] Task model {path} has extra keys. Example: {extra[:5]}")
        task_key_to_files.append(task_key_to_file)

    return base_key_to_file, base_key_to_shape, task_key_to_files


def estimate_tensor_cost(shape: List[int]) -> int:
    if len(shape) == 0:
        return 1
    if len(shape) == 1:
        return max(1, shape[0])
    if len(shape) == 2:
        rows, cols = shape
        return max(1, rows * cols * min(rows, cols))
    return max(1, math.prod(shape))


def partition_keys_by_cost(
    keys: List[str],
    key_to_shape: Dict[str, List[int]],
    num_workers: int,
) -> List[List[str]]:
    key_costs = [(key, estimate_tensor_cost(key_to_shape[key])) for key in keys]
    key_costs.sort(key=lambda item: item[1], reverse=True)

    buckets = [[] for _ in range(num_workers)]
    loads = [0 for _ in range(num_workers)]
    for key, cost in key_costs:
        worker = min(range(num_workers), key=lambda idx: loads[idx])
        buckets[worker].append(key)
        loads[worker] += cost

    print("[Partition] estimated loads:")
    for idx, load in enumerate(loads):
        print(f"  worker {idx}: {load:,} cost units, {len(buckets[idx])} tensors")
    return buckets


def is_attention_param(name: str) -> bool:
    text = name.lower()
    return any(x in text for x in ("self_attn", ".attn.", "attention", "q_proj", "k_proj", "v_proj", "o_proj"))


def is_ffn_param(name: str) -> bool:
    text = name.lower()
    return any(x in text for x in ("mlp", "ffn", "feed_forward", "gate_proj", "up_proj", "down_proj"))


def is_norm_param(name: str) -> bool:
    text = name.lower()
    return any(x in text for x in ("norm", "ln_", "layernorm"))


def is_embedding_or_head_param(name: str) -> bool:
    text = name.lower()
    return any(x in text for x in ("embed_tokens", "wte", "lm_head", "word_embeddings"))


def module_family(name: str) -> str:
    if is_attention_param(name):
        return "attention"
    if is_ffn_param(name):
        return "ffn"
    if is_norm_param(name):
        return "norm"
    if is_embedding_or_head_param(name):
        return "embed_lm_head"
    return "other"


def tensor_mode_for_name(
    name: str,
    ndim: int,
) -> str:
    if is_embedding_or_head_param(name):
        return "ta"
    if is_norm_param(name):
        return "ta"
    if ndim == 1:
        return "ta"
    return "resmerge"


def apply_task_arithmetic_mean(
    base_weight: torch.Tensor,
    task_weights: List[torch.Tensor],
) -> torch.Tensor:
    acc = torch.zeros_like(base_weight, dtype=MERGE_DTYPE, device="cpu")
    inv_n = 1.0 / float(len(task_weights))
    for weight in task_weights:
        acc.add_(weight.to(dtype=MERGE_DTYPE), alpha=inv_n)
    out = acc.to(dtype=base_weight.dtype)
    del acc
    return out


@torch.inference_mode()
def exact_topk_svd_components(
    delta: torch.Tensor,
    topk: int,
    svd_driver: Optional[str] = None,
) -> List[torch.Tensor]:
    if delta.ndim != 2:
        raise ValueError(f"SVD decomposition expects a 2D tensor, got shape={tuple(delta.shape)}")
    if delta.is_cuda:
        u, s, vh = torch.linalg.svd(delta, full_matrices=False, driver=svd_driver)
    else:
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    rank = min(topk, s.numel())
    return [s[idx] * torch.outer(u[:, idx], vh[idx, :]) for idx in range(rank)]


@torch.inference_mode()
def decompose_head_tail(
    delta: torch.Tensor,
    rank_topk: int,
    svd_driver: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    delta = delta.to(MERGE_DTYPE)
    if delta.ndim != 2:
        raise ValueError(f"ResMerge only decomposes 2D tensors, got shape={tuple(delta.shape)}")
    if float(delta.norm().item()) <= 0.0:
        zero = torch.zeros_like(delta)
        return zero, zero

    head = torch.zeros_like(delta)
    for component in exact_topk_svd_components(delta, topk=rank_topk, svd_driver=svd_driver):
        head.add_(component)
    residual = delta - head
    return head, residual


@torch.inference_mode()
def mean_tensor(tensors: List[torch.Tensor]) -> torch.Tensor:
    out = torch.zeros_like(tensors[0])
    inv_n = 1.0 / float(len(tensors))
    for tensor in tensors:
        out.add_(tensor, alpha=inv_n)
    return out


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@torch.inference_mode()
def frobenius_cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-8) -> float:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    numerator = torch.dot(left_flat, right_flat)
    denominator = left_flat.norm() * right_flat.norm() + eps
    return float((numerator / denominator).item())


@torch.inference_mode()
def pairwise_cosine_stats(vectors: List[torch.Tensor], eps: float = 1e-8) -> Dict[str, float]:
    values: List[float] = []
    positives: List[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            value = frobenius_cosine(vectors[i], vectors[j], eps=eps)
            values.append(value)
            positives.append(max(0.0, value))
    if not values:
        return {"mean": 1.0, "pos_mean": 1.0}
    return {
        "mean": sum(values) / float(len(values)),
        "pos_mean": sum(positives) / float(len(positives)),
    }


@torch.inference_mode()
def build_src_a_residual(
    residuals: List[torch.Tensor],
    family: str,
    beta_min: float,
    beta_max: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    num_experts = len(residuals)
    device = residuals[0].device
    shape = residuals[0].shape
    residual_norms = torch.tensor(
        [float(residual.norm().item()) for residual in residuals],
        device=device,
        dtype=torch.float32,
    )
    if num_experts == 0 or float(residual_norms.max().item()) <= eps:
        return torch.zeros_like(residuals[0])

    q_list = [
        residual.reshape(-1) / (residual_norms[idx] + eps)
        for idx, residual in enumerate(residuals)
    ]
    pair_stats = pairwise_cosine_stats(q_list, eps=eps)

    q_sum = torch.zeros_like(q_list[0])
    for q_vec in q_list:
        q_sum.add_(q_vec)

    q_sum_norm = float(q_sum.norm().item())
    if q_sum_norm <= eps:
        return mean_tensor(residuals)
    q_src = q_sum / (q_sum_norm + eps)
    coherence = clamp01(
        sum(frobenius_cosine(q_vec, q_src, eps=eps) for q_vec in q_list) / float(num_experts)
    )

    # Paper definition: map s_R and c_R with (x + 1) / 2, then average.
    s_bar = clamp01((float(pair_stats["mean"]) + 1.0) * 0.5)
    c_bar = clamp01((coherence + 1.0) * 0.5)
    family_scale = 1.0 if family in {"attention", "ffn"} else 0.0
    score = clamp01(0.5 * (s_bar + c_bar) * family_scale)

    beta = float(beta_max) - (float(beta_max) - float(beta_min)) * score

    target_norm = float(residual_norms.mean().item())

    random_coherence = 1.0 / math.sqrt(float(max(1, num_experts)))
    if coherence <= random_coherence:
        return mean_tensor(residuals)

    # c_R is already in [0, 1], so use the raw spherical coherence here.
    effective_norm = (coherence ** beta) * target_norm
    return (q_src * effective_norm).reshape(shape)


@torch.inference_mode()
def head_agreement(
    heads: List[torch.Tensor],
    eps: float = 1e-8,
) -> float:
    normalized_heads = []
    for head in heads:
        norm = float(head.norm().item())
        if norm > eps:
            normalized_heads.append(head.reshape(-1) / norm)
    if not normalized_heads:
        return 0.0

    stats = pairwise_cosine_stats(normalized_heads)
    return clamp01(stats["pos_mean"])


@torch.inference_mode()
def merge_one_tensor_resmerge(
    name: str,
    base_weight_cpu: torch.Tensor,
    task_weights_cpu: List[torch.Tensor],
    rank_topk: int,
    device: torch.device,
    beta_min: float,
    beta_max: float,
    head_ratio_budget: float,
    svd_driver: Optional[str],
) -> torch.Tensor:
    if not torch.is_tensor(base_weight_cpu):
        return base_weight_cpu
    if not torch.is_floating_point(base_weight_cpu):
        return base_weight_cpu.clone()
    if base_weight_cpu.ndim not in (1, 2):
        raise ValueError(f"Only 1D/2D tensors are supported, got {name}: {tuple(base_weight_cpu.shape)}")

    mode = tensor_mode_for_name(
        name=name,
        ndim=base_weight_cpu.ndim,
    )
    if mode != "resmerge":
        return apply_task_arithmetic_mean(base_weight_cpu, task_weights_cpu)

    base = base_weight_cpu.to(device=device, dtype=MERGE_DTYPE, non_blocking=False)
    task_weights = [weight.to(device=device, dtype=MERGE_DTYPE, non_blocking=False) for weight in task_weights_cpu]
    deltas = [weight - base for weight in task_weights]

    heads: List[torch.Tensor] = []
    residuals: List[torch.Tensor] = []
    for delta in deltas:
        head, residual = decompose_head_tail(delta, rank_topk=rank_topk, svd_driver=svd_driver)
        heads.append(head)
        residuals.append(residual)

    residual_backbone = build_src_a_residual(
        residuals=residuals,
        family=module_family(name),
        beta_min=beta_min,
        beta_max=beta_max,
    )

    correction = torch.zeros_like(residual_backbone)
    reliability = head_agreement(heads)
    budget = float(head_ratio_budget) * reliability
    anchor_norm = float(residual_backbone.norm().item())
    mean_head = mean_tensor(heads)
    mean_head_norm = float(mean_head.norm().item())
    if budget > 0.0 and anchor_norm > 1e-8 and mean_head_norm > 1e-8:
        correction = (mean_head / mean_head_norm) * (budget * anchor_norm)

    merged = base + residual_backbone + correction
    out = merged.to(device="cpu", dtype=base_weight_cpu.dtype)

    del base, task_weights, deltas, heads, residuals, residual_backbone, correction, mean_head, merged
    return out


def merge_one_tensor_retry_cuda_oom(
    device: torch.device,
    retry_cuda_oom: bool,
    **kwargs: Any,
) -> torch.Tensor:
    try:
        return merge_one_tensor_resmerge(device=device, **kwargs)
    except torch.cuda.OutOfMemoryError:
        if device.type != "cuda" or not retry_cuda_oom:
            raise
        print(f"[Warn] CUDA OOM while merging {kwargs.get('name', '<unknown>')}; retrying once.")
        clear_device_cache(device)
        gc.collect()
        return merge_one_tensor_resmerge(device=device, **kwargs)


def worker_merge_stream(
    worker_rank: int,
    gpu_id: int,
    base_key_to_file: Dict[str, str],
    task_key_to_files: List[Dict[str, str]],
    assigned_keys: List[str],
    args_dict: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    torch.set_grad_enabled(False)
    partial_state_dict: Dict[str, torch.Tensor] = {}
    merge_kwargs = {
        key: value
        for key, value in args_dict.items()
        if key not in {"verbose_every", "empty_cache_every", "retry_cuda_oom"}
    }

    start = time.time()
    print(f"[Worker {worker_rank}] start on cuda:{gpu_id}; tensors={len(assigned_keys)}")
    for idx, name in enumerate(assigned_keys):
        base_weight = load_tensor(base_key_to_file, name)
        task_weights = [load_tensor(task_map, name) for task_map in task_key_to_files]
        merged = merge_one_tensor_retry_cuda_oom(
            name=name,
            base_weight_cpu=base_weight,
            task_weights_cpu=task_weights,
            device=device,
            retry_cuda_oom=args_dict["retry_cuda_oom"],
            **merge_kwargs,
        )
        partial_state_dict[name] = merged

        del base_weight, task_weights, merged
        if args_dict["empty_cache_every"] > 0 and (idx + 1) % args_dict["empty_cache_every"] == 0:
            clear_device_cache(device)
            gc.collect()
        if args_dict["verbose_every"] > 0 and (idx + 1) % args_dict["verbose_every"] == 0:
            elapsed = (time.time() - start) / 60.0
            print(f"[Worker {worker_rank}] processed {idx + 1}/{len(assigned_keys)} tensors in {elapsed:.2f} min")

    clear_device_cache(device)
    return partial_state_dict


def merge_models_stream_multi_gpu(
    base_path: str,
    task_paths: List[str],
    gpu_ids: List[int],
    args_dict: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    print("[Merge] Resolving safetensors indexes ...")
    base_key_to_file, base_key_to_shape, task_key_to_files = resolve_and_validate_checkpoints(base_path, task_paths)
    keys = list(base_key_to_file.keys())
    buckets = partition_keys_by_cost(keys, base_key_to_shape, len(gpu_ids))

    merged_state_dict: Dict[str, torch.Tensor] = {}
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = []
        for worker_rank, gpu_id in enumerate(gpu_ids):
            futures.append(
                executor.submit(
                    worker_merge_stream,
                    worker_rank,
                    gpu_id,
                    base_key_to_file,
                    task_key_to_files,
                    buckets[worker_rank],
                    args_dict,
                )
            )
        for future in as_completed(futures):
            partial_state_dict = future.result()
            overlap = set(merged_state_dict).intersection(partial_state_dict)
            if overlap:
                raise RuntimeError(f"Overlapping tensor keys across workers. Example: {list(overlap)[:5]}")
            merged_state_dict.update(partial_state_dict)
            del partial_state_dict
            gc.collect()

    return merged_state_dict


def merge_models_stream_single(
    base_path: str,
    task_paths: List[str],
    device: torch.device,
    args_dict: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    print("[Merge] Resolving safetensors indexes ...")
    base_key_to_file, _, task_key_to_files = resolve_and_validate_checkpoints(base_path, task_paths)
    keys = list(base_key_to_file.keys())
    merge_kwargs = {
        key: value
        for key, value in args_dict.items()
        if key not in {"verbose_every", "empty_cache_every", "retry_cuda_oom"}
    }

    merged_state_dict: Dict[str, torch.Tensor] = {}
    start = time.time()
    for idx, name in enumerate(keys):
        base_weight = load_tensor(base_key_to_file, name)
        task_weights = [load_tensor(task_map, name) for task_map in task_key_to_files]
        merged_state_dict[name] = merge_one_tensor_retry_cuda_oom(
            name=name,
            base_weight_cpu=base_weight,
            task_weights_cpu=task_weights,
            device=device,
            retry_cuda_oom=args_dict["retry_cuda_oom"],
            **merge_kwargs,
        )

        del base_weight, task_weights
        if args_dict["empty_cache_every"] > 0 and (idx + 1) % args_dict["empty_cache_every"] == 0:
            clear_device_cache(device)
            gc.collect()
        if args_dict["verbose_every"] > 0 and (idx + 1) % args_dict["verbose_every"] == 0:
            elapsed = (time.time() - start) / 60.0
            print(f"[Merge] processed {idx + 1}/{len(keys)} tensors in {elapsed:.2f} min")

    clear_device_cache(device)
    gc.collect()
    return merged_state_dict


def rewrite_output_config_like_base(base_path: str, output_dir: str, save_dtype: torch.dtype) -> None:
    base_config = os.path.join(base_path, "config.json")
    output_config = os.path.join(output_dir, "config.json")
    if not os.path.exists(base_config):
        return
    with open(base_config, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["_name_or_path"] = base_path
    config["torch_dtype"] = dtype_to_config_str(save_dtype)
    with open(output_config, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def copy_generation_config_if_exists(base_path: str, output_dir: str) -> None:
    src = os.path.join(base_path, "generation_config.json")
    dst = os.path.join(output_dir, "generation_config.json")
    if os.path.exists(src):
        shutil.copy2(src, dst)


def maybe_add_tied_lm_head(state_dict: Dict[str, torch.Tensor], config: Any) -> None:
    if (
        getattr(config, "tie_word_embeddings", False)
        and "lm_head.weight" not in state_dict
        and "model.embed_tokens.weight" in state_dict
    ):
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
        print("[Save] Added tied lm_head.weight from model.embed_tokens.weight for strict load.")


@torch.inference_mode()
def save_merged_model_and_tokenizer(
    base_path: str,
    merged_state_dict: Dict[str, torch.Tensor],
    output_dir: str,
    cache_dir: str,
    save_dtype: torch.dtype,
    max_shard_size: str,
    attn_implementation: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    state_dict_summary(merged_state_dict, name="merged")

    print(f"[Save] Loading base config: {base_path}")
    config = AutoConfig.from_pretrained(base_path, cache_dir=cache_dir, trust_remote_code=True)
    model_kwargs = {"trust_remote_code": True}
    if attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation

    print("[Save] Building model from config and loading merged weights ...")
    model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    maybe_add_tied_lm_head(merged_state_dict, config)
    load_result = model.load_state_dict(merged_state_dict, strict=True)
    if getattr(load_result, "missing_keys", None):
        raise RuntimeError(f"Missing keys while loading merged state dict: {load_result.missing_keys[:20]}")
    if getattr(load_result, "unexpected_keys", None):
        raise RuntimeError(f"Unexpected keys while loading merged state dict: {load_result.unexpected_keys[:20]}")

    model = model.to(dtype=save_dtype)
    if hasattr(model.config, "torch_dtype"):
        model.config.torch_dtype = save_dtype
    if hasattr(model.config, "_name_or_path"):
        model.config._name_or_path = base_path

    print(f"[Save] Saving model to {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size=max_shard_size)

    print("[Save] Saving tokenizer and config files ...")
    tokenizer = AutoTokenizer.from_pretrained(base_path, cache_dir=cache_dir, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    rewrite_output_config_like_base(base_path, output_dir, save_dtype)
    copy_generation_config_if_exists(base_path, output_dir)
    print("[Save] Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResMerge: SRC-A residual backbone with reliability-gated lightweight head correction."
    )
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--task_models", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cache")

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--merge_impl",
        type=str,
        default="stream",
        choices=["stream"],
        help="Kept for compatibility. ResMerge uses streaming tensor loading.",
    )
    parser.add_argument("--empty_cache_every", type=int, default=20)
    parser.add_argument("--retry_cuda_oom", action="store_true")

    parser.add_argument("--rank_topk", type=int, default=1)
    parser.add_argument("--src_beta_min", type=float, default=0.7)
    parser.add_argument("--src_beta_max", type=float, default=1.0)
    parser.add_argument("--head_ratio_budget", type=float, default=0.20)

    parser.add_argument("--save_dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--max_shard_size", type=str, default="5GB")
    parser.add_argument("--save_attn_implementation", type=str, default="eager", choices=["auto", "eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--svd_driver", type=str, default="auto", choices=["auto", "gesvdj", "gesvd"])
    parser.add_argument("--verbose_every", type=int, default=200)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rank_topk < 1:
        raise ValueError("--rank_topk must be >= 1")
    if args.src_beta_min < 0.0 or args.src_beta_max < 0.0:
        raise ValueError("--src_beta_min and --src_beta_max must be >= 0")
    if args.src_beta_min > args.src_beta_max:
        raise ValueError("--src_beta_min must be <= --src_beta_max")
    if args.head_ratio_budget < 0.0:
        raise ValueError("--head_ratio_budget must be >= 0")


def main() -> None:
    args = parse_args()
    validate_args(args)

    save_dtype = dtype_from_str(args.save_dtype)
    svd_driver = None if args.svd_driver == "auto" else args.svd_driver
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if args.num_workers > 0:
        gpu_ids = gpu_ids[: args.num_workers]

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        for gpu_id in gpu_ids:
            if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
                raise ValueError(f"Invalid GPU id {gpu_id}; visible GPUs={torch.cuda.device_count()}")
    else:
        gpu_ids = []

    args_dict: Dict[str, Any] = {
        "rank_topk": args.rank_topk,
        "beta_min": args.src_beta_min,
        "beta_max": args.src_beta_max,
        "head_ratio_budget": args.head_ratio_budget,
        "svd_driver": svd_driver,
        "verbose_every": args.verbose_every,
        "empty_cache_every": args.empty_cache_every,
        "retry_cuda_oom": args.retry_cuda_oom,
    }

    print("========== ResMerge Config ==========")
    print(f"Base model: {args.base_model}")
    print(f"Task models ({len(args.task_models)}):")
    for path in args.task_models:
        print(f"  - {path}")
    print(
        "method=SRC-A residual backbone + reliability-gated lightweight head correction; "
        f"rank_topk={args.rank_topk}; rho={args.head_ratio_budget}"
    )
    print(
        f"beta=[{args.src_beta_min}, {args.src_beta_max}], "
        "family_scales(attn/ffn/other)=1.0/1.0/0.0 (fixed)"
    )
    print("residual_mean_fallback: c_R <= 1/sqrt(num_experts)")
    print(f"device={args.device}, gpu_ids={gpu_ids if args.device == 'cuda' else 'N/A'}")
    print(f"save_dtype={save_dtype}, save_attn_implementation={args.save_attn_implementation}")
    print("=====================================")

    start = time.time()
    if args.device == "cuda" and len(gpu_ids) > 1:
        merged_state_dict = merge_models_stream_multi_gpu(
            base_path=args.base_model,
            task_paths=args.task_models,
            gpu_ids=gpu_ids,
            args_dict=args_dict,
        )
    else:
        if args.device == "cuda":
            torch.cuda.set_device(gpu_ids[0])
            device = torch.device(f"cuda:{gpu_ids[0]}")
        else:
            device = torch.device("cpu")
        merged_state_dict = merge_models_stream_single(
            base_path=args.base_model,
            task_paths=args.task_models,
            device=device,
            args_dict=args_dict,
        )
    elapsed = time.time() - start
    print(f"[Timer] Merge completed in {elapsed / 60.0:.2f} min")

    cleanup_cuda()
    save_merged_model_and_tokenizer(
        base_path=args.base_model,
        merged_state_dict=merged_state_dict,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        save_dtype=save_dtype,
        max_shard_size=args.max_shard_size,
        attn_implementation=args.save_attn_implementation,
    )


if __name__ == "__main__":
    main()
