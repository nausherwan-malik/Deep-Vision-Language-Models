from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: str | Path) -> None:
    serializable = asdict(data) if is_dataclass(data) else data
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def bytes_to_gb(num_bytes: int) -> float:
    return num_bytes / float(1024 ** 3)


def gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return bytes_to_gb(torch.cuda.memory_allocated())


def max_gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return bytes_to_gb(torch.cuda.max_memory_allocated())


def format_parameter_count(num_params: int) -> str:
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    return str(num_params)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(eps)


def masked_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask.to(values.dtype)).sum()


def masked_normalize(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    valid_values = values[mask.bool()]
    if valid_values.numel() == 0:
        return values
    mean = valid_values.mean()
    std = valid_values.std(unbiased=False).clamp_min(eps)
    normalized = values.clone()
    normalized[mask.bool()] = (valid_values - mean) / std
    return normalized


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.detach().data.norm(2).item()
        squared_norm += param_norm ** 2
    return math.sqrt(squared_norm)


def to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [to_device(item, device) for item in batch]
    if isinstance(batch, tuple):
        return tuple(to_device(item, device) for item in batch)
    return batch


def detach_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu")


class Timer:
    def __init__(self) -> None:
        self.start_time = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time


def default_output_dir(name: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return ensure_dir(Path("outputs") / f"{name}-{timestamp}")


def maybe_truncate_text(text: str, limit: int = 180) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def set_torch_perf_flags() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True


def env_or_default(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value
