from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from config import ModelConfig
from utils import count_parameters, format_parameter_count, gpu_memory_gb, resolve_dtype


def build_quantization_config(config: ModelConfig) -> BitsAndBytesConfig | None:
    if config.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=resolve_dtype(config.dtype),
        )
    if config.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def _common_model_kwargs(config: ModelConfig) -> dict:
    quantization_config = build_quantization_config(config)
    kwargs = {
        "torch_dtype": resolve_dtype(config.dtype),
        "trust_remote_code": config.trust_remote_code,
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    return kwargs


def load_policy_tokenizer(config: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.policy_name, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return tokenizer


def load_reward_tokenizer(config: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.reward_backbone_name,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    return tokenizer


def maybe_enable_gradient_checkpointing(model: torch.nn.Module, enabled: bool = True) -> torch.nn.Module:
    if not enabled:
        return model
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def apply_lora(
    model: torch.nn.Module,
    config: ModelConfig,
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> torch.nn.Module:
    if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        task_type=task_type,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    return get_peft_model(model, peft_config)


def load_policy_model(config: ModelConfig) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(config.policy_name, **_common_model_kwargs(config))
    model.config.use_cache = False
    return model


def load_reference_policy(
    config: ModelConfig,
    adapter_path: str | Path | None = None,
) -> torch.nn.Module:
    base_model = load_policy_model(config)
    if adapter_path is not None:
        base_model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
    return freeze_model(base_model)


def load_reward_model(config: ModelConfig, num_labels: int = 1) -> torch.nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_backbone_name,
        num_labels=num_labels,
        **_common_model_kwargs(config),
    )
    return model


def load_value_backbone(config: ModelConfig) -> torch.nn.Module:
    model = AutoModel.from_pretrained(config.reward_backbone_name, **_common_model_kwargs(config))
    return model


def attach_adapter_checkpoint(
    base_model: torch.nn.Module,
    adapter_path: str | Path,
    is_trainable: bool = False,
) -> torch.nn.Module:
    return PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=is_trainable)


def model_memory_report(model: torch.nn.Module, name: str) -> str:
    total_params, trainable_params = count_parameters(model)
    return (
        f"{name}: total={format_parameter_count(total_params)}, "
        f"trainable={format_parameter_count(trainable_params)}, "
        f"gpu_memory={gpu_memory_gb():.2f} GB"
    )
