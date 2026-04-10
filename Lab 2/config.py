from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    policy_name: str = "HuggingFaceTB/SmolLM2-360M"
    reward_backbone_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    max_length: int = 1024
    max_new_tokens: int = 128
    dtype: str = "bfloat16"
    use_gradient_checkpointing: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False


@dataclass
class HHDataConfig:
    dataset_name: str = "Anthropic/hh-rlhf"
    dataset_config: str = "harmless-base"
    train_split: str = "train"
    eval_split: str = "test"
    max_train_examples: int | None = None
    max_eval_examples: int | None = None
    num_workers: int = 2


@dataclass
class GSM8KConfig:
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    train_split: str = "train"
    eval_split: str = "test"
    max_train_examples: int | None = None
    max_eval_examples: int | None = None


@dataclass
class OptimizerConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_grad_norm: float = 1.0


@dataclass
class SFTConfig:
    epochs: int = 1
    batch_size: int = 8
    grad_accum_steps: int = 4
    eval_every_steps: int = 100
    learning_rate: float = 2e-4


@dataclass
class RewardModelConfig:
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-4
    lambda_reg: float = 1e-3


@dataclass
class RLConfig:
    method: str = "ppo"
    steps: int = 200
    prompts_per_step: int = 8
    grad_accum_steps: int = 1
    update_epochs: int = 4
    mini_batch_size: int = 8
    learning_rate: float = 1e-5
    value_learning_rate: float = 5e-5
    gamma: float = 1.0
    gae_lambda: float = 0.95
    beta: float = 0.1
    clip_epsilon: float = 0.2
    temperature: float = 0.7
    top_p: float = 0.9
    grpo_group_size: int = 4
    eval_every_steps: int = 25
    eval_prompts: int = 200
    sampled_kl: bool = True


@dataclass
class EvalConfig:
    eval_prompts: int = 200
    temperature: float = 0.0
    max_new_tokens: int = 128
    sample_table_size: int = 5

