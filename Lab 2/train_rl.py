from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import PeftModel, TaskType
from torch.optim import AdamW
from torch.utils.data import DataLoader

from alignment.common import shuffle_minibatches
from alignment.dpo import dpo_loss
from alignment.grpo import collect_grpo_rollout, grpo_loss, reward_model_reward_fn
from alignment.ppo import collect_ppo_rollout, ppo_loss
from alignment.rlvr import build_verifiable_examples, format_compliance, rlvr_reward_fn, verifiable_reward
from config import EvalConfig, GSM8KConfig, HHDataConfig, ModelConfig, RLConfig
from data.gsm8k import load_gsm8k_examples
from data.hh_rlhf import DPOCollator, PromptDataset, load_hh_rlhf_examples
from eval import evaluate_alignment_model, evaluate_gsm8k_pass_at_1, evaluate_preference_accuracy
from model.heads import ValueModel
from model.loading import (
    apply_lora,
    freeze_model,
    load_policy_model,
    load_policy_tokenizer,
    load_reference_policy,
    load_reward_model,
    load_reward_tokenizer,
    load_value_backbone,
    maybe_enable_gradient_checkpointing,
    model_memory_report,
)
from utils import default_output_dir, ensure_dir, grad_norm, save_json, set_seed, set_torch_perf_flags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tasks C3-C8: PPO, DPO, GRPO, and RLVR training.")
    parser.add_argument("--method", choices=["ppo", "dpo", "grpo", "rlvr"], default=RLConfig.method)
    parser.add_argument("--policy-name", default=ModelConfig.policy_name)
    parser.add_argument("--reward-backbone-name", default=ModelConfig.reward_backbone_name)
    parser.add_argument("--sft-adapter-dir", required=True)
    parser.add_argument("--reward-model-dir", default=None)
    parser.add_argument("--dataset-name", default=HHDataConfig.dataset_name)
    parser.add_argument("--dataset-config", default=HHDataConfig.dataset_config)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=ModelConfig.max_length)
    parser.add_argument("--max-new-tokens", type=int, default=ModelConfig.max_new_tokens)
    parser.add_argument("--steps", type=int, default=RLConfig.steps)
    parser.add_argument("--prompts-per-step", type=int, default=RLConfig.prompts_per_step)
    parser.add_argument("--mini-batch-size", type=int, default=RLConfig.mini_batch_size)
    parser.add_argument("--update-epochs", type=int, default=RLConfig.update_epochs)
    parser.add_argument("--learning-rate", type=float, default=RLConfig.learning_rate)
    parser.add_argument("--value-learning-rate", type=float, default=RLConfig.value_learning_rate)
    parser.add_argument("--beta", type=float, default=RLConfig.beta)
    parser.add_argument("--clip-epsilon", type=float, default=RLConfig.clip_epsilon)
    parser.add_argument("--gamma", type=float, default=RLConfig.gamma)
    parser.add_argument("--gae-lambda", type=float, default=RLConfig.gae_lambda)
    parser.add_argument("--temperature", type=float, default=RLConfig.temperature)
    parser.add_argument("--top-p", type=float, default=RLConfig.top_p)
    parser.add_argument("--group-size", type=int, default=RLConfig.grpo_group_size)
    parser.add_argument("--eval-every-steps", type=int, default=RLConfig.eval_every_steps)
    parser.add_argument("--eval-prompts", type=int, default=EvalConfig.eval_prompts)
    parser.add_argument("--full-kl", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--gsm8k-train-limit", type=int, default=None)
    parser.add_argument("--gsm8k-eval-limit", type=int, default=128)
    return parser.parse_args()


def sample_prompts(pool: list[str], batch_size: int) -> list[str]:
    if len(pool) >= batch_size:
        return random.sample(pool, batch_size)
    return random.choices(pool, k=batch_size)


def append_log(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def load_trainable_policy(model_config: ModelConfig, sft_adapter_dir: str) -> torch.nn.Module:
    base = load_policy_model(model_config)
    model = PeftModel.from_pretrained(base, sft_adapter_dir, is_trainable=True)
    model.config.use_cache = False
    return model


def load_frozen_reward_model(model_config: ModelConfig, reward_model_dir: str, device: torch.device):
    reward_tokenizer = load_reward_tokenizer(model_config)
    reward_model = load_reward_model(model_config, num_labels=1)
    reward_model = PeftModel.from_pretrained(reward_model, reward_model_dir, is_trainable=False)
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
    freeze_model(reward_model)
    reward_model.to(device)
    return reward_model, reward_tokenizer


def run_dpo(args, output_dir: Path, device: torch.device) -> None:
    model_config = ModelConfig(
        policy_name=args.policy_name,
        reward_backbone_name=args.reward_backbone_name,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )
    policy_tokenizer = load_policy_tokenizer(model_config)
    policy_model = load_trainable_policy(model_config, args.sft_adapter_dir).to(device)
    ref_model = load_reference_policy(model_config, adapter_path=args.sft_adapter_dir).to(device)
    reward_model, reward_tokenizer = load_frozen_reward_model(model_config, args.reward_model_dir, device)
    print(model_memory_report(policy_model, "policy"))

    train_examples = load_hh_rlhf_examples(
        split=HHDataConfig.train_split,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        limit=args.max_train_examples,
    )
    eval_examples = load_hh_rlhf_examples(
        split=HHDataConfig.eval_split,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        limit=args.max_eval_examples,
    )
    train_loader = DataLoader(
        train_examples,
        batch_size=args.mini_batch_size,
        shuffle=True,
        collate_fn=DPOCollator(policy_tokenizer, max_length=args.max_length),
    )
    eval_loader = DataLoader(
        eval_examples,
        batch_size=args.mini_batch_size,
        shuffle=False,
        collate_fn=DPOCollator(policy_tokenizer, max_length=args.max_length),
    )
    optimizer = AdamW((parameter for parameter in policy_model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    metrics_path = output_dir / "metrics.jsonl"

    global_step = 0
    for batch in train_loader:
        global_step += 1
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        loss, metrics = dpo_loss(
            policy_model=policy_model,
            ref_model=ref_model,
            chosen_input_ids=batch["chosen_input_ids"],
            chosen_attention_mask=batch["chosen_attention_mask"],
            chosen_response_mask=batch["chosen_response_mask"],
            rejected_input_ids=batch["rejected_input_ids"],
            rejected_attention_mask=batch["rejected_attention_mask"],
            rejected_response_mask=batch["rejected_response_mask"],
            beta=args.beta,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

        row = {"step": global_step, **metrics, "grad_norm": grad_norm(policy_model.parameters())}
        if global_step % args.eval_every_steps == 0:
            eval_prompts = [example.prompt for example in eval_examples[: args.eval_prompts]]
            row.update(
                evaluate_alignment_model(
                    policy_model,
                    ref_model,
                    reward_model,
                    policy_tokenizer,
                    reward_tokenizer,
                    eval_prompts,
                    max_length=args.max_length,
                    max_new_tokens=args.max_new_tokens,
                )
            )
            row["heldout_preference_accuracy"] = evaluate_preference_accuracy(policy_model, eval_loader, device)
        append_log(metrics_path, row)
        if global_step >= args.steps:
            break

    policy_model.save_pretrained(output_dir / "policy_adapter")


def run_ppo(args, output_dir: Path, device: torch.device) -> None:
    model_config = ModelConfig(
        policy_name=args.policy_name,
        reward_backbone_name=args.reward_backbone_name,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )
    policy_tokenizer = load_policy_tokenizer(model_config)
    policy_model = load_trainable_policy(model_config, args.sft_adapter_dir).to(device)
    ref_model = load_reference_policy(model_config, adapter_path=args.sft_adapter_dir).to(device)
    reward_model, reward_tokenizer = load_frozen_reward_model(model_config, args.reward_model_dir, device)

    value_backbone = load_value_backbone(model_config)
    maybe_enable_gradient_checkpointing(value_backbone, enabled=model_config.use_gradient_checkpointing)
    value_backbone = apply_lora(value_backbone, model_config, task_type=TaskType.FEATURE_EXTRACTION)
    value_model = ValueModel(value_backbone).to(device)
    print(model_memory_report(policy_model, "policy"))
    print(model_memory_report(value_model, "value_model"))

    examples = load_hh_rlhf_examples(
        split=HHDataConfig.train_split,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        limit=args.max_train_examples,
    )
    eval_examples = load_hh_rlhf_examples(
        split=HHDataConfig.eval_split,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        limit=args.max_eval_examples,
    )
    prompt_pool = [example.prompt for example in examples]
    eval_prompts = [example.prompt for example in eval_examples[: args.eval_prompts]]
    policy_optimizer = AdamW((parameter for parameter in policy_model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    value_optimizer = AdamW((parameter for parameter in value_model.parameters() if parameter.requires_grad), lr=args.value_learning_rate)
    metrics_path = output_dir / "metrics.jsonl"

    for step in range(1, args.steps + 1):
        prompts = sample_prompts(prompt_pool, args.prompts_per_step)
        policy_model.eval()
        value_model.eval()
        ref_model.eval()
        reward_model.eval()
        rollout = collect_ppo_rollout(
            policy_model=policy_model,
            ref_model=ref_model,
            value_model=value_model,
            reward_model=reward_model,
            policy_tokenizer=policy_tokenizer,
            reward_tokenizer=reward_tokenizer,
            prompts=prompts,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            beta=args.beta,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        ).to(torch.device("cpu"))
        policy_model.train()
        value_model.train()

        start_time = time.perf_counter()
        last_metrics = {}
        for _ in range(args.update_epochs):
            for indices in shuffle_minibatches(len(prompts), args.mini_batch_size):
                mini_batch = rollout.select(indices).to(device)
                policy_loss_value, value_loss_value, metrics = ppo_loss(
                    policy_model=policy_model,
                    value_model=value_model,
                    rollout_batch=mini_batch,
                    clip_epsilon=args.clip_epsilon,
                )
                total_loss = policy_loss_value + 0.5 * value_loss_value
                policy_optimizer.zero_grad(set_to_none=True)
                value_optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(value_model.parameters(), 1.0)
                policy_optimizer.step()
                value_optimizer.step()
                last_metrics = metrics

        row = {
            "step": step,
            **last_metrics,
            "policy_grad_norm": grad_norm(policy_model.parameters()),
            "value_grad_norm": grad_norm(value_model.parameters()),
            "step_time_sec": time.perf_counter() - start_time,
        }
        if step % args.eval_every_steps == 0:
            row.update(
                evaluate_alignment_model(
                    policy_model,
                    ref_model,
                    reward_model,
                    policy_tokenizer,
                    reward_tokenizer,
                    eval_prompts,
                    max_length=args.max_length,
                    max_new_tokens=args.max_new_tokens,
                )
            )
        append_log(metrics_path, row)

    policy_model.save_pretrained(output_dir / "policy_adapter")
    value_model.backbone.save_pretrained(output_dir / "value_backbone_adapter")


def run_grpo(args, output_dir: Path, device: torch.device, rlvr: bool = False) -> None:
    model_config = ModelConfig(
        policy_name=args.policy_name,
        reward_backbone_name=args.reward_backbone_name,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )
    policy_tokenizer = load_policy_tokenizer(model_config)
    policy_model = load_trainable_policy(model_config, args.sft_adapter_dir).to(device)
    ref_model = load_reference_policy(model_config, adapter_path=args.sft_adapter_dir).to(device)
    print(model_memory_report(policy_model, "policy"))

    metrics_path = output_dir / "metrics.jsonl"

    if rlvr:
        train_examples = build_verifiable_examples(
            load_gsm8k_examples(split=GSM8KConfig.train_split, limit=args.gsm8k_train_limit)
        )
        eval_examples = load_gsm8k_examples(split=GSM8KConfig.eval_split, limit=args.gsm8k_eval_limit)
        prompt_pool = train_examples
        reward_model = None
        reward_tokenizer = None
    else:
        reward_model, reward_tokenizer = load_frozen_reward_model(model_config, args.reward_model_dir, device)
        train_examples = load_hh_rlhf_examples(
            split=HHDataConfig.train_split,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            limit=args.max_train_examples,
        )
        eval_examples = load_hh_rlhf_examples(
            split=HHDataConfig.eval_split,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            limit=args.max_eval_examples,
        )
        prompt_pool = [example.prompt for example in train_examples]
    optimizer = AdamW((parameter for parameter in policy_model.parameters() if parameter.requires_grad), lr=args.learning_rate)

    for step in range(1, args.steps + 1):
        if rlvr:
            batch_examples = sample_prompts(prompt_pool, args.prompts_per_step)
            prompts = [example.prompt for example in batch_examples]
            gold_answers = [example.gold_answer for example in batch_examples for _ in range(args.group_size)]
            reward_fn = rlvr_reward_fn(gold_answers)
        else:
            prompts = sample_prompts(prompt_pool, args.prompts_per_step)
            reward_fn = reward_model_reward_fn(reward_model, reward_tokenizer, max_length=args.max_length)

        policy_model.eval()
        ref_model.eval()
        if reward_model is not None:
            reward_model.eval()
        rollout = collect_grpo_rollout(
            policy_model=policy_model,
            ref_model=ref_model,
            policy_tokenizer=policy_tokenizer,
            prompts=prompts,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            group_size=args.group_size,
            reward_fn=reward_fn,
        ).to(torch.device("cpu"))
        policy_model.train()

        start_time = time.perf_counter()
        last_metrics = {}
        total_sequences = len(prompts) * args.group_size
        for _ in range(args.update_epochs):
            for indices in shuffle_minibatches(total_sequences, args.mini_batch_size):
                mini_batch = rollout.select(indices).to(device)
                loss, metrics = grpo_loss(
                    policy_model=policy_model,
                    ref_model=ref_model,
                    rollout_batch=mini_batch,
                    clip_epsilon=args.clip_epsilon,
                    beta=args.beta,
                    sampled_kl=not args.full_kl,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                last_metrics = metrics

        row = {
            "step": step,
            **last_metrics,
            "grad_norm": grad_norm(policy_model.parameters()),
            "step_time_sec": time.perf_counter() - start_time,
        }
        if rlvr and step % args.eval_every_steps == 0:
            eval_metrics = evaluate_gsm8k_pass_at_1(
                policy_model,
                policy_tokenizer,
                eval_examples,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
            )
            row.update(eval_metrics)
        elif not rlvr and step % args.eval_every_steps == 0:
            eval_prompts = [example.prompt for example in eval_examples[: args.eval_prompts]]
            row.update(
                evaluate_alignment_model(
                    policy_model,
                    ref_model,
                    reward_model,
                    policy_tokenizer,
                    reward_tokenizer,
                    eval_prompts,
                    max_length=args.max_length,
                    max_new_tokens=args.max_new_tokens,
                )
            )
        append_log(metrics_path, row)

    policy_model.save_pretrained(output_dir / "policy_adapter")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    set_torch_perf_flags()
    output_dir = ensure_dir(args.output_dir or default_output_dir(args.method))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.method in {"dpo", "ppo", "grpo"} and args.reward_model_dir is None:
        raise ValueError("--reward-model-dir is required for DPO, PPO, and GRPO.")

    if args.method == "dpo":
        run_dpo(args, output_dir, device)
    elif args.method == "ppo":
        run_ppo(args, output_dir, device)
    elif args.method == "grpo":
        run_grpo(args, output_dir, device, rlvr=False)
    else:
        run_grpo(args, output_dir, device, rlvr=True)

    save_json(vars(args), output_dir / "args.json")


if __name__ == "__main__":
    main()
