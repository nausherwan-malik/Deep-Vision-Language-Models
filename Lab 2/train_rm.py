from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import TaskType
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import HHDataConfig, ModelConfig, RewardModelConfig
from data.hh_rlhf import RewardModelCollator, load_hh_rlhf_examples
from eval import evaluate_reward_model_accuracy, reward_histogram
from model.loading import apply_lora, freeze_model, load_reward_model, load_reward_tokenizer, maybe_enable_gradient_checkpointing, model_memory_report
from utils import default_output_dir, ensure_dir, grad_norm, save_json, set_seed, set_torch_perf_flags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task C1: reward model training.")
    parser.add_argument("--reward-backbone-name", default=ModelConfig.reward_backbone_name)
    parser.add_argument("--dataset-name", default=HHDataConfig.dataset_name)
    parser.add_argument("--dataset-config", default=HHDataConfig.dataset_config)
    parser.add_argument("--train-split", default=HHDataConfig.train_split)
    parser.add_argument("--eval-split", default=HHDataConfig.eval_split)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=ModelConfig.max_length)
    parser.add_argument("--epochs", type=int, default=RewardModelConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=RewardModelConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=RewardModelConfig.learning_rate)
    parser.add_argument("--lambda-reg", type=float, default=RewardModelConfig.lambda_reg)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-head-only", action="store_true")
    return parser.parse_args()


def margin_ranking_loss(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor, lambda_reg: float) -> torch.Tensor:
    preference = -F.logsigmoid(chosen_scores - rejected_scores).mean()
    regularization = lambda_reg * (chosen_scores.square().mean() + rejected_scores.square().mean())
    return preference + regularization


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    set_torch_perf_flags()
    output_dir = ensure_dir(args.output_dir or default_output_dir("rm"))

    model_config = ModelConfig(reward_backbone_name=args.reward_backbone_name, max_length=args.max_length)
    tokenizer = load_reward_tokenizer(model_config)
    train_examples = load_hh_rlhf_examples(
        split=args.train_split,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        limit=args.max_train_examples,
    )
    eval_examples = load_hh_rlhf_examples(
        split=args.eval_split,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        limit=args.max_eval_examples,
    )

    train_loader = DataLoader(
        train_examples,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=RewardModelCollator(tokenizer, max_length=args.max_length),
    )
    eval_loader = DataLoader(
        eval_examples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=RewardModelCollator(tokenizer, max_length=args.max_length),
    )

    model = load_reward_model(model_config, num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id
    maybe_enable_gradient_checkpointing(model, enabled=model_config.use_gradient_checkpointing)
    if not args.train_head_only:
        model = apply_lora(model, model_config, task_type=TaskType.SEQ_CLS)
    else:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("score")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model_memory_report(model, "reward_model"))

    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    log_rows = []

    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            chosen_scores = model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
            ).logits.squeeze(-1)
            rejected_scores = model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
            ).logits.squeeze(-1)
            loss = margin_ranking_loss(chosen_scores, rejected_scores, args.lambda_reg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            row = {
                "epoch": epoch,
                "step": step,
                "loss": loss.item(),
                "train_pref_accuracy": (chosen_scores > rejected_scores).float().mean().item(),
                "grad_norm": grad_norm(model.parameters()),
            }
            log_rows.append(row)
            if step % 50 == 0:
                print(json.dumps(row, indent=2))

    eval_accuracy, chosen_scores, rejected_scores = evaluate_reward_model_accuracy(model, eval_loader, device)
    histograms = {
        "chosen_histogram": reward_histogram(chosen_scores),
        "rejected_histogram": reward_histogram(rejected_scores),
    }
    save_json(log_rows, Path(output_dir) / "metrics.json")
    save_json({"eval_preference_accuracy": eval_accuracy, **histograms}, Path(output_dir) / "eval.json")
    save_json(vars(args), Path(output_dir) / "args.json")
    model.save_pretrained(Path(output_dir) / "reward_model")
    tokenizer.save_pretrained(output_dir)
    print(json.dumps({"eval_preference_accuracy": eval_accuracy}, indent=2))


if __name__ == "__main__":
    main()
