from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
from peft import TaskType
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import HHDataConfig, ModelConfig, SFTConfig
from data.hh_rlhf import SFTCollator, load_hh_rlhf_examples, preview_examples
from eval import evaluate_sft_perplexity
from model.loading import apply_lora, load_policy_model, load_policy_tokenizer, maybe_enable_gradient_checkpointing, model_memory_report
from utils import default_output_dir, ensure_dir, grad_norm, save_json, set_seed, set_torch_perf_flags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task C2: SFT warm-up for the policy model.")
    parser.add_argument("--policy-name", default=ModelConfig.policy_name)
    parser.add_argument("--dataset-name", default=HHDataConfig.dataset_name)
    parser.add_argument("--dataset-config", default=HHDataConfig.dataset_config)
    parser.add_argument("--train-split", default=HHDataConfig.train_split)
    parser.add_argument("--eval-split", default=HHDataConfig.eval_split)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=ModelConfig.max_length)
    parser.add_argument("--epochs", type=int, default=SFTConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=SFTConfig.batch_size)
    parser.add_argument("--grad-accum-steps", type=int, default=SFTConfig.grad_accum_steps)
    parser.add_argument("--learning-rate", type=float, default=SFTConfig.learning_rate)
    parser.add_argument("--eval-every-steps", type=int, default=SFTConfig.eval_every_steps)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    set_torch_perf_flags()
    output_dir = ensure_dir(args.output_dir or default_output_dir("sft"))

    model_config = ModelConfig(policy_name=args.policy_name, max_length=args.max_length)
    tokenizer = load_policy_tokenizer(model_config)
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
    print(preview_examples(train_examples, limit=3))

    train_loader = DataLoader(
        train_examples,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=SFTCollator(tokenizer, max_length=args.max_length),
    )
    eval_loader = DataLoader(
        eval_examples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=SFTCollator(tokenizer, max_length=args.max_length),
    )

    model = load_policy_model(model_config)
    maybe_enable_gradient_checkpointing(model, enabled=model_config.use_gradient_checkpointing)
    model = apply_lora(model, model_config, task_type=TaskType.CAUSAL_LM)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model_memory_report(model, "policy"))

    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    global_step = 0
    log_rows = []
    model.train()

    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()

            if (global_step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            row = {
                "epoch": epoch,
                "step": global_step,
                "loss": outputs.loss.item(),
                "grad_norm": grad_norm(model.parameters()),
            }
            if global_step % args.eval_every_steps == 0:
                row["eval_perplexity"] = evaluate_sft_perplexity(model, eval_loader, device)
                print(json.dumps(row, indent=2))
            log_rows.append(row)

    sample_prompts = [example.prompt for example in itertools.islice(eval_examples, 5)]
    generated = []
    model.eval()
    with torch.no_grad():
        for prompt in sample_prompts:
            batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            output = model.generate(
                **batch,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated.append(tokenizer.decode(output[0][batch["input_ids"].size(1) :], skip_special_tokens=True))

    adapter_dir = Path(output_dir) / "policy_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)
    save_json({"samples": [{"prompt": prompt, "response": response} for prompt, response in zip(sample_prompts, generated)]}, Path(output_dir) / "samples.json")
    save_json(log_rows, Path(output_dir) / "metrics.json")
    save_json(vars(args), Path(output_dir) / "args.json")
    print(f"Saved SFT adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
