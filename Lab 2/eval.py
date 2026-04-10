from __future__ import annotations

import math
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from alignment.common import batch_greedy_generate, compute_mc_kl, compute_reward_win_rate, score_texts_with_reward_model
from alignment.dpo import sequence_log_probs
from alignment.rlvr import format_compliance, verifiable_reward
from data.gsm8k import build_gsm8k_prompt
from utils import maybe_truncate_text


@torch.no_grad()
def evaluate_sft_perplexity(model, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for batch in dataloader:
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        )
        losses.append(outputs.loss.detach().float())
    if not losses:
        return float("nan")
    mean_loss = torch.stack(losses).mean().item()
    return math.exp(min(mean_loss, 20.0))


@torch.no_grad()
def evaluate_reward_model_accuracy(model, dataloader: DataLoader, device: torch.device) -> tuple[float, list[float], list[float]]:
    model.eval()
    chosen_scores = []
    rejected_scores = []
    for batch in dataloader:
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        chosen = model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        ).logits.squeeze(-1)
        rejected = model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        ).logits.squeeze(-1)
        chosen_scores.extend(chosen.detach().cpu().tolist())
        rejected_scores.extend(rejected.detach().cpu().tolist())
    accuracy = sum(c > r for c, r in zip(chosen_scores, rejected_scores)) / max(len(chosen_scores), 1)
    return accuracy, chosen_scores, rejected_scores


def reward_histogram(scores: Sequence[float], bins: int = 20) -> dict[str, list[float]]:
    if not scores:
        return {"bin_edges": [], "counts": []}
    tensor = torch.tensor(scores, dtype=torch.float32)
    min_value = tensor.min().item()
    max_value = tensor.max().item()
    if min_value == max_value:
        max_value += 1e-6
    hist = torch.histc(tensor, bins=bins, min=min_value, max=max_value)
    step = (max_value - min_value) / bins
    edges = [min_value + step * index for index in range(bins + 1)]
    return {"bin_edges": edges, "counts": hist.tolist()}


@torch.no_grad()
def evaluate_preference_accuracy(policy_model, dataloader: DataLoader, device: torch.device) -> float:
    policy_model.eval()
    correct = 0
    total = 0
    for batch in dataloader:
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        chosen = sequence_log_probs(
            policy_model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_response_mask"],
        )
        rejected = sequence_log_probs(
            policy_model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_response_mask"],
        )
        correct += (chosen > rejected).sum().item()
        total += chosen.numel()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_alignment_model(
    policy_model,
    ref_model,
    reward_model,
    policy_tokenizer,
    reward_tokenizer,
    prompts: Sequence[str],
    max_length: int,
    max_new_tokens: int,
) -> dict[str, float]:
    baseline_responses = batch_greedy_generate(ref_model, policy_tokenizer, prompts, max_length, max_new_tokens)
    candidate_responses = batch_greedy_generate(policy_model, policy_tokenizer, prompts, max_length, max_new_tokens)
    win_rate = compute_reward_win_rate(
        reward_model,
        reward_tokenizer,
        prompts,
        baseline_responses,
        candidate_responses,
        max_length=max_length,
    )
    kl = compute_mc_kl(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=policy_tokenizer,
        prompts=prompts,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    mean_reward = score_texts_with_reward_model(
        reward_model,
        reward_tokenizer,
        [prompt + response for prompt, response in zip(prompts, candidate_responses)],
        max_length=max_length,
    ).mean().item()
    return {"rm_win_rate_vs_sft": win_rate, "mc_kl": kl, "mean_rm_score": mean_reward}


@torch.no_grad()
def build_sample_response_table(
    models: dict[str, torch.nn.Module],
    tokenizer,
    reward_model,
    reward_tokenizer,
    prompts: Sequence[str],
    max_length: int,
    max_new_tokens: int,
) -> list[dict[str, str | float]]:
    rows = []
    generated_per_model = {
        name: batch_greedy_generate(model, tokenizer, prompts, max_length, max_new_tokens)
        for name, model in models.items()
    }
    for prompt_index, prompt in enumerate(prompts):
        row: dict[str, str | float] = {"prompt": maybe_truncate_text(prompt, 220)}
        for name, responses in generated_per_model.items():
            response = responses[prompt_index]
            score = score_texts_with_reward_model(
                reward_model,
                reward_tokenizer,
                [prompt + response],
                max_length=max_length,
            )[0].item()
            row[f"{name}_response"] = response
            row[f"{name}_rm_score"] = score
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate_gsm8k_pass_at_1(
    policy_model,
    tokenizer,
    examples,
    max_length: int,
    max_new_tokens: int,
) -> dict[str, float]:
    prompts = [build_gsm8k_prompt(example.question) for example in examples]
    responses = batch_greedy_generate(policy_model, tokenizer, prompts, max_length, max_new_tokens)
    rewards = verifiable_reward(responses, [example.gold_answer for example in examples])
    return {
        "pass_at_1": rewards.mean().item(),
        "format_compliance": format_compliance(responses),
        "mean_response_length": sum(len(response.split()) for response in responses) / max(len(responses), 1),
    }
