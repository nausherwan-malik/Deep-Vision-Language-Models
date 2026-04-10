from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from utils import masked_mean


@dataclass
class GenerationBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    prompt_lengths: torch.Tensor
    response_lengths: torch.Tensor
    responses: list[str]


def gather_shifted_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def compute_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    return_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
        use_cache=False,
    )
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    token_mask = response_mask[:, 1:] & attention_mask[:, 1:].bool()
    token_log_probs = gather_shifted_log_probs(logits, labels)
    if return_logits:
        return token_log_probs, token_mask, logits
    return token_log_probs, token_mask


def compute_sequence_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    token_log_probs, token_mask = compute_response_log_probs(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
    )
    return (token_log_probs * token_mask.to(token_log_probs.dtype)).sum(dim=-1)


def compute_token_values(
    value_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = value_model(input_ids=input_ids, attention_mask=attention_mask)[:, :-1]
    token_mask = response_mask[:, 1:] & attention_mask[:, 1:].bool()
    return values, token_mask


def build_generation_masks(
    sequences: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, prompt_width = prompt_attention_mask.shape
    total_width = sequences.shape[1]
    prompt_lengths = prompt_attention_mask.sum(dim=-1)
    attention_mask = torch.zeros_like(sequences, dtype=torch.long)
    response_mask = torch.zeros_like(sequences, dtype=torch.bool)
    response_lengths = torch.zeros(batch_size, dtype=torch.long, device=sequences.device)

    for row_index in range(batch_size):
        prompt_length = int(prompt_lengths[row_index].item())
        left_pad = prompt_width - prompt_length
        generated = sequences[row_index, prompt_width:]

        if eos_token_id is None:
            response_length = generated.numel()
        else:
            eos_positions = (generated == eos_token_id).nonzero(as_tuple=False)
            response_length = int(eos_positions[0].item() + 1) if eos_positions.numel() > 0 else generated.numel()

        response_lengths[row_index] = response_length
        valid_until = prompt_width + response_length
        if prompt_length > 0:
            attention_mask[row_index, left_pad:prompt_width] = 1
        if response_length > 0:
            attention_mask[row_index, prompt_width:valid_until] = 1
            response_mask[row_index, prompt_width:valid_until] = True

    return attention_mask, response_mask, prompt_lengths, response_lengths


@torch.no_grad()
def generate_responses(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> GenerationBatch:
    device = next(model.parameters()).device
    prompt_batch = tokenizer(
        list(prompts),
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    prompt_batch = {key: value.to(device) for key, value in prompt_batch.items()}
    generation_kwargs = {
        **prompt_batch,
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    generation = model.generate(**generation_kwargs)
    sequences = generation.sequences
    attention_mask, response_mask, prompt_lengths, response_lengths = build_generation_masks(
        sequences=sequences,
        prompt_attention_mask=prompt_batch["attention_mask"],
        eos_token_id=tokenizer.eos_token_id,
    )
    responses = []
    for row_index in range(sequences.size(0)):
        response_tokens = sequences[row_index][response_mask[row_index]]
        responses.append(tokenizer.decode(response_tokens, skip_special_tokens=True))
    return GenerationBatch(
        input_ids=sequences,
        attention_mask=attention_mask,
        response_mask=response_mask,
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        responses=responses,
    )


@torch.no_grad()
def score_texts_with_reward_model(
    reward_model: torch.nn.Module,
    tokenizer,
    texts: Sequence[str],
    max_length: int,
) -> torch.Tensor:
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    device = next(reward_model.parameters()).device
    batch = tokenizer(
        list(texts),
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    logits = reward_model(**batch).logits.squeeze(-1)
    tokenizer.padding_side = original_padding_side
    return logits.detach()


def sampled_kl(log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
    return log_probs - ref_log_probs


def full_token_kl(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    policy_probs = policy_log_probs.exp()
    kl = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)
    return kl * token_mask.to(kl.dtype)


def masked_sequence_mean(values: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    return (values * token_mask.to(values.dtype)).sum(dim=-1) / token_mask.sum(dim=-1).clamp_min(1).to(values.dtype)


def batch_greedy_generate(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    max_length: int,
    max_new_tokens: int,
) -> list[str]:
    generations = generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
    )
    return generations.responses


def compute_reward_win_rate(
    reward_model: torch.nn.Module,
    reward_tokenizer,
    prompts: Sequence[str],
    baseline_responses: Sequence[str],
    candidate_responses: Sequence[str],
    max_length: int,
) -> float:
    baseline_scores = score_texts_with_reward_model(
        reward_model, reward_tokenizer, [p + r for p, r in zip(prompts, baseline_responses)], max_length
    )
    candidate_scores = score_texts_with_reward_model(
        reward_model, reward_tokenizer, [p + r for p, r in zip(prompts, candidate_responses)], max_length
    )
    return (candidate_scores > baseline_scores).float().mean().item()


def compute_mc_kl(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    max_length: int,
    max_new_tokens: int,
) -> float:
    generations = generate_responses(
        model=policy_model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
    )
    log_probs, token_mask = compute_response_log_probs(
        policy_model,
        generations.input_ids,
        generations.attention_mask,
        generations.response_mask,
    )
    with torch.no_grad():
        ref_log_probs, _ = compute_response_log_probs(
            ref_model,
            generations.input_ids,
            generations.attention_mask,
            generations.response_mask,
        )
    return masked_mean(sampled_kl(log_probs, ref_log_probs), token_mask).item()


def shuffle_minibatches(batch_size: int, mini_batch_size: int) -> list[torch.Tensor]:
    indices = torch.randperm(batch_size)
    return [indices[start : start + mini_batch_size] for start in range(0, batch_size, mini_batch_size)]
