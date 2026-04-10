from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .common import (
    compute_response_log_probs,
    full_token_kl,
    generate_responses,
    masked_sequence_mean,
    score_texts_with_reward_model,
)
from utils import masked_normalize


@dataclass
class GroupRolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    token_mask: torch.Tensor
    old_log_probs: torch.Tensor
    ref_log_probs: torch.Tensor
    advantages: torch.Tensor
    rewards: torch.Tensor
    prompts: list[str]
    responses: list[str]
    group_size: int
    degenerate_fraction: float

    def to(self, device: torch.device) -> "GroupRolloutBatch":
        tensor_fields = {
            "input_ids",
            "attention_mask",
            "response_mask",
            "token_mask",
            "old_log_probs",
            "ref_log_probs",
            "advantages",
            "rewards",
        }
        updated = {}
        for field_name, value in self.__dict__.items():
            updated[field_name] = value.to(device) if field_name in tensor_fields else value
        return GroupRolloutBatch(**updated)

    def select(self, indices: torch.Tensor) -> "GroupRolloutBatch":
        selected = indices.tolist()
        return GroupRolloutBatch(
            input_ids=self.input_ids[indices],
            attention_mask=self.attention_mask[indices],
            response_mask=self.response_mask[indices],
            token_mask=self.token_mask[indices],
            old_log_probs=self.old_log_probs[indices],
            ref_log_probs=self.ref_log_probs[indices],
            advantages=self.advantages[indices],
            rewards=self.rewards[indices],
            prompts=[self.prompts[index] for index in selected],
            responses=[self.responses[index] for index in selected],
            group_size=self.group_size,
            degenerate_fraction=self.degenerate_fraction,
        )


def _broadcast_group_advantages(
    rewards: torch.Tensor,
    token_mask: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, float]:
    prompt_count = rewards.numel() // group_size
    grouped = rewards.view(prompt_count, group_size)
    group_means = grouped.mean(dim=-1, keepdim=True)
    centered = grouped - group_means
    degenerate = (centered.abs().sum(dim=-1) == 0).float().mean().item()
    broadcast = centered.reshape(-1, 1).expand(-1, token_mask.size(1)).clone()
    broadcast = broadcast * token_mask.to(broadcast.dtype)
    standardized = masked_normalize(broadcast, token_mask)
    return standardized, degenerate


@torch.no_grad()
def collect_grpo_rollout(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    policy_tokenizer,
    prompts: Sequence[str],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    group_size: int,
    reward_fn: Callable[[Sequence[str], Sequence[str]], torch.Tensor],
) -> GroupRolloutBatch:
    repeated_prompts = [prompt for prompt in prompts for _ in range(group_size)]
    generated = generate_responses(
        model=policy_model,
        tokenizer=policy_tokenizer,
        prompts=repeated_prompts,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )
    old_log_probs, token_mask = compute_response_log_probs(
        policy_model,
        generated.input_ids,
        generated.attention_mask,
        generated.response_mask,
    )
    ref_log_probs, _ = compute_response_log_probs(
        ref_model,
        generated.input_ids,
        generated.attention_mask,
        generated.response_mask,
    )
    rewards = reward_fn(repeated_prompts, generated.responses).to(old_log_probs.device)
    advantages, degenerate_fraction = _broadcast_group_advantages(rewards, token_mask, group_size)
    return GroupRolloutBatch(
        input_ids=generated.input_ids.detach(),
        attention_mask=generated.attention_mask.detach(),
        response_mask=generated.response_mask.detach(),
        token_mask=token_mask.detach(),
        old_log_probs=old_log_probs.detach(),
        ref_log_probs=ref_log_probs.detach(),
        advantages=advantages.detach(),
        rewards=rewards.detach(),
        prompts=repeated_prompts,
        responses=generated.responses,
        group_size=group_size,
        degenerate_fraction=degenerate_fraction,
    )


def reward_model_reward_fn(reward_model, reward_tokenizer, max_length: int):
    @torch.no_grad()
    def _reward_fn(prompts: Sequence[str], responses: Sequence[str]) -> torch.Tensor:
        scores = score_texts_with_reward_model(
            reward_model=reward_model,
            tokenizer=reward_tokenizer,
            texts=[prompt + response for prompt, response in zip(prompts, responses)],
            max_length=max_length,
        )
        return scores

    return _reward_fn


def grpo_loss(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    rollout_batch: GroupRolloutBatch,
    clip_epsilon: float,
    beta: float,
    sampled_kl: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if sampled_kl:
        new_log_probs, token_mask, policy_logits = compute_response_log_probs(
            policy_model,
            rollout_batch.input_ids,
            rollout_batch.attention_mask,
            rollout_batch.response_mask,
            return_logits=True,
        )
        with torch.no_grad():
            _, _, ref_logits = compute_response_log_probs(
                ref_model,
                rollout_batch.input_ids,
                rollout_batch.attention_mask,
                rollout_batch.response_mask,
                return_logits=True,
            )
        token_kl = (new_log_probs - rollout_batch.ref_log_probs) * token_mask.to(new_log_probs.dtype)
    else:
        new_log_probs, token_mask, policy_logits = compute_response_log_probs(
            policy_model,
            rollout_batch.input_ids,
            rollout_batch.attention_mask,
            rollout_batch.response_mask,
            return_logits=True,
        )
        with torch.no_grad():
            _, _, ref_logits = compute_response_log_probs(
                ref_model,
                rollout_batch.input_ids,
                rollout_batch.attention_mask,
                rollout_batch.response_mask,
                return_logits=True,
            )
        token_kl = full_token_kl(policy_logits, ref_logits, token_mask)

    ratio = torch.exp(new_log_probs - rollout_batch.old_log_probs)
    unclipped = ratio * rollout_batch.advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * rollout_batch.advantages
    surrogate = torch.minimum(unclipped, clipped)
    sequence_surrogate = masked_sequence_mean(surrogate, token_mask)
    sequence_kl = masked_sequence_mean(token_kl, token_mask)
    loss = -(sequence_surrogate - beta * sequence_kl).mean()
    metrics = {
        "loss": loss.item(),
        "mean_reward": rollout_batch.rewards.mean().item(),
        "mean_kl": sequence_kl.mean().item(),
        "degenerate_fraction": rollout_batch.degenerate_fraction,
        "ratio_min": ratio[token_mask].min().item() if token_mask.any() else 1.0,
        "ratio_max": ratio[token_mask].max().item() if token_mask.any() else 1.0,
    }
    return loss, metrics
