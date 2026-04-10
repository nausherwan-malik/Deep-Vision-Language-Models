from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .common import (
    GenerationBatch,
    compute_response_log_probs,
    compute_token_values,
    generate_responses,
    score_texts_with_reward_model,
)
from utils import masked_mean, masked_normalize


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    token_mask: torch.Tensor
    old_log_probs: torch.Tensor
    ref_log_probs: torch.Tensor
    values: torch.Tensor
    task_rewards: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    prompts: list[str]
    responses: list[str]

    def to(self, device: torch.device) -> "RolloutBatch":
        tensor_fields = {
            "input_ids",
            "attention_mask",
            "response_mask",
            "token_mask",
            "old_log_probs",
            "ref_log_probs",
            "values",
            "task_rewards",
            "rewards",
            "advantages",
            "returns",
        }
        updated = {}
        for field_name, value in self.__dict__.items():
            updated[field_name] = value.to(device) if field_name in tensor_fields else value
        return RolloutBatch(**updated)

    def select(self, indices: torch.Tensor) -> "RolloutBatch":
        selected = indices.tolist()
        return RolloutBatch(
            input_ids=self.input_ids[indices],
            attention_mask=self.attention_mask[indices],
            response_mask=self.response_mask[indices],
            token_mask=self.token_mask[indices],
            old_log_probs=self.old_log_probs[indices],
            ref_log_probs=self.ref_log_probs[indices],
            values=self.values[indices],
            task_rewards=self.task_rewards[indices],
            rewards=self.rewards[indices],
            advantages=self.advantages[indices],
            returns=self.returns[indices],
            prompts=[self.prompts[index] for index in selected],
            responses=[self.responses[index] for index in selected],
        )


def compose_rewards(
    token_mask: torch.Tensor,
    task_rewards: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    rewards = -beta * (old_log_probs - ref_log_probs) * token_mask.to(old_log_probs.dtype)
    lengths = token_mask.sum(dim=-1)
    for row_index, length in enumerate(lengths.tolist()):
        if length <= 0:
            continue
        rewards[row_index, length - 1] += task_rewards[row_index]
    return rewards


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    token_mask: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(rewards.size(0), device=rewards.device, dtype=rewards.dtype)
    next_values = torch.zeros(rewards.size(0), device=rewards.device, dtype=rewards.dtype)

    for time_index in reversed(range(rewards.size(1))):
        valid = token_mask[:, time_index].to(rewards.dtype)
        delta = rewards[:, time_index] + gamma * next_values - values[:, time_index]
        last_advantage = delta + gamma * gae_lambda * last_advantage
        last_advantage = last_advantage * valid
        advantages[:, time_index] = last_advantage
        next_values = torch.where(valid.bool(), values[:, time_index], next_values)
    returns = (advantages + values) * token_mask.to(values.dtype)
    return advantages, returns


@torch.no_grad()
def collect_ppo_rollout(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    value_model: torch.nn.Module,
    reward_model: torch.nn.Module,
    policy_tokenizer,
    reward_tokenizer,
    prompts: Sequence[str],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    beta: float,
    gamma: float,
    gae_lambda: float,
) -> RolloutBatch:
    generated = generate_responses(
        model=policy_model,
        tokenizer=policy_tokenizer,
        prompts=prompts,
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
    values, _ = compute_token_values(
        value_model,
        generated.input_ids,
        generated.attention_mask,
        generated.response_mask,
    )
    task_rewards = score_texts_with_reward_model(
        reward_model,
        reward_tokenizer,
        [prompt + response for prompt, response in zip(prompts, generated.responses)],
        max_length=max_length,
    )
    rewards = compose_rewards(token_mask, task_rewards, old_log_probs, ref_log_probs, beta)
    advantages, returns = compute_gae(rewards, values, token_mask, gamma=gamma, gae_lambda=gae_lambda)
    normalized_advantages = masked_normalize(advantages, token_mask)

    return RolloutBatch(
        input_ids=generated.input_ids.detach(),
        attention_mask=generated.attention_mask.detach(),
        response_mask=generated.response_mask.detach(),
        token_mask=token_mask.detach(),
        old_log_probs=old_log_probs.detach(),
        ref_log_probs=ref_log_probs.detach(),
        values=values.detach(),
        task_rewards=task_rewards.detach(),
        rewards=rewards.detach(),
        advantages=normalized_advantages.detach(),
        returns=returns.detach(),
        prompts=list(prompts),
        responses=generated.responses,
    )


def ppo_loss(
    policy_model: torch.nn.Module,
    value_model: torch.nn.Module,
    rollout_batch: RolloutBatch,
    clip_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    new_log_probs, token_mask = compute_response_log_probs(
        policy_model,
        rollout_batch.input_ids,
        rollout_batch.attention_mask,
        rollout_batch.response_mask,
    )
    new_values, _ = compute_token_values(
        value_model,
        rollout_batch.input_ids,
        rollout_batch.attention_mask,
        rollout_batch.response_mask,
    )
    ratio = torch.exp(new_log_probs - rollout_batch.old_log_probs)
    unclipped = ratio * rollout_batch.advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * rollout_batch.advantages
    surrogate = torch.minimum(unclipped, clipped)
    policy_loss = -masked_mean(surrogate, token_mask)
    value_error = (new_values - rollout_batch.returns.detach()) ** 2
    value_loss = masked_mean(value_error, token_mask)
    metrics = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "mean_reward": rollout_batch.task_rewards.mean().item(),
        "mean_kl": masked_mean(new_log_probs - rollout_batch.ref_log_probs, token_mask).item(),
        "ratio_min": ratio[token_mask].min().item() if token_mask.any() else 1.0,
        "ratio_max": ratio[token_mask].max().item() if token_mask.any() else 1.0,
    }
    return policy_loss, value_loss, metrics
