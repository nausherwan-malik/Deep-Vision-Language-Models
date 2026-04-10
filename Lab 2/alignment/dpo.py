from __future__ import annotations

import torch
import torch.nn.functional as F

from .common import compute_sequence_log_probs


def sequence_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    return compute_sequence_log_probs(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
    )


def dpo_loss(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    chosen_input_ids: torch.Tensor,
    chosen_attention_mask: torch.Tensor,
    chosen_response_mask: torch.Tensor,
    rejected_input_ids: torch.Tensor,
    rejected_attention_mask: torch.Tensor,
    rejected_response_mask: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    policy_chosen = sequence_log_probs(
        policy_model, chosen_input_ids, chosen_attention_mask, chosen_response_mask
    )
    policy_rejected = sequence_log_probs(
        policy_model, rejected_input_ids, rejected_attention_mask, rejected_response_mask
    )
    with torch.no_grad():
        ref_chosen = sequence_log_probs(ref_model, chosen_input_ids, chosen_attention_mask, chosen_response_mask)
        ref_rejected = sequence_log_probs(ref_model, rejected_input_ids, rejected_attention_mask, rejected_response_mask)

    delta_policy = policy_chosen - policy_rejected
    delta_ref = ref_chosen - ref_rejected
    z = beta * (delta_policy - delta_ref)
    loss = -F.logsigmoid(z).mean()
    metrics = {
        "loss": loss.item(),
        "z_margin": z.mean().item(),
        "preference_accuracy": (delta_policy > 0).float().mean().item(),
        "policy_margin": delta_policy.mean().item(),
        "ref_margin": delta_ref.mean().item(),
    }
    return loss, metrics
