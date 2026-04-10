from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from data.gsm8k import GSM8KExample, build_gsm8k_prompt, extract_numeric_answer


@dataclass
class VerifiableExample:
    prompt: str
    gold_answer: str


def build_verifiable_examples(examples: Sequence[GSM8KExample]) -> list[VerifiableExample]:
    return [VerifiableExample(prompt=build_gsm8k_prompt(example.question), gold_answer=example.gold_answer) for example in examples]


def verifiable_reward(predictions: Sequence[str], gold_answers: Sequence[str]) -> torch.Tensor:
    rewards = []
    for prediction, gold_answer in zip(predictions, gold_answers):
        extracted = extract_numeric_answer(prediction)
        rewards.append(1.0 if extracted is not None and extracted == gold_answer else 0.0)
    return torch.tensor(rewards, dtype=torch.float32)


def rlvr_reward_fn(gold_answers: Sequence[str]):
    def _reward_fn(prompts: Sequence[str], responses: Sequence[str]) -> torch.Tensor:
        del prompts
        return verifiable_reward(responses, gold_answers)

    return _reward_fn


def format_compliance(predictions: Sequence[str]) -> float:
    return sum(extract_numeric_answer(prediction) is not None for prediction in predictions) / max(len(predictions), 1)
