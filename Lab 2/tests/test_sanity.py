from __future__ import annotations

import unittest

from data.gsm8k import extract_numeric_answer

try:
    import torch

    from alignment.ppo import compute_gae
except ModuleNotFoundError:
    torch = None
    compute_gae = None


class SanityTests(unittest.TestCase):
    def test_gsm8k_answer_extractor(self) -> None:
        self.assertEqual(extract_numeric_answer("#### 1,234"), "1234")
        self.assertEqual(extract_numeric_answer("The answer is -7.50"), "-7.5")
        self.assertEqual(extract_numeric_answer(r"\boxed{42}"), "42")
        self.assertIsNone(extract_numeric_answer("no numeric answer here"))

    @unittest.skipIf(torch is None, "torch is not installed in this environment")
    def test_gae_matches_hand_computation(self) -> None:
        rewards = torch.tensor([[0.05, -0.02, 1.6]], dtype=torch.float32)
        values = torch.tensor([[1.5, 1.55, 1.45]], dtype=torch.float32)
        token_mask = torch.tensor([[True, True, True]])
        advantages, returns = compute_gae(rewards, values, token_mask, gamma=1.0, gae_lambda=1.0)

        expected_advantages = torch.tensor([[0.13, 0.03, 0.15]], dtype=torch.float32)
        expected_returns = torch.tensor([[1.63, 1.58, 1.60]], dtype=torch.float32)

        self.assertTrue(torch.allclose(advantages, expected_advantages, atol=1e-5))
        self.assertTrue(torch.allclose(returns, expected_returns, atol=1e-5))

    @unittest.skipIf(torch is None, "torch is not installed in this environment")
    def test_ppo_clipping_blocks_positive_gradient_above_upper_bound(self) -> None:
        ratio = torch.tensor([1.5], dtype=torch.float32, requires_grad=True)
        advantage = torch.tensor([1.0], dtype=torch.float32)
        epsilon = 0.2

        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage
        surrogate = torch.minimum(unclipped, clipped)
        loss = -surrogate.mean()
        loss.backward()

        self.assertTrue(torch.allclose(surrogate, torch.tensor([1.2])))
        self.assertLess(abs(ratio.grad.item()), 1e-8)


if __name__ == "__main__":
    unittest.main()
