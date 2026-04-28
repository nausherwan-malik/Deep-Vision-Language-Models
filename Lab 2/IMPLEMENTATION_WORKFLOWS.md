# Alignment Implementation Workflows

This document maps the implemented workflows for `PPO`, `DPO`, `GRPO`, and `RLVR` in this repository. For each method, it lists the execution flow and the exact functions involved, grouped by file.

## Shared Setup

All methods in `train_rl.py` start from the SFT policy adapter and build around a frozen reference policy.

### Main entry points

- `train_rl.py`
  - `run_dpo()`
  - `run_ppo()`
  - `run_grpo()`
  - `sample_prompts()`
  - `append_log()`
  - `load_trainable_policy()`
  - `load_frozen_reward_model()`

### Model and tokenizer loading

- `model/loading.py`
  - `load_policy_tokenizer()`: loads the policy tokenizer, sets `pad_token = eos_token`, `padding_side = "left"`, `truncation_side = "left"`
  - `load_reward_tokenizer()`: loads the reward tokenizer, sets `pad_token = eos_token`, `padding_side = "right"`, `truncation_side = "left"`
  - `load_policy_model()`: loads the causal LM backbone for the policy
  - `load_reference_policy()`: creates the frozen reference policy, optionally from the SFT adapter
  - `load_reward_model()`: loads the reward model backbone
  - `load_value_backbone()`: loads the backbone used by the PPO value head
  - `apply_lora()`: wraps a model with LoRA adapters
  - `maybe_enable_gradient_checkpointing()`
  - `freeze_model()`
  - `model_memory_report()`

### Shared rollout and masking helpers

- `alignment/common.py`
  - `GenerationBatch`: generated sequence container
  - `gather_shifted_log_probs()`: extracts next-token log-probs for the realized labels
  - `compute_response_log_probs()`: token-level log-probs over response tokens only
  - `compute_sequence_log_probs()`: sums response-token log-probs into `log p(y | x)`
  - `compute_token_values()`: token-level value predictions aligned to response tokens
  - `build_generation_masks()`: reconstructs prompt/response masks after `generate()`
  - `generate_responses()`: batched prompt generation
  - `score_texts_with_reward_model()`: scores full prompt-response texts with the reward model
  - `sampled_kl()`: sampled-token KL estimate
  - `full_token_kl()`: full-vocabulary per-token KL
  - `masked_sequence_mean()`: sequence mean over valid response tokens
  - `batch_greedy_generate()`: deterministic generation helper used in evaluation
  - `compute_reward_win_rate()`: reward-model comparison metric
  - `compute_mc_kl()`: Monte Carlo KL estimate for evaluation
  - `shuffle_minibatches()`: minibatch index shuffler

### Shared utilities

- `utils.py`
  - `masked_mean()`
  - `masked_normalize()`
  - `grad_norm()`

## DPO Workflow

`DPO` is implemented as offline preference optimization over HH-RLHF chosen/rejected pairs.

### Workflow

1. Load policy tokenizer, trainable policy, frozen reference policy, and frozen reward model.
2. Load HH-RLHF train and eval examples.
3. Collate each batch into tokenized `chosen` and `rejected` sequences, each with:
   - `input_ids`
   - `attention_mask`
   - `response_mask`
4. Compute sequence log-probs for the chosen and rejected responses under the current policy.
5. Compute the same sequence log-probs under the frozen reference policy.
6. Form the DPO margin and optimize `-log(sigmoid(z))`.
7. Periodically evaluate generation quality and held-out preference accuracy.

### Functions by file

- `train_rl.py`
  - `run_dpo()`: orchestration for DPO training
  - `load_trainable_policy()`
  - `load_frozen_reward_model()`
  - `append_log()`

- `model/loading.py`
  - `load_policy_tokenizer()`
  - `load_reference_policy()`
  - `model_memory_report()`

- `data/hh_rlhf.py`
  - `load_hh_rlhf_examples()`: loads HH-RLHF examples
  - `DPOCollator`: builds chosen/rejected tokenized batches
  - `_tokenize_prompt_response_pairs()`: creates `input_ids`, `attention_mask`, and `response_mask`
  - `split_prompt_response()`
  - `_shared_prompt()`
  - `parse_hh_rlhf_record()`

- `alignment/dpo.py`
  - `sequence_log_probs()`: thin wrapper over `compute_sequence_log_probs()`
  - `dpo_loss()`: exact DPO objective

- `alignment/common.py`
  - `compute_sequence_log_probs()`
  - `compute_response_log_probs()`
  - `gather_shifted_log_probs()`

- `eval.py`
  - `evaluate_alignment_model()`
  - `evaluate_preference_accuracy()`

## PPO Workflow

`PPO` is implemented as online rollout-based RL with a separate value model and a frozen reward model.

### Workflow

1. Load policy tokenizer, trainable policy, frozen reference policy, frozen reward model, and reward tokenizer.
2. Build a separate value model:
   - load reward-style backbone
   - enable checkpointing if configured
   - apply LoRA
   - wrap with `ValueModel`
3. Sample prompts from the HH training prompt pool.
4. Generate one sampled response per prompt with the current policy.
5. On the generated trajectories, cache:
   - old policy token log-probs
   - reference policy token log-probs
   - value predictions
   - reward-model scalar scores on full prompt-response text
6. Compose token rewards:
   - per-token KL shaping term
   - terminal task reward on the final response token
7. Compute GAE advantages and returns.
8. Normalize advantages.
9. Reuse the rollout for multiple update epochs and minibatches.
10. Optimize:
   - PPO clipped surrogate loss for the policy
   - squared-error value loss for the value model
11. Periodically run alignment evaluation and save adapters.

### Functions by file

- `train_rl.py`
  - `run_ppo()`: orchestration for PPO training
  - `sample_prompts()`
  - `append_log()`
  - `load_trainable_policy()`
  - `load_frozen_reward_model()`

- `model/loading.py`
  - `load_policy_tokenizer()`
  - `load_reference_policy()`
  - `load_value_backbone()`
  - `apply_lora()`
  - `maybe_enable_gradient_checkpointing()`
  - `model_memory_report()`

- `model/heads.py`
  - `ValueModel`: scalar value head on top of the backbone

- `data/hh_rlhf.py`
  - `load_hh_rlhf_examples()`

- `alignment/ppo.py`
  - `RolloutBatch`: PPO rollout container
  - `collect_ppo_rollout()`: collects trajectories and cached statistics
  - `compose_rewards()`: KL shaping plus terminal reward
  - `compute_gae()`: token-level GAE
  - `ppo_loss()`: clipped policy loss and value loss

- `alignment/common.py`
  - `generate_responses()`
  - `build_generation_masks()`
  - `compute_response_log_probs()`
  - `compute_token_values()`
  - `score_texts_with_reward_model()`
  - `shuffle_minibatches()`

- `utils.py`
  - `masked_mean()`
  - `masked_normalize()`
  - `grad_norm()`

- `eval.py`
  - `evaluate_alignment_model()`

## GRPO Workflow

`GRPO` is implemented as grouped online policy optimization without a value model.

### Workflow

1. Load policy tokenizer, trainable policy, frozen reference policy, frozen reward model, and reward tokenizer.
2. Sample prompts from the HH training prompt pool.
3. Repeat each prompt `group_size` times.
4. Generate `group_size` sampled responses per prompt.
5. Compute:
   - old policy token log-probs
   - reference policy token log-probs
   - reward-model scores for each sampled response
6. Within each prompt group:
   - compute the group mean reward
   - center each reward by subtracting the mean
   - broadcast the centered reward across all valid response tokens
   - normalize those broadcast advantages
7. Reuse the rollout over multiple minibatch epochs.
8. Compute the policy update using a clipped surrogate ratio and a KL regularizer against the reference policy.
9. Use either:
   - sampled-token KL, or
   - full-vocabulary KL when `--full-kl` is enabled
10. Periodically run alignment evaluation and save the policy adapter.

### Functions by file

- `train_rl.py`
  - `run_grpo()` with `rlvr=False`: orchestration for GRPO training
  - `sample_prompts()`
  - `append_log()`
  - `load_trainable_policy()`
  - `load_frozen_reward_model()`

- `model/loading.py`
  - `load_policy_tokenizer()`
  - `load_reference_policy()`
  - `model_memory_report()`

- `data/hh_rlhf.py`
  - `load_hh_rlhf_examples()`

- `alignment/grpo.py`
  - `GroupRolloutBatch`: grouped rollout container
  - `collect_grpo_rollout()`: grouped generation and scoring
  - `_broadcast_group_advantages()`: computes centered group-relative advantages
  - `reward_model_reward_fn()`: wraps the reward model as a reward function
  - `grpo_loss()`: clipped surrogate plus KL regularization

- `alignment/common.py`
  - `generate_responses()`
  - `build_generation_masks()`
  - `compute_response_log_probs()`
  - `score_texts_with_reward_model()`
  - `full_token_kl()`
  - `masked_sequence_mean()`
  - `shuffle_minibatches()`

- `utils.py`
  - `masked_normalize()`
  - `grad_norm()`

- `eval.py`
  - `evaluate_alignment_model()`

## RLVR Workflow

In this repository, `RLVR` reuses the GRPO training path and swaps the learned reward model for a verifiable reward on GSM8K.

### Workflow

1. Load policy tokenizer, trainable policy, and frozen reference policy.
2. Load GSM8K train and eval examples.
3. Convert GSM8K examples into verifiable `(prompt, gold_answer)` pairs.
4. For each step:
   - sample prompt examples
   - repeat each prompt `group_size` times
   - repeat each gold answer `group_size` times
   - build an RLVR reward function bound to those gold answers
5. Use the same grouped rollout collection as GRPO:
   - generate grouped responses
   - compute old policy and reference log-probs
   - score each response with the verifiable reward
6. Compute group-relative advantages exactly as in GRPO.
7. Optimize with the same `grpo_loss()` function.
8. Periodically evaluate GSM8K pass@1 and format compliance.

### Functions by file

- `train_rl.py`
  - `run_grpo()` with `rlvr=True`: orchestration for RLVR training
  - `sample_prompts()`
  - `append_log()`
  - `load_trainable_policy()`

- `model/loading.py`
  - `load_policy_tokenizer()`
  - `load_reference_policy()`
  - `model_memory_report()`

- `data/gsm8k.py`
  - `load_gsm8k_examples()`
  - `build_gsm8k_prompt()`
  - `extract_numeric_answer()`
  - `parse_gold_answer()`

- `alignment/rlvr.py`
  - `VerifiableExample`: prompt + gold answer container
  - `build_verifiable_examples()`: converts GSM8K records into prompts with gold answers
  - `verifiable_reward()`: binary correctness reward
  - `rlvr_reward_fn()`: binds gold answers into a reward callback
  - `format_compliance()`: checks whether a numeric answer is extractable

- `alignment/grpo.py`
  - `collect_grpo_rollout()`
  - `_broadcast_group_advantages()`
  - `grpo_loss()`

- `alignment/common.py`
  - `generate_responses()`
  - `build_generation_masks()`
  - `compute_response_log_probs()`
  - `full_token_kl()`
  - `masked_sequence_mean()`
  - `shuffle_minibatches()`

- `eval.py`
  - `evaluate_gsm8k_pass_at_1()`

## File-to-Role Summary

- `train_rl.py`: top-level training loops for DPO, PPO, GRPO, RLVR
- `alignment/common.py`: shared sequence accounting, generation, masking, reward scoring, KL helpers
- `alignment/dpo.py`: DPO objective
- `alignment/ppo.py`: PPO rollout collection, reward shaping, GAE, PPO loss
- `alignment/grpo.py`: GRPO rollout collection, group-relative advantages, GRPO loss
- `alignment/rlvr.py`: verifiable reward logic for GSM8K
- `data/hh_rlhf.py`: HH-RLHF loading and collators
- `data/gsm8k.py`: GSM8K loading, prompt formatting, answer extraction
- `model/loading.py`: model/tokenizer loading, reference policy creation, LoRA setup
- `model/heads.py`: value head used by PPO
- `eval.py`: evaluation for alignment models and GSM8K
- `utils.py`: masking, normalization, gradient norm utilities

## Short Conceptual Comparison

- `DPO`: offline preference optimization on chosen/rejected pairs, no rollout generation in the training loop
- `PPO`: online RL with sampled rollouts, reward model scoring, KL shaping, and a learned value function
- `GRPO`: online grouped optimization using relative rewards within prompt groups, no value model
- `RLVR`: GRPO with a verifiable binary reward instead of a learned reward model
