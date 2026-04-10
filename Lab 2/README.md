# AI623 PA2 Coding Solution

This repository implements the coding section of `DVLM_PA2.pdf` from scratch, without using TRL/trlX/OpenRLHF/RL4LMs. The code covers:

- HH-RLHF parsing and task-specific dataloaders
- LoRA-based SFT warm-up
- Reward model fine-tuning with explicit margin-ranking loss
- PPO with cached old-policy log-probs, KL shaping, GAE, and a separate value model
- DPO with explicit sequence log-prob computation
- GRPO with group-relative advantages
- RLVR on GSM8K with a verifiable numeric reward
- Evaluation helpers for RM win-rate, KL, preference accuracy, sample tables, and GSM8K pass@1
b    

## Structure

```text
.
├── alignment/
│   ├── common.py      # shared rollout, generation, KL, log-prob, and reward helpers
│   ├── dpo.py         # DPO sequence log-probs and loss
│   ├── grpo.py        # GRPO rollout + loss
│   ├── ppo.py         # PPO rollout, GAE, reward shaping, clipped loss
│   └── rlvr.py        # GSM8K verifiable reward utilities
├── data/
│   ├── gsm8k.py       # GSM8K loading, prompt formatting, answer extraction
│   └── hh_rlhf.py     # HH-RLHF parsing, collators for SFT/RM/DPO
├── model/
│   ├── heads.py       # scalar value head
│   └── loading.py     # tokenizers, model loading, LoRA, frozen reference utilities
├── tests/
│   └── test_sanity.py # extractor + math sanity checks
├── config.py          # baseline/default configs from the PDF
├── eval.py            # evaluation helpers for Tasks C1/C2/C8
├── train_rm.py        # Task C1
├── train_rl.py        # Tasks C3/C4/C5/C6/C7/C8
├── train_sft.py       # Task C2
└── requirements.txt
```

## Task Mapping

### C0: Data Pipeline and Model Loading

- `data/hh_rlhf.py`
  - `split_prompt_response()` extracts `(prompt, response)` from the full `chosen` / `rejected` conversations.
  - `load_hh_rlhf_examples()` loads `Anthropic/hh-rlhf` and returns parsed triples.
  - `SFTCollator`, `RewardModelCollator`, and `DPOCollator` build the three dataloader formats required by the assignment.
- `model/loading.py`
  - `load_policy_tokenizer()` enforces `pad_token = eos_token`, `padding_side = left`.
  - `load_reward_tokenizer()` uses right padding for the reward model.
  - `load_policy_model()`, `load_reward_model()`, `load_value_backbone()` load the separate policy, reward, and critic backbones.
  - `apply_lora()` applies LoRA with `r = 8`, `alpha = 16`, `dropout = 0.05`.
  - `load_reference_policy()` creates a frozen reference copy from the SFT adapter checkpoint.

### C1: Reward Model Training

- `train_rm.py`
  - Loads `AutoModelForSequenceClassification`
  - Trains with explicit `-log(sigmoid(r+ - r-)) + lambda_reg * (r+^2 + r-^2)`
  - Reports training preference accuracy
  - Evaluates held-out preference accuracy and saves reward histograms

### C2: SFT Warm-Up

- `train_sft.py`
  - Trains a causal LM with prompt tokens masked from the loss
  - Logs training loss and held-out perplexity
  - Generates a few greedy samples after training
  - Saves the LoRA adapter checkpoint that becomes both the trainable starting policy and the frozen KL reference

### C3: PPO

- `alignment/ppo.py`
  - `collect_ppo_rollout()` generates sampled rollouts, caches old-policy log-probs, reference log-probs, value estimates, and reward-model scores
  - `compose_rewards()` adds per-token KL shaping plus the terminal task reward
  - `compute_gae()` implements GAE
  - `ppo_loss()` implements the clipped surrogate and value loss
- `train_rl.py --method ppo`
  - Uses a separate value model (`model/heads.py`)
  - Performs multiple minibatch epochs over each rollout batch

### C4: DPO

- `alignment/dpo.py`
  - `sequence_log_probs()` computes `log pi(y | x)` by summing token log-probs over response tokens only
  - `dpo_loss()` implements the exact DPO objective from the PDF
- `train_rl.py --method dpo`
  - Runs offline preference optimization over HH-RLHF pairs

### C5: GRPO

- `alignment/grpo.py`
  - Samples `K` completions per prompt
  - Computes group means and centered group-relative advantages
  - Broadcasts the group advantage to all response tokens
  - Supports sampled-token KL by default and full-vocabulary KL via `--full-kl`
- `train_rl.py --method grpo`

### C6: RLVR on GSM8K

- `data/gsm8k.py`
  - `extract_numeric_answer()` parses answers from `#### ...`, `The answer is ...`, boxed forms, or the final number fallback
  - `build_gsm8k_prompt()` formats the prompt exactly for step-by-step reasoning with a final numeric answer
- `alignment/rlvr.py`
  - `verifiable_reward()` implements binary correctness
  - `build_verifiable_examples()` prepares `(prompt, gold_answer)` pairs
- `train_rl.py --method rlvr`
  - Reuses the GRPO machinery but swaps the learned reward for the verifiable GSM8K reward

### C7: Focused Ablations

The training scripts expose the main ablation knobs:

- PPO/GRPO KL coefficient: `--beta`
- PPO/GRPO clipping: `--clip-epsilon`
- GRPO group size: `--group-size`
- DPO inverse temperature: `--beta`
- Full vs sampled KL in GRPO: `--full-kl`

Run separate jobs while changing one argument at a time and compare the saved `metrics.jsonl` files.

### C8: Evaluation

- `eval.py`
  - `evaluate_alignment_model()` computes RM win-rate vs SFT, Monte Carlo KL, and mean RM score
  - `evaluate_preference_accuracy()` evaluates DPO-style held-out pair accuracy
  - `build_sample_response_table()` creates side-by-side prompt/response/RM-score tables
  - `evaluate_gsm8k_pass_at_1()` reports pass@1, format compliance, and mean response length for RLVR

## Key Implementation Choices

### HH-RLHF parsing

The Anthropic HH-RLHF `chosen` and `rejected` fields contain the entire conversation. The code splits on the final `"\n\nAssistant:"` turn and uses that shared prefix as the prompt. This is what the assignment expects for recovering `(x, y+, y-)`.

### Padding and masking

- Policy tokenizer: left padding
- Reward tokenizer: right padding
- SFT loss: only response tokens are supervised
- DPO loss: only response tokens contribute to sequence log-prob sums

These are the exact failure points called out in the PDF, so the code treats them explicitly.

### Separate models

The code keeps the following modules separate:

- Policy: SmolLM-family causal LM + LoRA
- Reference policy: frozen copy of the SFT checkpoint
- Reward model: Llama sequence classifier + scalar reward head
- Value model: separate Llama backbone + scalar value head

That matches the assignment’s intended alignment pipeline.

### No trainer abstractions

All loops are written explicitly with PyTorch:

- forward pass
- loss construction
- backward pass
- clipping
- optimizer step
- evaluation

No `Trainer`, TRL PPO trainer, or DPO trainer is used.

## How To Run

Install the dependencies first:

```bash
pip install -r requirements.txt
```

If you use `meta-llama/Llama-3.2-1B-Instruct`, you also need to accept the model license and log in with Hugging Face:

```bash
huggingface-cli login
```

### 1. Train the SFT warm-up model

```bash
python train_sft.py \
  --policy-name HuggingFaceTB/SmolLM2-360M \
  --max-train-examples 2000 \
  --max-eval-examples 256
```

This saves a LoRA adapter in `outputs/.../policy_adapter`.

### 2. Train the reward model

```bash
python train_rm.py \
  --reward-backbone-name meta-llama/Llama-3.2-1B-Instruct \
  --max-train-examples 4000 \
  --max-eval-examples 512
```

This saves a reward-model adapter in `outputs/.../reward_model`.

### 3. Train DPO

```bash
python train_rl.py \
  --method dpo \
  --sft-adapter-dir outputs/<sft-run>/policy_adapter \
  --reward-model-dir outputs/<rm-run>/reward_model \
  --steps 200 \
  --mini-batch-size 8 \
  --beta 0.1
```

### 4. Train PPO

```bash
python train_rl.py \
  --method ppo \
  --sft-adapter-dir outputs/<sft-run>/policy_adapter \
  --reward-model-dir outputs/<rm-run>/reward_model \
  --steps 200 \
  --prompts-per-step 8 \
  --mini-batch-size 8 \
  --update-epochs 4 \
  --beta 0.1 \
  --clip-epsilon 0.2
```

### 5. Train GRPO

```bash
python train_rl.py \
  --method grpo \
  --sft-adapter-dir outputs/<sft-run>/policy_adapter \
  --reward-model-dir outputs/<rm-run>/reward_model \
  --steps 200 \
  --prompts-per-step 8 \
  --group-size 4 \
  --beta 0.1 \
  --clip-epsilon 0.2
```

### 6. Train RLVR on GSM8K

```bash
python train_rl.py \
  --method rlvr \
  --sft-adapter-dir outputs/<sft-run>/policy_adapter \
  --steps 300 \
  --prompts-per-step 8 \
  --group-size 4 \
  --beta 0.05 \
  --clip-epsilon 0.2 \
  --max-new-tokens 256
```

## Outputs

Each run creates an `outputs/<run-name>/` directory containing at least:

- `args.json`
- `metrics.json` or `metrics.jsonl`
- saved adapter checkpoint(s)
- sample outputs where relevant

That is enough to build the report tables requested in Tasks C7 and C8.

## Sanity Checks Included

`tests/test_sanity.py` includes:

- GSM8K numeric answer extraction checks
- GAE hand-computation check
- PPO clipping zero-gradient check for the saturated positive-advantage case

Run them with:

```bash
python -m unittest discover -s tests -v
```

## What I Verified Locally

In this environment I was able to verify:

- the repository compiles with `python -m compileall .`
- the lightweight sanity suite runs with `python -m unittest discover -s tests -v`

The heavy training paths were not executed here because the local environment does not currently include the full ML stack (`torch`, `datasets`, model weights, and GPU runtime). The code is written to use those libraries once they are installed.
