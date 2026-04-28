# DVLM PA3 Coding Section Explanation

This workspace now contains a complete implementation for both coding sections of `DVLM_PA3.pdf`.

## Files

- `part_a_continuous_connector.py`: Part A continuous connector VLM.
- `part_b_discrete_vqvae.py`: Part B discrete VQ-VAE and unified-token LM pipeline.
- `requirements.txt`: Allowed dependencies from the assignment.

## Part A: Continuous Connector VLMs

The Part A script builds the CIFAR-10 pipeline required by Tasks A-C0 through A-C6.

### Data

`build_cifar()` downloads CIFAR-10 and creates stratified subsets with seed `42`:

- train: `1000` images per class by default,
- test: `200` images per class by default.

`CifarCaptionDataset` creates one synthetic caption per image using six rotating caption templates. `CifarVQADataset` applies all five VQA templates to every image:

- object recognition,
- yes/no presence,
- vehicle vs. living,
- flight capability,
- animal category.

`AlpacaTextDataset` loads `1000` `tatsu-lab/alpaca` examples for language replay and baseline perplexity.

### Models

`load_models()` loads:

- frozen `openai/clip-vit-base-patch32`,
- `HuggingFaceTB/SmolLM2-360M-Instruct`,
- the CLIP image processor.

The code explicitly verifies CLIP output has `50` tokens and discards CLS with `last_hidden_state[:, 1:, :]`, leaving `49 x 768` patch tokens.

`MLPConnector` implements:

```text
Linear(768, 960) -> GELU -> Linear(960, 960)
```

The connector has about `1.66M` trainable parameters.

### Training Phases

Phase 1 trains only the connector on caption pairs:

```text
[BOS embedding, 49 visual embeddings, caption embeddings]
```

Labels are `-100` for BOS and visual positions, and caption token IDs for caption positions. After training, `norm_ratio()` checks the visual/text embedding norm ratio and rescales if needed.

Phase 2 applies LoRA to `q_proj`, `k_proj`, `v_proj`, and `o_proj`, then trains connector plus LoRA with:

```text
Lmixed = LVQA + lambda * LLM
```

The VQA sequence is:

```text
[BOS embedding, visual embeddings, question embeddings, answer embeddings, EOS]
```

Only answer and EOS tokens receive labels. Alpaca batches stay text-only, so visual embeddings cannot leak into replay.

Phase 3 continues VQA alignment without replay:

```text
L = LVQA
```

### Evaluation

`evaluate_vqa()` reports exact-match VQA accuracy overall, per template, and per CIFAR class. `compute_ppl()` computes Alpaca perplexity and the forgetting ratio:

```text
R = PPLfine / PPL0
```

`modality_gap()` computes:

```text
MG = || mean(normalized visual) - mean(normalized text) ||_2
```

along with within-visual, within-text, and cross-modal cosine averages.

### Typical Run

```bash
python part_a_continuous_connector.py
```

For a small pipeline check:

```bash
python part_a_continuous_connector.py --smoke
```

For the replay-weight ablation:

```bash
python part_a_continuous_connector.py --run-ablation
```

Outputs are written under `weights/`, including connector checkpoints and `part_a_results.json`.

## Part B: Discrete VQ-VAE

The Part B script implements Tasks B-C0 through B-C6.

### Synthetic Dataset

`generate_dataset()` creates six 16 x 16 RGB classes:

- spiral,
- triangle,
- circle,
- cross,
- checkerboard,
- gradient.

It uses an 80/20 stratified split. With the default `1000` images per class, this gives:

- `4800` train images,
- `1200` validation images.

`EncodedMultimodalDataset` builds:

- 4 VQA samples per image,
- 1 image-generation sample per image.

### VQ-VAE

`VQVAE` contains:

- `VQVAEEncoder`: `Conv(3->32, stride 2)`, `Conv(32->64, stride 2)`, `Conv(64->64)`, each with GroupNorm and ReLU.
- `VectorQuantizer`: `K=256`, `d=64`, straight-through estimator, gradient or EMA updates, and dead-code restart.
- `VQVAEDecoder`: two transposed convolutions back to 16 x 16 and a final sigmoid image head.

The VQ-VAE loss is:

```text
MSE reconstruction + codebook loss + beta * commitment loss
```

`codebook_analysis()` saves code usage, perplexity, dead-code count, token maps, and the cosine-similarity matrix.

### Unified Token Stream

The visual-token offset follows the assignment:

```text
visual token id = codebook index + Vtxt + 2
```

The VQA stream is:

```text
[BOS, <image>, v1 ... v16, </image>, question, answer, EOS]
```

The image-generation stream is:

```text
[BOS, prompt, <image>, v1 ... v16, </image>, EOS]
```

VQA labels are active only for answer and EOS positions. Image-generation labels are active for visual tokens and the closing image token.

### Virtual Vocabulary Overlay

`VirtualVocabCausalLM` implements the assignment’s overlay idea without making the original text embedding table trainable:

- base text embeddings stay frozen,
- `OverlayEmbedding` supplies trainable rows for `<image>`, `</image>`, and the `K` visual tokens,
- `ExpandedLMHead` appends trainable logits for the same new rows.

`projector_warmup()` trains a temporary `P: R64 -> R960` on VQ-VAE codebook vectors, transplants `P(codebook)` into the visual-token rows, then discards the projector.

### Mixed Training

LoRA is applied after the visual rows are initialised. The mixed objective is:

```text
Lmixed = LVQA + lambda * LLM + gamma_img * LIMG
```

`train_mixed()` performs sequential VQA, image-generation, and Alpaca forward/backward passes before one optimizer step, which matches the assignment’s low-memory recommendation.

### Logit Masking

`mask_logits()` handles modality routing:

- VQA/text generation masks IDs `>= Vtxt`,
- image generation masks text and special tokens, leaving visual token IDs only.

Masked logits use the minimum finite value for the current dtype, so softmax assigns effectively zero probability.

### Typical Run

```bash
python part_b_discrete_vqvae.py --ema
```

For a small pipeline check:

```bash
python part_b_discrete_vqvae.py --smoke --ema
```

For the mixed-objective ablation:

```bash
python part_b_discrete_vqvae.py --run-ablation --ema
```

Outputs are written under `weights/`, including `vqvae_best.pt`, `vqvae_codebook_analysis.pt`, `part_b_lora_virtual_vocab.pt`, and `part_b_results.json`.

## Notes

The scripts use the exact model names and dataset names from the PDF. They require internet access the first time HuggingFace and torchvision download the models/datasets. On Kaggle or Colab, install `requirements.txt`, run the smoke tests first, then launch the full defaults.
