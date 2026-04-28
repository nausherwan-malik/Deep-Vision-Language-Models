#!/usr/bin/env python3
"""Part B solution: discrete VQ-VAE plus unified-token LM training.

The script implements synthetic image generation, VQ-VAE training with gradient
or EMA codebook updates, code-token construction, virtual vocabulary expansion,
mixed VQA/image/text LoRA fine-tuning, logit masking, and evaluation utilities.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SHAPES = ("spiral", "triangle", "circle", "cross", "checkerboard", "gradient")
GEOMETRIC = {"triangle", "circle", "cross", "checkerboard"}
SYMMETRY_AXES = {
    "spiral": "0",
    "triangle": "3",
    "circle": "infinite",
    "cross": "4",
    "checkerboard": "4",
    "gradient": "1",
}
VQA_TEMPLATES = (
    ("what shape is in this image?", "class"),
    ("is there a {label}?", "presence"),
    ("geometric or non-geometric?", "geo"),
    ("how many axes of symmetry?", "symmetry"),
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def amp_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.bfloat16


def autocast_context(device: torch.device):
    enabled = device.type in {"cuda", "mps"}
    return torch.autocast(device_type=device.type, dtype=amp_dtype(device), enabled=enabled)


def count_trainable_parameters(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


def snapshot_trainable(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: param.detach().cpu().clone() for name, param in module.named_parameters() if param.requires_grad}


def restore_trainable(module: nn.Module, snapshot: Dict[str, torch.Tensor]) -> None:
    params = dict(module.named_parameters())
    for name, value in snapshot.items():
        params[name].data.copy_(value.to(params[name].device, dtype=params[name].dtype))


def make_shape_image(label: str, rng: np.random.Generator, size: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    x = (xx - (size - 1) / 2) / ((size - 1) / 2)
    y = (yy - (size - 1) / 2) / ((size - 1) / 2)
    img = np.zeros((size, size, 3), dtype=np.float32)
    color = rng.uniform(0.45, 1.0, size=3).astype(np.float32)
    bg = rng.uniform(0.0, 0.12, size=3).astype(np.float32)
    img[:] = bg
    if label == "circle":
        mask = x**2 + y**2 <= rng.uniform(0.32, 0.48) ** 2
    elif label == "triangle":
        mask = (y > -0.45) & (y < 0.60) & (np.abs(x) < (0.60 - y) * 0.65)
    elif label == "cross":
        width = rng.uniform(0.17, 0.25)
        mask = (np.abs(x) < width) | (np.abs(y) < width)
    elif label == "checkerboard":
        block = rng.integers(2, 4)
        mask = ((xx // block + yy // block) % 2) == 0
    elif label == "gradient":
        grad = (xx + yy) / (2 * (size - 1))
        img = np.stack([grad, 1.0 - grad, 0.25 + 0.5 * grad], axis=-1).astype(np.float32)
        return np.clip(img + rng.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)
    elif label == "spiral":
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        curve = np.mod(theta + 9.0 * r, 2 * np.pi)
        mask = (curve < 0.45) & (r < 0.9) & (r > 0.08)
    else:
        raise ValueError(label)
    img[mask] = color
    img = np.clip(img + rng.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)
    return img


@dataclass(frozen=True)
class ShapeSample:
    image: torch.Tensor
    label_id: int
    label: str


def generate_dataset(n_per_class: int = 1000, seed: int = 42) -> Tuple[List[ShapeSample], List[ShapeSample]]:
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label_id, label in enumerate(SHAPES):
        samples = []
        for _ in range(n_per_class):
            arr = make_shape_image(label, rng)
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            samples.append(ShapeSample(tensor, label_id, label))
        rng.shuffle(samples)
        split = int(0.8 * n_per_class)
        train.extend(samples[:split])
        val.extend(samples[split:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class ShapeImageDataset(Dataset):
    def __init__(self, samples: Sequence[ShapeSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.samples[idx]
        return {"image": item.image, "label": item.label, "label_id": item.label_id}


class VQVAEEncoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VQVAEDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int = 256, dim: int = 64, beta: float = 0.25, ema: bool = False, gamma: float = 0.99, dead_threshold: float = 2.0):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.beta = beta
        self.ema = ema
        self.gamma = gamma
        self.dead_threshold = dead_threshold
        self.codebook = nn.Parameter(torch.empty(num_codes, dim))
        nn.init.uniform_(self.codebook, -1.0 / num_codes, 1.0 / num_codes)
        if ema:
            self.codebook.requires_grad_(False)
        self.register_buffer("ema_count", torch.zeros(num_codes))
        self.register_buffer("ema_sum", torch.zeros(num_codes, dim))

    def forward(self, ze: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        b, c, h, w = ze.shape
        flat = ze.permute(0, 2, 3, 1).reshape(-1, c)
        distances = flat.pow(2).sum(dim=1, keepdim=True) - 2 * flat @ self.codebook.t() + self.codebook.pow(2).sum(dim=1)
        indices = torch.argmin(distances, dim=1)
        zq_flat = F.embedding(indices, self.codebook)
        zq = zq_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(zq, ze.detach())
        commitment_loss = F.mse_loss(ze, zq.detach())
        zq_st = ze + (zq - ze).detach()

        if self.training and self.ema:
            self._ema_update(flat.detach(), indices.detach())

        one_hot = F.one_hot(indices, self.num_codes).float()
        usage = one_hot.mean(dim=0)
        perplexity = torch.exp(-(usage * (usage + 1e-10).log()).sum())
        dead_codes = (usage * indices.numel() < self.dead_threshold).sum()
        aux = {
            "indices": indices.view(b, h, w),
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
            "dead_codes": dead_codes,
            "usage": usage,
        }
        return zq_st, indices.view(b, h, w), aux

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, indices: torch.Tensor) -> None:
        one_hot = F.one_hot(indices, self.num_codes).type_as(flat)
        counts = one_hot.sum(dim=0)
        sums = one_hot.t() @ flat
        self.ema_count.mul_(self.gamma).add_(counts, alpha=1.0 - self.gamma)
        self.ema_sum.mul_(self.gamma).add_(sums, alpha=1.0 - self.gamma)
        n = self.ema_count.sum()
        smoothed = (self.ema_count + 1e-5) / (n + self.num_codes * 1e-5) * n
        self.codebook.data.copy_(self.ema_sum / smoothed.unsqueeze(1).clamp_min(1e-5))
        dead = torch.where(counts < self.dead_threshold)[0]
        if dead.numel() > 0 and flat.numel() > 0:
            repl = flat[torch.randint(0, flat.shape[0], (dead.numel(),), device=flat.device)]
            self.codebook.data[dead] = repl
            self.ema_sum[dead] = repl
            self.ema_count[dead] = self.dead_threshold + 1.0


class VQVAE(nn.Module):
    def __init__(self, num_codes: int = 256, latent_dim: int = 64, beta: float = 0.25, ema: bool = False):
        super().__init__()
        self.encoder = VQVAEEncoder(latent_dim)
        self.quantizer = VectorQuantizer(num_codes, latent_dim, beta=beta, ema=ema)
        self.decoder = VQVAEDecoder(latent_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        ze = self.encoder(x)
        zq, indices, aux = self.quantizer(ze)
        recon = self.decoder(zq)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + aux["codebook_loss"] + self.quantizer.beta * aux["commitment_loss"]
        aux.update({"ze": ze, "zq": zq, "recon": recon, "recon_loss": recon_loss, "loss": loss, "indices": indices})
        return aux

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        ze = self.encoder(x)
        _zq, indices, _aux = self.quantizer(ze)
        return indices

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        zq = F.embedding(indices, self.quantizer.codebook).permute(0, 3, 1, 2).contiguous()
        return self.decoder(zq)


class AlpacaTextDataset(Dataset):
    def __init__(self, tokenizer, max_examples: int = 1000, max_length: int = 384):
        data = load_dataset("tatsu-lab/alpaca", split=f"train[:{max_examples}]")
        self.samples = []
        for row in data:
            instruction = row["instruction"].strip()
            inp = row.get("input", "").strip()
            output = row["output"].strip()
            prompt = f"### Instruction:\n{instruction}\n"
            if inp:
                prompt += f"\n### Input:\n{inp}\n"
            text = f"{prompt}\n### Response:\n{output}{tokenizer.eos_token}"
            enc = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)
            self.samples.append(torch.tensor(enc["input_ids"], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


class EncodedMultimodalDataset(Dataset):
    def __init__(self, samples: Sequence[ShapeSample], code_maps: torch.Tensor, tokenizer, mode: str, vtxt: int):
        self.rows = []
        self.tokenizer = tokenizer
        self.mode = mode
        self.vtxt = vtxt
        self.image_token_id = vtxt
        self.end_image_token_id = vtxt + 1
        for sample, code_map in zip(samples, code_maps):
            visual_ids = (code_map.reshape(-1).long() + vtxt + 2).tolist()
            if mode == "vqa":
                for question_template, name in VQA_TEMPLATES:
                    question = question_template.format(label=sample.label)
                    answer = self._answer(sample.label, name)
                    ids, labels = self._encode_vqa(question, answer, visual_ids)
                    self.rows.append({"input_ids": ids, "labels": labels, "label": sample.label, "template": name, "answer": answer, "question": question})
            elif mode == "imagegen":
                prompt = self._image_prompt(sample.label, len(self.rows))
                ids, labels = self._encode_imagegen(prompt, visual_ids)
                self.rows.append({"input_ids": ids, "labels": labels, "label": sample.label, "template": "imagegen", "answer": sample.label, "question": prompt})
            else:
                raise ValueError(mode)

    def _answer(self, label: str, name: str) -> str:
        if name == "class":
            return label
        if name == "presence":
            return "yes"
        if name == "geo":
            return "geometric" if label in GEOMETRIC else "non-geometric"
        if name == "symmetry":
            return SYMMETRY_AXES[label]
        raise ValueError(name)

    def _image_prompt(self, label: str, idx: int) -> str:
        templates = ("generate a {label} image.", "draw the synthetic {label}.", "make a tiny {label} picture.")
        return templates[idx % len(templates)].format(label=label)

    def _text_ids(self, text: str) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _encode_vqa(self, question: str, answer: str, visual_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        bos = [self.tokenizer.bos_token_id or self.tokenizer.eos_token_id]
        q_ids = self._text_ids(question + "\n")
        a_ids = self._text_ids(answer + self.tokenizer.eos_token)
        ids = bos + [self.image_token_id] + visual_ids + [self.end_image_token_id] + q_ids + a_ids
        labels = [-100] * (len(ids) - len(a_ids)) + a_ids
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def _encode_imagegen(self, prompt: str, visual_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        bos = [self.tokenizer.bos_token_id or self.tokenizer.eos_token_id]
        p_ids = self._text_ids(prompt + "\n")
        tail = visual_ids + [self.end_image_token_id, self.tokenizer.eos_token_id]
        ids = bos + p_ids + [self.image_token_id] + tail
        labels = [-100] * (len(ids) - len(tail)) + tail
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.rows[idx]


class OverlayEmbedding(nn.Module):
    def __init__(self, base: nn.Embedding, num_new: int, init_mean_rows: int = 2):
        super().__init__()
        self.base = base
        self.vtxt = base.num_embeddings
        self.num_new = num_new
        self.dim = base.embedding_dim
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.new = nn.Embedding(num_new, self.dim)
        with torch.no_grad():
            mean = self.base.weight.mean(dim=0)
            self.new.weight[:init_mean_rows].copy_(mean.unsqueeze(0).expand(init_mean_rows, -1))
            if num_new > init_mean_rows:
                self.new.weight[init_mean_rows:].zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        base_ids = input_ids.clamp_max(self.vtxt - 1)
        out = self.base(base_ids)
        mask = input_ids >= self.vtxt
        if mask.any():
            out = out.clone()
            out[mask] = self.new(input_ids[mask] - self.vtxt)
        return out


class ExpandedLMHead(nn.Module):
    def __init__(self, base_head: nn.Module, hidden_dim: int, num_new: int, init_from: OverlayEmbedding):
        super().__init__()
        self.base_head = base_head
        for p in self.base_head.parameters():
            p.requires_grad_(False)
        self.new_weight = nn.Parameter(init_from.new.weight.detach().clone())
        self.new_bias = nn.Parameter(torch.zeros(num_new))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        base_logits = self.base_head(hidden)
        new_logits = F.linear(hidden, self.new_weight, self.new_bias)
        return torch.cat([base_logits, new_logits], dim=-1)


class VirtualVocabCausalLM(nn.Module):
    """Wraps a frozen base LM with trainable virtual input/output rows."""

    def __init__(self, model: nn.Module, num_new: int):
        super().__init__()
        self.model = model
        base_emb = model.get_input_embeddings()
        self.overlay = OverlayEmbedding(base_emb, num_new)
        hidden_dim = model.config.hidden_size
        output = model.get_output_embeddings()
        self.expanded_head = ExpandedLMHead(output, hidden_dim, num_new, self.overlay)
        self.vtxt = base_emb.num_embeddings
        self.vocab_size = self.vtxt + num_new

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, use_cache=False, past_key_values=None):
        if inputs_embeds is None:
            inputs_embeds = self.overlay(input_ids)
        out = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]
        logits = self.expanded_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, logits.shape[-1]), shift_labels.view(-1), ignore_index=-100)
        return type("VirtualOutput", (), {"loss": loss, "logits": logits, "past_key_values": getattr(out, "past_key_values", None), "hidden_states": out.hidden_states})


def collate_token_rows(batch, pad_id: int, device: torch.device):
    max_len = max(row["input_ids"].numel() for row in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, row in enumerate(batch):
        ids = row["input_ids"]
        lab = row["labels"]
        input_ids[i, -ids.numel() :] = ids
        labels[i, -lab.numel() :] = lab
        attention_mask[i, -ids.numel() :] = 1
    return input_ids.to(device), attention_mask.to(device), labels.to(device), batch


def collate_text(batch, tokenizer, device: torch.device):
    max_len = max(x.numel() for x in batch)
    ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, seq in enumerate(batch):
        ids[i, -seq.numel() :] = seq
        mask[i, -seq.numel() :] = 1
    labels = ids.clone()
    labels[mask == 0] = -100
    return ids.to(device), mask.to(device), labels.to(device)


def infinite(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def train_vqvae(args, train_samples: Sequence[ShapeSample], val_samples: Sequence[ShapeSample], device: torch.device) -> VQVAE:
    model = VQVAE(num_codes=args.codebook_size, latent_dim=args.latent_dim, beta=args.beta, ema=args.ema).to(device)
    trainable, total = count_trainable_parameters(model)
    print(f"VQ-VAE params: {total:,}; trainable {trainable:,}")
    params = [p for n, p in model.named_parameters() if (not args.ema or "quantizer.codebook" not in n)]
    opt = torch.optim.AdamW(params, lr=args.vqvae_lr)
    train_loader = DataLoader(ShapeImageDataset(train_samples), batch_size=args.vqvae_batch_size, shuffle=True, collate_fn=lambda b: torch.stack([x["image"] for x in b]).to(device))
    val_loader = DataLoader(ShapeImageDataset(val_samples), batch_size=args.vqvae_batch_size, shuffle=False, collate_fn=lambda b: torch.stack([x["image"] for x in b]).to(device))
    best = float("inf")
    Path(args.weights_dir).mkdir(exist_ok=True)
    for epoch in range(args.vqvae_epochs):
        model.train()
        logs = defaultdict(list)
        for x in tqdm(train_loader, desc=f"vqvae-{epoch+1}", leave=False):
            opt.zero_grad(set_to_none=True)
            out = model(x)
            out["loss"].backward()
            opt.step()
            for key in ("loss", "recon_loss", "codebook_loss", "commitment_loss", "perplexity", "dead_codes"):
                logs[key].append(float(out[key].detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x in val_loader:
                val_losses.append(float(model(x)["recon_loss"].detach().cpu()))
        val = float(np.mean(val_losses))
        print(
            f"epoch {epoch+1}: loss={np.mean(logs['loss']):.4f} recon={np.mean(logs['recon_loss']):.4f} "
            f"perp={np.mean(logs['perplexity']):.1f} dead={np.mean(logs['dead_codes']):.1f} val_recon={val:.4f}"
        )
        if val < best:
            best = val
            torch.save(model.state_dict(), Path(args.weights_dir) / "vqvae_best.pt")
    return model


@torch.no_grad()
def encode_all(vqvae: VQVAE, samples: Sequence[ShapeSample], device: torch.device, batch_size: int = 256) -> torch.Tensor:
    vqvae.eval()
    codes = []
    for i in range(0, len(samples), batch_size):
        x = torch.stack([s.image for s in samples[i : i + batch_size]]).to(device)
        codes.append(vqvae.encode_indices(x).cpu())
    return torch.cat(codes, dim=0)


def codebook_analysis(vqvae: VQVAE, val_samples: Sequence[ShapeSample], device: torch.device, out_dir: Path) -> Dict[str, float]:
    codes = encode_all(vqvae, val_samples, device)
    counts = torch.bincount(codes.reshape(-1), minlength=vqvae.quantizer.num_codes).float()
    probs = counts / counts.sum().clamp_min(1)
    perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum()).item()
    dead = int((counts == 0).sum().item())
    cos = F.normalize(vqvae.quantizer.codebook.detach().cpu(), dim=-1) @ F.normalize(vqvae.quantizer.codebook.detach().cpu(), dim=-1).T
    torch.save({"counts": counts, "cosine": cos, "codes": codes[:6]}, out_dir / "vqvae_codebook_analysis.pt")
    return {"perplexity": perplexity, "dead_codes": dead}


def add_virtual_tokens(tokenizer, k: int) -> Tuple[int, List[str]]:
    vtxt = len(tokenizer)
    tokens = ["<image>", "</image>"] + [f"<vis_{i:03d}>" for i in range(k)]
    tokenizer.add_tokens(tokens, special_tokens=False)
    return vtxt, tokens


def projector_warmup(wrapper: VirtualVocabCausalLM, vqvae: VQVAE, args, device: torch.device) -> None:
    codebook = vqvae.quantizer.codebook.detach().to(device)
    projector = nn.Linear(codebook.shape[1], wrapper.overlay.dim).to(device)
    nn.init.kaiming_uniform_(projector.weight, a=math.sqrt(5))
    opt = torch.optim.AdamW(projector.parameters(), lr=args.projector_lr)
    target_norm = wrapper.overlay.base.weight.detach().float().norm(dim=-1).mean().to(device)
    for _ in range(args.projector_epochs):
        opt.zero_grad(set_to_none=True)
        proj = projector(codebook)
        loss = (proj.norm(dim=-1).mean() - target_norm).pow(2) + 0.01 * proj.pow(2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        wrapper.overlay.new.weight[0].copy_(wrapper.overlay.base.weight.mean(dim=0))
        wrapper.overlay.new.weight[1].copy_(wrapper.overlay.base.weight.mean(dim=0))
        wrapper.overlay.new.weight[2:].copy_(projector(codebook).float())
        wrapper.expanded_head.new_weight.copy_(wrapper.overlay.new.weight)
        ratio = wrapper.overlay.new.weight[2:].norm(dim=-1).mean() / wrapper.overlay.base.weight.norm(dim=-1).mean()
        if ratio < 0.2 or ratio > 5.0:
            scale = float(torch.clamp(1.0 / ratio, 0.05, 20.0).item())
            wrapper.overlay.new.weight[2:].mul_(scale)
            wrapper.expanded_head.new_weight[2:].mul_(scale)
            print(f"rescaled visual virtual embeddings by {scale:.3f}")
        print(f"visual/text embedding norm ratio: {float(ratio):.3f}")


def load_lm_with_virtual_vocab(args, device: torch.device, k: int):
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    vtxt, tokens = add_virtual_tokens(tokenizer, k)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.lm_model, torch_dtype=dtype).to(device)
    base.config.use_cache = False
    for p in base.parameters():
        p.requires_grad_(False)
    wrapper = VirtualVocabCausalLM(base, num_new=2 + k).to(device)
    print(f"virtual vocab: text={vtxt}, total={wrapper.vocab_size}, tokenizer={len(tokenizer)}")
    return tokenizer, wrapper, vtxt


def apply_lora(wrapper: VirtualVocabCausalLM, args) -> VirtualVocabCausalLM:
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    wrapper.model = get_peft_model(wrapper.model, config)
    return wrapper


def loss_from_token_batch(model: VirtualVocabCausalLM, batch) -> torch.Tensor:
    input_ids, attention_mask, labels = batch[:3]
    return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss


@torch.no_grad()
def compute_ppl(model: VirtualVocabCausalLM, text_loader: DataLoader, max_batches: int = 25) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(text_loader):
        if i >= max_batches:
            break
        losses.append(float(loss_from_token_batch(model, batch).detach().cpu()))
    return math.exp(float(np.mean(losses)))


def mask_logits(logits: torch.Tensor, mode: str, vtxt: int, total_vocab: int) -> torch.Tensor:
    masked = logits.clone()
    if mode == "text":
        masked[..., vtxt:] = torch.finfo(masked.dtype).min
    elif mode == "image":
        masked[..., : vtxt + 2] = torch.finfo(masked.dtype).min
    else:
        raise ValueError(mode)
    return masked


@torch.no_grad()
def generate_text_answer(model: VirtualVocabCausalLM, prefix_ids: torch.Tensor, tokenizer, vtxt: int, max_new: int = 8) -> str:
    model.eval()
    ids = prefix_ids.clone()
    for _ in range(max_new):
        mask = torch.ones_like(ids, dtype=torch.long)
        logits = model(input_ids=ids, attention_mask=mask).logits[:, -1, :]
        logits = mask_logits(logits, "text", vtxt, model.vocab_size)
        next_id = int(torch.argmax(logits, dim=-1).item())
        if next_id == tokenizer.eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([[next_id]], device=ids.device)], dim=1)
    return tokenizer.decode(ids[0, prefix_ids.shape[1] :].tolist(), skip_special_tokens=True).strip()


@torch.no_grad()
def generate_image_codes(model: VirtualVocabCausalLM, prompt_ids: torch.Tensor, vtxt: int, k: int, temperature: float = 1.0) -> torch.Tensor:
    model.eval()
    ids = prompt_ids.clone()
    codes = []
    for _ in range(16):
        mask = torch.ones_like(ids)
        logits = model(input_ids=ids, attention_mask=mask).logits[:, -1, :]
        logits = mask_logits(logits, "image", vtxt, model.vocab_size) / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, 1).item())
        next_id = min(max(next_id, vtxt + 2), vtxt + 2 + k - 1)
        codes.append(next_id - vtxt - 2)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=ids.device)], dim=1)
    return torch.tensor(codes, dtype=torch.long, device=ids.device).view(1, 4, 4)


@torch.no_grad()
def evaluate_vqa(model: VirtualVocabCausalLM, loader: DataLoader, tokenizer, vtxt: int, max_examples: int = 500) -> Dict[str, object]:
    total = correct = 0
    by_template = defaultdict(lambda: [0, 0])
    by_class = defaultdict(lambda: [0, 0])
    y_true, y_pred = [], []
    examples = []
    for input_ids, attention_mask, labels, rows in tqdm(loader, desc="eval-vqa", leave=False):
        for i, row in enumerate(rows):
            real_ids = input_ids[i][attention_mask[i].bool()]
            real_labels = labels[i][attention_mask[i].bool()]
            labelled = torch.where(real_labels != -100)[0]
            prefix_len = int(labelled[0].item()) if labelled.numel() else real_ids.numel()
            prefix_ids = real_ids[:prefix_len].unsqueeze(0)
            pred = generate_text_answer(model, prefix_ids, tokenizer, vtxt).lower().strip(" .\n\t")
            gold = str(row["answer"]).lower().strip()
            hit = pred == gold
            total += 1
            correct += int(hit)
            by_template[row["template"]][0] += int(hit)
            by_template[row["template"]][1] += 1
            by_class[row["label"]][0] += int(hit)
            by_class[row["label"]][1] += 1
            if row["template"] == "class":
                y_true.append(gold)
                y_pred.append(pred)
            if len(examples) < 6:
                examples.append({"q": row["question"], "gold": gold, "pred": pred, "ok": hit, "label": row["label"]})
            if total >= max_examples:
                break
        if total >= max_examples:
            break
    cm = confusion_matrix(y_true, y_pred, labels=list(SHAPES)).tolist() if y_true else []
    return {
        "overall": correct / max(total, 1),
        "per_template": {k: v[0] / max(v[1], 1) for k, v in by_template.items()},
        "per_class": {k: v[0] / max(v[1], 1) for k, v in by_class.items()},
        "shape_confusion": cm,
        "examples": examples,
    }


def train_mixed(args, model: VirtualVocabCausalLM, vqa_loader, img_loader, text_loader, device: torch.device, lam: float, gamma_img: float):
    params = [
        {"params": [p for p in model.model.parameters() if p.requires_grad], "lr": args.lora_lr},
        {"params": [model.overlay.new.weight, model.expanded_head.new_weight, model.expanded_head.new_bias], "lr": args.visual_lr},
    ]
    opt = torch.optim.AdamW(params)
    steps_per_epoch = max(len(vqa_loader), len(img_loader), len(text_loader))
    total_steps = max(1, args.lm_epochs * steps_per_epoch // args.grad_accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lora_lr, args.visual_lr], total_steps=total_steps, pct_start=0.10)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    vqa_iter, img_iter, text_iter = infinite(vqa_loader), infinite(img_loader), infinite(text_loader)
    opt.zero_grad(set_to_none=True)
    global_step = 0
    for epoch in range(args.lm_epochs):
        logs = defaultdict(list)
        model.train()
        for step in tqdm(range(steps_per_epoch), desc=f"lm-{epoch+1}"):
            vqa_batch = next(vqa_iter)
            img_batch = next(img_iter)
            text_batch = next(text_iter)
            with autocast_context(device):
                lvqa = loss_from_token_batch(model, vqa_batch)
                limg = loss_from_token_batch(model, img_batch)
                ltxt = loss_from_token_batch(model, text_batch)
                loss = (lvqa + gamma_img * limg + lam * ltxt) / args.grad_accum
            scaler.scale(loss).backward()
            logs["lvqa"].append(float(lvqa.detach().cpu()))
            logs["limg"].append(float(limg.detach().cpu()))
            logs["ltxt"].append(float(ltxt.detach().cpu()))
            if (step + 1) % args.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1
            if step % 25 == 0:
                print(f"step {step}: lvqa={np.mean(logs['lvqa']):.3f} limg={np.mean(logs['limg']):.3f} ltxt={np.mean(logs['ltxt']):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm-model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-per-class", type=int, default=1000)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--vqvae-epochs", type=int, default=80)
    parser.add_argument("--vqvae-batch-size", type=int, default=64)
    parser.add_argument("--vqvae-lr", type=float, default=3e-4)
    parser.add_argument("--projector-epochs", type=int, default=300)
    parser.add_argument("--projector-lr", type=float, default=5e-4)
    parser.add_argument("--lm-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lambda-replay", type=float, default=0.2)
    parser.add_argument("--gamma-img", type=float, default=0.5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-lr", type=float, default=5e-4)
    parser.add_argument("--visual-lr", type=float, default=5e-5)
    parser.add_argument("--alpaca-examples", type=int, default=1000)
    parser.add_argument("--eval-examples", type=int, default=500)
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_per_class = min(args.n_per_class, 30)
        args.vqvae_epochs = min(args.vqvae_epochs, 2)
        args.projector_epochs = min(args.projector_epochs, 5)
        args.lm_epochs = min(args.lm_epochs, 1)
        args.batch_size = min(args.batch_size, 4)
        args.alpaca_examples = min(args.alpaca_examples, 20)
        args.eval_examples = min(args.eval_examples, 20)

    set_seed(args.seed)
    device = device_from_arg(args.device)
    out_dir = Path(args.weights_dir)
    out_dir.mkdir(exist_ok=True)
    t0 = time.time()

    train_samples, val_samples = generate_dataset(args.n_per_class, args.seed)
    vqvae = train_vqvae(args, train_samples, val_samples, device)
    analysis = codebook_analysis(vqvae, val_samples, device, out_dir)
    print("VQ-VAE analysis:", analysis)
    train_codes = encode_all(vqvae, train_samples, device)
    val_codes = encode_all(vqvae, val_samples, device)
    vqvae.to("cpu")

    tokenizer, model, vtxt = load_lm_with_virtual_vocab(args, device, args.codebook_size)
    projector_warmup(model, vqvae, args, device)
    model = apply_lora(model, args)
    lm_start = snapshot_trainable(model)

    train_vqa = EncodedMultimodalDataset(train_samples, train_codes, tokenizer, "vqa", vtxt)
    val_vqa = EncodedMultimodalDataset(val_samples, val_codes, tokenizer, "vqa", vtxt)
    train_img = EncodedMultimodalDataset(train_samples, train_codes, tokenizer, "imagegen", vtxt)
    text_data = AlpacaTextDataset(tokenizer, args.alpaca_examples)

    vqa_loader = DataLoader(train_vqa, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_token_rows(b, tokenizer.pad_token_id, device))
    val_vqa_loader = DataLoader(val_vqa, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_token_rows(b, tokenizer.pad_token_id, device))
    img_loader = DataLoader(train_img, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_token_rows(b, tokenizer.pad_token_id, device))
    text_loader = DataLoader(text_data, batch_size=args.text_batch_size, shuffle=True, collate_fn=lambda b: collate_text(b, tokenizer, device))

    ppl0 = compute_ppl(model, text_loader)
    print(f"PPL0: {ppl0:.3f}")
    conditions = [(args.lambda_replay, args.gamma_img)]
    if args.run_ablation:
        conditions = [(0.0, 0.0), (0.05, 0.05), (0.2, 0.5), (0.5, 0.5)]
    results = []
    for lam, gam in conditions:
        restore_trainable(model, lm_start)
        train_mixed(args, model, vqa_loader, img_loader, text_loader, device, lam, gam)
        vqa_metrics = evaluate_vqa(model, val_vqa_loader, tokenizer, vtxt, args.eval_examples)
        ppl = compute_ppl(model, text_loader)
        results.append({"lambda": lam, "gamma_img": gam, "vqa": vqa_metrics, "ppl": ppl, "R": ppl / ppl0})
        print(json.dumps(results[-1], indent=2))

    torch.save({"model": model.state_dict(), "vtxt": vtxt, "args": vars(args)}, out_dir / "part_b_lora_virtual_vocab.pt")
    summary = {"vqvae": analysis, "ppl0": ppl0, "runs": results, "wall_minutes": (time.time() - t0) / 60.0}
    (out_dir / "part_b_results.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'part_b_results.json'}")


if __name__ == "__main__":
    main()
