#!/usr/bin/env python3
"""Part A solution: continuous connector VLM on CIFAR-10.

This script implements the coding tasks in DVLM PA3 Part A:
data construction, CLIP patch extraction, MLP connector, three training
phases, Alpaca replay, VQA evaluation, and modality-gap diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel


CIFAR_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
VEHICLES = {"airplane", "automobile", "ship", "truck"}
LIVING = {"bird", "cat", "deer", "dog", "frog", "horse"}
CAN_FLY = {"airplane", "bird"}
ANIMALS = LIVING

CAPTION_TEMPLATES = (
    "a photo of a {label}.",
    "this image shows a {label}.",
    "there is a {label} in the picture.",
    "a small {label} is visible.",
    "the object is a {label}.",
    "a CIFAR-10 image of a {label}.",
)

VQA_TEMPLATES = (
    ("what object is shown?", "class"),
    ("is there a {label}?", "presence"),
    ("vehicle or living thing?", "vehicle_living"),
    ("can it fly?", "can_fly"),
    ("is this an animal?", "animal"),
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


def select_stratified_indices(targets: Sequence[int], per_class: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, target in enumerate(targets):
        by_class[int(target)].append(idx)
    selected: List[int] = []
    for cls in range(len(CIFAR_CLASSES)):
        candidates = by_class[cls][:]
        rng.shuffle(candidates)
        selected.extend(candidates[:per_class])
    rng.shuffle(selected)
    return selected


def answer_for(label: str, template_name: str) -> str:
    if template_name == "class":
        return label
    if template_name == "presence":
        return "yes"
    if template_name == "vehicle_living":
        return "vehicle" if label in VEHICLES else "living"
    if template_name == "can_fly":
        return "yes" if label in CAN_FLY else "no"
    if template_name == "animal":
        return "yes" if label in ANIMALS else "no"
    raise ValueError(f"unknown VQA template {template_name}")


@dataclass(frozen=True)
class CifarItem:
    index: int
    label_id: int
    label: str


class CifarCaptionDataset(Dataset):
    def __init__(self, items: Sequence[CifarItem], split: CIFAR10):
        self.items = list(items)
        self.split = split

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.items[idx]
        image, _ = self.split[item.index]
        caption = CAPTION_TEMPLATES[idx % len(CAPTION_TEMPLATES)].format(label=item.label)
        return {"image": image, "label_id": item.label_id, "label": item.label, "caption": caption}


class CifarVQADataset(Dataset):
    def __init__(self, items: Sequence[CifarItem], split: CIFAR10):
        self.samples: List[Tuple[CifarItem, str, str, str]] = []
        self.split = split
        for item in items:
            for question_template, template_name in VQA_TEMPLATES:
                question = question_template.format(label=item.label)
                self.samples.append((item, question, answer_for(item.label, template_name), template_name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item, question, answer, template = self.samples[idx]
        image, _ = self.split[item.index]
        return {
            "image": image,
            "label_id": item.label_id,
            "label": item.label,
            "question": question,
            "answer": answer,
            "template": template,
        }


class AlpacaTextDataset(Dataset):
    def __init__(self, tokenizer, max_examples: int = 1000, max_length: int = 384):
        data = load_dataset("tatsu-lab/alpaca", split=f"train[:{max_examples}]")
        self.samples: List[Dict[str, torch.Tensor]] = []
        for row in data:
            instruction = row["instruction"].strip()
            inp = row.get("input", "").strip()
            output = row["output"].strip()
            prompt = f"### Instruction:\n{instruction}\n"
            if inp:
                prompt += f"\n### Input:\n{inp}\n"
            text = f"{prompt}\n### Response:\n{output}{tokenizer.eos_token}"
            enc = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)
            self.samples.append({"input_ids": torch.tensor(enc["input_ids"], dtype=torch.long)})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class MLPConnector(nn.Module):
    def __init__(self, in_dim: int = 768, hidden_dim: int = 960, out_dim: int = 960):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(patch_tokens)


def build_cifar(args) -> Tuple[CIFAR10, CIFAR10, List[CifarItem], List[CifarItem]]:
    root = Path(args.data_dir)
    train_split = CIFAR10(root=str(root), train=True, download=True)
    test_split = CIFAR10(root=str(root), train=False, download=True)
    train_indices = select_stratified_indices(train_split.targets, args.train_per_class, args.seed)
    test_indices = select_stratified_indices(test_split.targets, args.test_per_class, args.seed)
    train_items = [CifarItem(i, int(train_split.targets[i]), CIFAR_CLASSES[int(train_split.targets[i])]) for i in train_indices]
    test_items = [CifarItem(i, int(test_split.targets[i]), CIFAR_CLASSES[int(test_split.targets[i])]) for i in test_indices]
    return train_split, test_split, train_items, test_items


def load_models(args, device: torch.device):
    processor = CLIPImageProcessor.from_pretrained(args.clip_model)
    print("CLIP mean/std:", processor.image_mean, processor.image_std)

    clip = CLIPVisionModel.from_pretrained(args.clip_model).to(device)
    clip.eval()
    for param in clip.parameters():
        param.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    lm = AutoModelForCausalLM.from_pretrained(args.lm_model, torch_dtype=torch_dtype).to(device)
    lm.config.use_cache = False
    print("LM hidden/vocab:", lm.config.hidden_size, lm.config.vocab_size)
    assert lm.config.hidden_size == args.d_lm
    assert lm.config.vocab_size == args.vocab_size
    return processor, clip, tokenizer, lm


def make_lora_model(lm: nn.Module, args) -> nn.Module:
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    return get_peft_model(lm, config)


def freeze_lm(lm: nn.Module) -> None:
    for param in lm.parameters():
        param.requires_grad_(False)


def collate_caption(batch, processor, tokenizer, device: torch.device):
    images = [row["image"] for row in batch]
    pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
    captions = [row["caption"] for row in batch]
    enc = tokenizer(captions, padding=True, add_special_tokens=False, return_tensors="pt").to(device)
    return pixels, enc, batch


def collate_vqa(batch, processor, tokenizer, device: torch.device):
    images = [row["image"] for row in batch]
    pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
    questions = tokenizer([row["question"] for row in batch], padding=True, add_special_tokens=False, return_tensors="pt").to(device)
    answers = tokenizer([row["answer"] + tokenizer.eos_token for row in batch], padding=True, add_special_tokens=False, return_tensors="pt").to(device)
    return pixels, questions, answers, batch


def collate_text(batch, tokenizer, device: torch.device):
    ids = [row["input_ids"] for row in batch]
    max_len = max(x.numel() for x in ids)
    input_ids = torch.full((len(ids), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(ids), max_len), dtype=torch.long)
    for i, seq in enumerate(ids):
        input_ids[i, -seq.numel() :] = seq
        attention_mask[i, -seq.numel() :] = 1
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


@torch.no_grad()
def clip_patches(clip: CLIPVisionModel, pixels: torch.Tensor) -> torch.Tensor:
    hidden = clip(pixel_values=pixels).last_hidden_state
    assert hidden.shape[1] == 50, f"expected CLS + 49 patches, got {hidden.shape}"
    return hidden[:, 1:, :].float()


def pad_embed_sequences(seqs: List[torch.Tensor], labels: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(seqs)
    max_len = max(x.shape[0] for x in seqs)
    dim = seqs[0].shape[-1]
    embeds = seqs[0].new_zeros(batch, max_len, dim)
    label_pad = torch.full((batch, max_len), -100, dtype=torch.long, device=seqs[0].device)
    mask = torch.zeros((batch, max_len), dtype=torch.long, device=seqs[0].device)
    for i, (emb, lab) in enumerate(zip(seqs, labels)):
        length = emb.shape[0]
        embeds[i, :length] = emb
        label_pad[i, :length] = lab
        mask[i, :length] = 1
    return embeds, label_pad, mask


def caption_loss(lm, connector, clip, pixels, caption_enc, tokenizer) -> torch.Tensor:
    patches = clip_patches(clip, pixels)
    visual = connector(patches)
    emb_layer = lm.get_input_embeddings()
    bos_ids = torch.full((pixels.shape[0], 1), tokenizer.bos_token_id or tokenizer.eos_token_id, device=pixels.device, dtype=torch.long)
    bos_emb = emb_layer(bos_ids)
    cap_ids = caption_enc["input_ids"]
    cap_mask = caption_enc["attention_mask"]
    cap_emb = emb_layer(cap_ids)

    seqs, labs = [], []
    for i in range(pixels.shape[0]):
        valid_cap = cap_mask[i].bool()
        emb = torch.cat([bos_emb[i], visual[i], cap_emb[i, valid_cap]], dim=0)
        lab = torch.full((1 + visual.shape[1] + valid_cap.sum().item(),), -100, dtype=torch.long, device=pixels.device)
        lab[1 + visual.shape[1] :] = cap_ids[i, valid_cap]
        seqs.append(emb)
        labs.append(lab)
    inputs_embeds, labels, attention_mask = pad_embed_sequences(seqs, labs)
    return lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss


def vqa_loss(lm, connector, clip, pixels, q_enc, a_enc, tokenizer) -> torch.Tensor:
    patches = clip_patches(clip, pixels)
    visual = connector(patches)
    emb_layer = lm.get_input_embeddings()
    bos_ids = torch.full((pixels.shape[0], 1), tokenizer.bos_token_id or tokenizer.eos_token_id, device=pixels.device, dtype=torch.long)
    bos_emb = emb_layer(bos_ids)
    q_emb = emb_layer(q_enc["input_ids"])
    a_emb = emb_layer(a_enc["input_ids"])

    seqs, labs = [], []
    for i in range(pixels.shape[0]):
        q_valid = q_enc["attention_mask"][i].bool()
        a_valid = a_enc["attention_mask"][i].bool()
        emb = torch.cat([bos_emb[i], visual[i], q_emb[i, q_valid], a_emb[i, a_valid]], dim=0)
        prefix = 1 + visual.shape[1] + q_valid.sum().item()
        lab = torch.full((emb.shape[0],), -100, dtype=torch.long, device=pixels.device)
        lab[prefix:] = a_enc["input_ids"][i, a_valid]
        seqs.append(emb)
        labs.append(lab)
    inputs_embeds, labels, attention_mask = pad_embed_sequences(seqs, labs)
    return lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss


def text_loss(lm, batch) -> torch.Tensor:
    input_ids, attention_mask, labels = batch
    return lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss


@torch.no_grad()
def compute_ppl(lm, loader: DataLoader, max_batches: int | None = None) -> float:
    lm.eval()
    losses: List[float] = []
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        loss = text_loss(lm, batch)
        losses.append(float(loss.detach().cpu()))
    return math.exp(float(np.mean(losses)))


@torch.no_grad()
def norm_ratio(lm, connector, clip, loader: DataLoader, max_batches: int = 4) -> float:
    connector.eval()
    visual_norms, text_norms = [], []
    emb = lm.get_input_embeddings().weight.detach().float()
    text_norms.append(emb.norm(dim=-1).mean().item())
    for step, (pixels, _, _) in enumerate(loader):
        if step >= max_batches:
            break
        v = connector(clip_patches(clip, pixels)).float()
        visual_norms.append(v.norm(dim=-1).mean().item())
    return float(np.mean(visual_norms) / np.mean(text_norms))


@torch.no_grad()
def greedy_generate_from_embeds(lm, prefix_embeds: torch.Tensor, tokenizer, max_new_tokens: int = 12) -> str:
    lm.eval()
    generated: List[int] = []
    inputs_embeds = prefix_embeds
    attention_mask = torch.ones(prefix_embeds.shape[:2], dtype=torch.long, device=prefix_embeds.device)
    past = None
    for _ in range(max_new_tokens):
        out = lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        token_id = int(torch.argmax(logits, dim=-1).item())
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
        past = out.past_key_values
        inputs_embeds = lm.get_input_embeddings()(torch.tensor([[token_id]], device=prefix_embeds.device))
        attention_mask = torch.ones((1, 1), dtype=torch.long, device=prefix_embeds.device)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def evaluate_vqa(lm, connector, clip, loader: DataLoader, tokenizer, max_examples: int = 500) -> Dict[str, object]:
    lm.eval()
    connector.eval()
    total = 0
    correct = 0
    by_template = defaultdict(lambda: [0, 0])
    by_class = defaultdict(lambda: [0, 0])
    examples = []
    emb_layer = lm.get_input_embeddings()
    for pixels, q_enc, _a_enc, rows in tqdm(loader, desc="eval-vqa", leave=False):
        patches = clip_patches(clip, pixels)
        visual = connector(patches)
        bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        bos = emb_layer(torch.full((pixels.shape[0], 1), bos_id, dtype=torch.long, device=pixels.device))
        q_emb = emb_layer(q_enc["input_ids"])
        for i, row in enumerate(rows):
            q_valid = q_enc["attention_mask"][i].bool()
            prefix = torch.cat([bos[i : i + 1], visual[i : i + 1], q_emb[i : i + 1, q_valid]], dim=1)
            pred = greedy_generate_from_embeds(lm, prefix, tokenizer, max_new_tokens=8).lower().strip(" .\n\t")
            gold = str(row["answer"]).lower().strip()
            hit = pred == gold
            total += 1
            correct += int(hit)
            by_template[row["template"]][0] += int(hit)
            by_template[row["template"]][1] += 1
            by_class[row["label"]][0] += int(hit)
            by_class[row["label"]][1] += 1
            if len(examples) < 6:
                examples.append({"q": row["question"], "gold": gold, "pred": pred, "ok": hit, "label": row["label"]})
            if total >= max_examples:
                break
        if total >= max_examples:
            break
    return {
        "overall": correct / max(total, 1),
        "per_template": {k: v[0] / max(v[1], 1) for k, v in by_template.items()},
        "per_class": {k: v[0] / max(v[1], 1) for k, v in by_class.items()},
        "examples": examples,
    }


@torch.no_grad()
def modality_gap(lm, connector, clip, loader: DataLoader, tokenizer, max_examples: int = 200) -> Dict[str, float]:
    lm.eval()
    connector.eval()
    visual_vectors, text_vectors = [], []
    seen = 0
    emb_layer = lm.get_input_embeddings()
    for pixels, q_enc, _a_enc, _rows in loader:
        visual = connector(clip_patches(clip, pixels)).mean(dim=1)
        visual_vectors.append(F.normalize(visual.float(), dim=-1).cpu())
        q_ids = q_enc["input_ids"]
        q_mask = q_enc["attention_mask"].bool()
        q_emb = emb_layer(q_ids).float()
        pooled = (q_emb * q_mask.unsqueeze(-1)).sum(dim=1) / q_mask.sum(dim=1, keepdim=True).clamp_min(1)
        text_vectors.append(F.normalize(pooled, dim=-1).cpu())
        seen += pixels.shape[0]
        if seen >= max_examples:
            break
    v = torch.cat(visual_vectors, dim=0)[:max_examples]
    t = torch.cat(text_vectors, dim=0)[:max_examples]
    v_mean = F.normalize(v.mean(dim=0), dim=0)
    t_mean = F.normalize(t.mean(dim=0), dim=0)
    cross = (v @ t.T).mean().item()
    vv = (v @ v.T).mean().item()
    tt = (t @ t.T).mean().item()
    return {"MG": (v_mean - t_mean).norm().item(), "visual_visual_cos": vv, "text_text_cos": tt, "cross_cos": cross}


def train_phase1(args, lm, connector, clip, train_loader, val_caption_loader, tokenizer, device):
    freeze_lm(lm)
    for p in connector.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(connector.parameters(), lr=args.phase1_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    lm.eval()
    connector.train()
    for epoch in range(args.phase1_epochs):
        running = []
        for pixels, caption_enc, _rows in tqdm(train_loader, desc=f"phase1-{epoch+1}"):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                loss = caption_loss(lm, connector, clip, pixels, caption_enc, tokenizer)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running.append(float(loss.detach().cpu()))
        print(f"phase1 epoch {epoch+1} loss {np.mean(running):.4f}")
    ratio = norm_ratio(lm, connector, clip, val_caption_loader)
    print(f"connector/text norm ratio after phase1: {ratio:.3f}")
    if ratio < 0.3 or ratio > 3.0:
        scale = float(np.clip(1.0 / ratio, 0.1, 10.0))
        connector.net[-1].weight.data.mul_(scale)
        connector.net[-1].bias.data.mul_(scale)
        print(f"rescaled connector final layer by {scale:.3f}")
    Path(args.weights_dir).mkdir(exist_ok=True)
    torch.save(connector.state_dict(), Path(args.weights_dir) / "connector_phaseA1.pt")


def train_phase2(args, lm, connector, clip, vqa_loader, text_loader, tokenizer, device, lambda_replay: float, phase_name: str):
    for p in connector.parameters():
        p.requires_grad_(True)
    lm.train()
    connector.train()
    trainable = list(p for p in lm.parameters() if p.requires_grad) + list(connector.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.phase2_lr)
    total_steps = max(1, args.phase2_epochs * len(vqa_loader) // args.grad_accum)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.phase2_lr,
        total_steps=total_steps,
        pct_start=0.10,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    text_iter = iter(text_loader)
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.phase2_epochs):
        running = []
        for step, (pixels, q_enc, a_enc, _rows) in enumerate(tqdm(vqa_loader, desc=f"{phase_name}-{epoch+1}")):
            try:
                text_batch = next(text_iter)
            except StopIteration:
                text_iter = iter(text_loader)
                text_batch = next(text_iter)
            with autocast_context(device):
                lvqa = vqa_loss(lm, connector, clip, pixels, q_enc, a_enc, tokenizer)
                ltxt = text_loss(lm, text_batch)
                loss = (lvqa + lambda_replay * ltxt) / args.grad_accum
            scaler.scale(loss).backward()
            running.append(float((lvqa + lambda_replay * ltxt).detach().cpu()))
            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
        print(f"{phase_name} epoch {epoch+1} loss {np.mean(running):.4f}")
    torch.save({"connector": connector.state_dict(), "lm": lm.state_dict()}, Path(args.weights_dir) / f"{phase_name}.pt")


def train_phase3(args, lm, connector, clip, vqa_loader, tokenizer, device):
    for p in connector.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad] + list(connector.parameters()), lr=args.phase3_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    lm.train()
    connector.train()
    for epoch in range(args.phase3_epochs):
        running = []
        for pixels, q_enc, a_enc, _rows in tqdm(vqa_loader, desc=f"phase3-{epoch+1}"):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                loss = vqa_loss(lm, connector, clip, pixels, q_enc, a_enc, tokenizer)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running.append(float(loss.detach().cpu()))
        print(f"phase3 epoch {epoch+1} loss {np.mean(running):.4f}")
    torch.save({"connector": connector.state_dict(), "lm": lm.state_dict()}, Path(args.weights_dir) / "connector_phaseA3.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--lm-model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--text-batch-size", type=int, default=4)
    parser.add_argument("--d-lm", type=int, default=960)
    parser.add_argument("--vocab-size", type=int, default=49152)
    parser.add_argument("--phase1-epochs", type=int, default=1)
    parser.add_argument("--phase2-epochs", type=int, default=1)
    parser.add_argument("--phase3-epochs", type=int, default=1)
    parser.add_argument("--phase1-lr", type=float, default=3e-4)
    parser.add_argument("--phase2-lr", type=float, default=5e-4)
    parser.add_argument("--phase3-lr", type=float, default=2e-4)
    parser.add_argument("--lambda-replay", type=float, default=0.2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--alpaca-examples", type=int, default=1000)
    parser.add_argument("--eval-examples", type=int, default=500)
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny local run for pipeline debugging")
    args = parser.parse_args()

    if args.smoke:
        args.train_per_class = min(args.train_per_class, 20)
        args.test_per_class = min(args.test_per_class, 5)
        args.batch_size = min(args.batch_size, 4)
        args.text_batch_size = min(args.text_batch_size, 2)
        args.alpaca_examples = min(args.alpaca_examples, 20)
        args.eval_examples = min(args.eval_examples, 20)

    set_seed(args.seed)
    device = device_from_arg(args.device)
    Path(args.weights_dir).mkdir(exist_ok=True)

    train_split, test_split, train_items, test_items = build_cifar(args)
    processor, clip, tokenizer, lm = load_models(args, device)
    connector = MLPConnector().to(device)
    trainable, total = count_trainable_parameters(connector)
    print(f"connector params: {trainable:,} / expected approx 1.66M")

    caption_train = CifarCaptionDataset(train_items, train_split)
    caption_test = CifarCaptionDataset(test_items, test_split)
    vqa_train = CifarVQADataset(train_items, train_split)
    vqa_val = CifarVQADataset(test_items, test_split)
    alpaca = AlpacaTextDataset(tokenizer, args.alpaca_examples)

    caption_loader = DataLoader(caption_train, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_caption(b, processor, tokenizer, device))
    caption_eval_loader = DataLoader(caption_test, batch_size=args.eval_batch_size, shuffle=False, collate_fn=lambda b: collate_caption(b, processor, tokenizer, device))
    vqa_loader = DataLoader(vqa_train, batch_size=max(1, args.batch_size // 2), shuffle=True, collate_fn=lambda b: collate_vqa(b, processor, tokenizer, device))
    vqa_eval_loader = DataLoader(vqa_val, batch_size=args.eval_batch_size, shuffle=False, collate_fn=lambda b: collate_vqa(b, processor, tokenizer, device))
    text_loader = DataLoader(alpaca, batch_size=args.text_batch_size, shuffle=True, collate_fn=lambda b: collate_text(b, tokenizer, device))

    ppl0 = compute_ppl(lm, text_loader, max_batches=25)
    print(f"PPL0 on Alpaca subset: {ppl0:.3f}")

    t0 = time.time()
    train_phase1(args, lm, connector, clip, caption_loader, caption_eval_loader, tokenizer, device)

    lm = make_lora_model(lm, args)
    lm.print_trainable_parameters()
    phase2_lm_start = snapshot_trainable(lm)
    phase2_connector_start = {name: param.detach().cpu().clone() for name, param in connector.named_parameters()}
    lambdas = [args.lambda_replay]
    if args.run_ablation:
        lambdas = [0.0, 0.05, 0.2, 0.5]
    results = []
    for lam in lambdas:
        restore_trainable(lm, phase2_lm_start)
        for name, param in connector.named_parameters():
            param.data.copy_(phase2_connector_start[name].to(param.device, dtype=param.dtype))
        phase = f"connector_phaseA2_lambda{lam:g}"
        train_phase2(args, lm, connector, clip, vqa_loader, text_loader, tokenizer, device, lam, phase)
        metrics = evaluate_vqa(lm, connector, clip, vqa_eval_loader, tokenizer, args.eval_examples)
        ppl = compute_ppl(lm, text_loader, max_batches=25)
        gap = modality_gap(lm, connector, clip, vqa_eval_loader, tokenizer, max_examples=min(200, args.eval_examples))
        row = {"phase": phase, "lambda": lam, "vqa": metrics, "ppl": ppl, "R": ppl / ppl0, "gap": gap}
        results.append(row)
        print(json.dumps(row, indent=2))

    train_phase3(args, lm, connector, clip, vqa_loader, tokenizer, device)
    final_metrics = evaluate_vqa(lm, connector, clip, vqa_eval_loader, tokenizer, args.eval_examples)
    final_ppl = compute_ppl(lm, text_loader, max_batches=25)
    final_gap = modality_gap(lm, connector, clip, vqa_eval_loader, tokenizer, max_examples=min(200, args.eval_examples))
    summary = {
        "ppl0": ppl0,
        "phase2": results,
        "phase3": {"vqa": final_metrics, "ppl": final_ppl, "R": final_ppl / ppl0, "gap": final_gap},
        "wall_minutes": (time.time() - t0) / 60.0,
    }
    out_path = Path(args.weights_dir) / "part_a_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
