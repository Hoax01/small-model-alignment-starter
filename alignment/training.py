"""Small, explicit training loops for the assignment's SFT and DPO tracks."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import format_prompt
from .objectives import (
    _response_logprobs_from_batch,
    build_dpo_batch,
    build_sft_batch,
    compute_dpo_loss,
    compute_dpo_loss_from_logps,
    compute_sft_loss,
)


@dataclass
class SFTConfig:
    output_dir: str
    device: str = "cuda:0"
    max_length: int = 512
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    epochs: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = True
    seed: int = 42


@dataclass
class DPOConfig:
    output_dir: str
    policy_device: str = "cuda:0"
    reference_device: str = "cuda:1"
    max_length: int = 512
    max_prompt_length: int = 256
    batch_size: int = 1
    gradient_accumulation_steps: int = 32
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    beta: float = 0.1
    epochs: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = True
    optimizer: str = "adamw"
    seed: int = 42


def _amp(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    from contextlib import nullcontext
    return nullcontext()


def _scaler(device: torch.device, enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and enabled)


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def _save(model, tokenizer, path: Path, state: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    if state is not None:
        (path / "training_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def _prompt_batch(rows):
    return [format_prompt(row["prompt"]) for row in rows]


def _dpo_prompt_batch(rows, tokenizer, max_prompt_length: int):
    if max_prompt_length <= 0:
        raise ValueError("max_prompt_length must be positive")
    prompts = _prompt_batch(rows)
    truncated = []
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) > max_prompt_length:
            prompt = tokenizer.decode(token_ids[:max_prompt_length], skip_special_tokens=True)
        truncated.append(prompt)
    return truncated


def _scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup = int(total_steps * max(0.0, warmup_ratio))

    def factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def evaluate_sft(model, tokenizer, rows, config: SFTConfig) -> float:
    device = torch.device(config.device)
    was_training = model.training
    model.eval()
    loader = DataLoader(rows, batch_size=config.batch_size, shuffle=False, collate_fn=list)
    loss_sum, token_sum = 0.0, 0
    for items in loader:
        prompts = _prompt_batch(items)
        responses = [row["response"] for row in items]
        batch = build_sft_batch(tokenizer, prompts, responses, config.max_length)
        count = int(batch["labels"][:, 1:].ne(-100).sum().item())
        with _amp(device, config.fp16):
            loss = compute_sft_loss(model, tokenizer, prompts, responses, config.max_length)
        loss_sum += float(loss.item()) * count
        token_sum += count
    model.train(was_training)
    return loss_sum / max(1, token_sum)


def train_sft(model, tokenizer, train_rows, validation_rows, config: SFTConfig) -> dict:
    """Run response-only SFT with accumulation, fp16 autocast, and checkpoints."""
    _seed(config.seed)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = out / "metrics.jsonl"
    if metrics.exists():
        metrics.unlink()

    device = torch.device(config.device)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if any(p.dtype != torch.float32 for p in trainable):
        raise ValueError("Trainable weights must remain FP32 while using fp16 gradient scaling")

    loader = DataLoader(
        train_rows, batch_size=config.batch_size, shuffle=True, collate_fn=list,
        generator=torch.Generator().manual_seed(config.seed),
    )
    if not len(loader):
        raise ValueError("The SFT training set is empty")

    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    updates_per_epoch = math.ceil(len(loader) / config.gradient_accumulation_steps)
    scheduler = _scheduler(optimizer, max(1, updates_per_epoch * config.epochs), config.warmup_ratio)
    scaler = _scaler(device, config.fp16)
    best_val, updates = float("inf"), 0

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_values = []
        started = time.time()
        for index, items in enumerate(loader):
            prompts, responses = _prompt_batch(items), [row["response"] for row in items]
            group_start = (index // config.gradient_accumulation_steps) * config.gradient_accumulation_steps
            group_size = min(config.gradient_accumulation_steps, len(loader) - group_start)
            with _amp(device, config.fp16):
                loss = compute_sft_loss(model, tokenizer, prompts, responses, config.max_length)
            scaler.scale(loss / group_size).backward()
            loss_values.append(float(loss.detach().item()))

            if (index + 1) % config.gradient_accumulation_steps == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                updates += 1

        val_loss = evaluate_sft(model, tokenizer, validation_rows, config) if validation_rows else None
        row = {
            "track": "sft", "epoch": epoch + 1, "step": updates,
            "train_loss": sum(loss_values) / max(1, len(loss_values)),
            "validation_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - started, 2),
        }
        _append_jsonl(metrics, row)
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            _save(model, tokenizer, out / "best", row)

    result = {
        "config": asdict(config),
        "best_validation_loss": None if math.isinf(best_val) else best_val,
        "updates": updates,
    }
    _save(model, tokenizer, out / "final", result)
    return result


@torch.no_grad()
def evaluate_dpo(policy, reference, tokenizer, rows, config: DPOConfig) -> dict[str, float]:
    policy_device, reference_device = torch.device(config.policy_device), torch.device(config.reference_device)
    was_training = policy.training
    policy.eval()
    reference.eval()
    loader = DataLoader(rows, batch_size=config.batch_size, shuffle=False, collate_fn=list)
    weighted_loss, count = 0.0, 0
    margins, correct = [], 0

    for items in loader:
        prompts = _dpo_prompt_batch(items, tokenizer, config.max_prompt_length)
        chosen, rejected = [r["chosen"] for r in items], [r["rejected"] for r in items]
        batch = build_dpo_batch(tokenizer, prompts, chosen, rejected, config.max_length)
        with _amp(policy_device, config.fp16):
            pc = _response_logprobs_from_batch(policy, batch["chosen"])
            pr = _response_logprobs_from_batch(policy, batch["rejected"])
        with _amp(reference_device, config.fp16):
            rc = _response_logprobs_from_batch(reference, batch["chosen"])
            rr = _response_logprobs_from_batch(reference, batch["rejected"])
        rc, rr = rc.to(policy_device), rr.to(policy_device)
        loss = compute_dpo_loss_from_logps(pc, pr, rc, rr, config.beta)
        reward_margin = config.beta * ((pc - rc) - (pr - rr))
        weighted_loss += float(loss.item()) * len(items)
        count += len(items)
        margins.extend(reward_margin.float().cpu().tolist())
        correct += int((reward_margin > 0).sum().item())

    policy.train(was_training)
    return {
        "loss": weighted_loss / max(1, count),
        "reward_accuracy": correct / max(1, count),
        "reward_margin": sum(margins) / max(1, count),
    }


def train_dpo(policy, reference, tokenizer, train_rows, validation_rows, config: DPOConfig) -> dict:
    """Run from-scratch DPO with a frozen reference model and reward diagnostics."""
    _seed(config.seed)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = out / "metrics.jsonl"
    if metrics.exists():
        metrics.unlink()

    policy_device, reference_device = torch.device(config.policy_device), torch.device(config.reference_device)
    policy.to(policy_device)
    reference.to(reference_device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    if any(p.dtype != torch.float32 for p in trainable):
        raise ValueError("Trainable policy weights must remain FP32 while using fp16 gradient scaling")

    loader = DataLoader(
        train_rows, batch_size=config.batch_size, shuffle=True, collate_fn=list,
        generator=torch.Generator().manual_seed(config.seed),
    )
    if not len(loader):
        raise ValueError("The DPO training set is empty")
    if config.optimizer.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(trainable, lr=config.learning_rate)
    else:
        optimizer = torch.optim.AdamW(
            trainable, lr=config.learning_rate, weight_decay=config.weight_decay
        )
    scaler, best_accuracy, updates = _scaler(policy_device, config.fp16), float("-inf"), 0

    for epoch in range(config.epochs):
        policy.train()
        reference.eval()
        optimizer.zero_grad(set_to_none=True)
        losses, started = [], time.time()
        for index, items in enumerate(loader):
            prompts = _dpo_prompt_batch(items, tokenizer, config.max_prompt_length)
            chosen, rejected = [r["chosen"] for r in items], [r["rejected"] for r in items]
            group_start = (index // config.gradient_accumulation_steps) * config.gradient_accumulation_steps
            group_size = min(config.gradient_accumulation_steps, len(loader) - group_start)
            with _amp(policy_device, config.fp16):
                loss = compute_dpo_loss(
                    policy, reference, tokenizer, prompts, chosen, rejected,
                    config.beta, config.max_length,
                )
            scaler.scale(loss / group_size).backward()
            losses.append(float(loss.detach().item()))

            if (index + 1) % config.gradient_accumulation_steps == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                updates += 1

        val = evaluate_dpo(policy, reference, tokenizer, validation_rows, config) if validation_rows else None
        row = {
            "track": "dpo", "epoch": epoch + 1, "step": updates,
            "train_loss": sum(losses) / max(1, len(losses)),
            "validation_loss": val["loss"] if val else None,
            "reward_accuracy": val["reward_accuracy"] if val else None,
            "reward_margin": val["reward_margin"] if val else None,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - started, 2),
        }
        _append_jsonl(metrics, row)
        if val and val["reward_accuracy"] > best_accuracy:
            best_accuracy = val["reward_accuracy"]
            _save(policy, tokenizer, out / "best", row)

    result = {
        "config": asdict(config),
        "best_validation_reward_accuracy": None if math.isinf(best_accuracy) else best_accuracy,
        "updates": updates,
    }
    _save(policy, tokenizer, out / "final", result)
    return result