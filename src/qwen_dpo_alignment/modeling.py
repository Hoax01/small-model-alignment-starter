from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
        )


def load_tokenizer(model_name_or_path: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _adapter_base_model_name(model_name_or_path: str) -> str | None:
    adapter_config_path = Path(model_name_or_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        return None
    with adapter_config_path.open("r", encoding="utf-8") as fin:
        adapter_config = json.load(fin)
    return adapter_config.get("base_model_name_or_path")


def load_causal_lm(model_name_or_path: str, *, device: str, fp16: bool = True):
    dtype = torch.float16 if fp16 and device.startswith("cuda") else torch.float32
    adapter_base_model = _adapter_base_model_name(model_name_or_path)
    load_path = adapter_base_model or model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if adapter_base_model is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Loading LoRA adapter checkpoints requires peft. Install requirements.txt or `pip install peft`.") from exc
        model = PeftModel.from_pretrained(model, model_name_or_path)
    model.to(device)
    return model


def amp_context(device: str, enabled: bool):
    if enabled and str(device).startswith("cuda"):
        return torch.amp.autocast("cuda")
    return nullcontext()


def make_grad_scaler(device: str, enabled: bool):
    if enabled and str(device).startswith("cuda"):
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.amp.GradScaler("cpu", enabled=False)


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, *, chunk_size: int = 64) -> torch.Tensor:
    """Memory-conscious causal LM loss that avoids casting all logits to fp32 at once."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    token_count = shift_labels.ne(-100).sum().clamp_min(1)
    losses = []
    for start in range(0, shift_logits.size(1), chunk_size):
        end = min(start + chunk_size, shift_logits.size(1))
        chunk_logits = shift_logits[:, start:end, :].float()
        chunk_labels = shift_labels[:, start:end]
        losses.append(
            F.cross_entropy(
                chunk_logits.reshape(-1, chunk_logits.size(-1)),
                chunk_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        )
    return torch.stack(losses).sum() / token_count


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor, *, chunk_size: int = 64) -> torch.Tensor:
    """Return summed log p(labels | prefix), ignoring labels equal to -100.

    The logits are usually fp16 under autocast. We cast chunks to fp32 before
    log_softmax because DPO is sensitive to small log-probability differences.
    Chunking avoids materializing the full [batch, seq, vocab] tensor in fp32.
    """
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    mask = labels.ne(-100)
    labels[~mask] = 0
    total_logps = torch.zeros(logits.size(0), device=logits.device, dtype=torch.float32)
    for start in range(0, logits.size(1), chunk_size):
        end = min(start + chunk_size, logits.size(1))
        chunk_logits = logits[:, start:end, :].float()
        chunk_labels = labels[:, start:end]
        chunk_mask = mask[:, start:end]
        token_logps = torch.gather(F.log_softmax(chunk_logits, dim=-1), 2, chunk_labels.unsqueeze(2)).squeeze(2)
        total_logps = total_logps + (token_logps * chunk_mask).sum(dim=1)
    return total_logps


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    metrics = {
        "reward_accuracy": (chosen_rewards > rejected_rewards).float().mean(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean(),
        "policy_logratio": pi_logratios.detach().mean(),
        "ref_logratio": ref_logratios.detach().mean(),
    }
    return losses.mean(), metrics

