from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

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


def load_causal_lm(model_name_or_path: str, *, device: str, fp16: bool = True):
    dtype = torch.float16 if fp16 and device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
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


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return summed log p(labels | prefix), ignoring labels equal to -100.

    The logits are usually fp16 under autocast. We cast to fp32 before log_softmax
    because DPO is very sensitive to small log-probability differences.
    """
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :].float()
    mask = labels.ne(-100)
    labels[~mask] = 0
    token_logps = torch.gather(F.log_softmax(logits, dim=-1), 2, labels.unsqueeze(2)).squeeze(2)
    return (token_logps * mask).sum(dim=1)


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

