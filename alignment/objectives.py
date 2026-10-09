"""Response-only supervised and direct-preference objectives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


def _token_ids(tokenizer, text: str) -> list[int]:
    if not isinstance(text, str):
        raise TypeError(f"Expected text to be str, got {type(text).__name__}")

    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids

    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("Expected one tokenized text, received multiple rows")
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _eos_ids(tokenizer) -> list[int]:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return [int(eos_id)]
    eos_text = getattr(tokenizer, "eos_token", None)
    return _token_ids(tokenizer, eos_text) if eos_text else []


def _validate_rows(*columns: Sequence[str]) -> int:
    lengths = [len(column) for column in columns]
    if len(set(lengths)) > 1:
        raise ValueError(f"Input columns must have the same length; got {lengths}")
    return lengths[0] if lengths else 0


def _build_lm_batch(tokenizer, prompts, responses, max_length: int) -> dict[str, torch.Tensor]:
    if max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    _validate_rows(prompts, responses)

    eos_ids = _eos_ids(tokenizer)
    sequences: list[list[int]] = []
    labels: list[list[int]] = []

    for prompt, response in zip(prompts, responses):
        prompt_ids = _token_ids(tokenizer, prompt)
        # An empty prompt still needs a preceding token to score its first
        # response token in a causal LM. Treat BOS as masked prompt context.
        if not prompt_ids:
            bos_id = getattr(tokenizer, "bos_token_id", None)
            if bos_id is None:
                bos_id = getattr(tokenizer, "pad_token_id", None)
            if bos_id is None:
                bos_id = getattr(tokenizer, "eos_token_id", None)
            prompt_ids = [int(bos_id) if bos_id is not None else 0]

        response_ids = _token_ids(tokenizer, response)
        if eos_ids and (not response_ids or response_ids[-len(eos_ids):] != eos_ids):
            response_ids.extend(eos_ids)

        full_ids = (prompt_ids + response_ids)[:max_length]
        retained_prompt_length = min(len(prompt_ids), len(full_ids))
        row_labels = [-100] * retained_prompt_length
        row_labels.extend(full_ids[retained_prompt_length:])
        sequences.append(full_ids)
        labels.append(row_labels)

    if not sequences:
        empty = torch.empty((0, 0), dtype=torch.long)
        return {"input_ids": empty, "attention_mask": empty.clone(), "labels": empty.clone()}

    width = max(len(sequence) for sequence in sequences)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0

    input_rows, mask_rows, label_rows = [], [], []
    for sequence, row_labels in zip(sequences, labels):
        padding = width - len(sequence)
        input_rows.append(sequence + [int(pad_id)] * padding)
        mask_rows.append([1] * len(sequence) + [0] * padding)
        label_rows.append(row_labels + [-100] * padding)

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
    }


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Build a right-padded batch with loss labels only on response tokens."""
    return _build_lm_batch(tokenizer, prompts, responses, max_length)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Build separate response-masked language-model batches for both sides."""
    _validate_rows(prompts, chosen_responses, rejected_responses)
    return {
        "chosen": _build_lm_batch(tokenizer, prompts, chosen_responses, max_length),
        "rejected": _build_lm_batch(tokenizer, prompts, rejected_responses, max_length),
    }


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = getattr(model, "device", None)
        return torch.device(device) if device is not None else torch.device("cpu")


def _model_logits(model, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    device = _model_device(model)
    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, Mapping) and "logits" in outputs:
        return outputs["logits"]
    return outputs[0]


def _response_logprobs_from_batch(model, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Sum causal next-token log probabilities over response tokens only."""
    logits = _model_logits(model, batch)
    labels = batch["labels"].to(logits.device)
    attention_mask = batch["attention_mask"].to(logits.device)

    if logits.shape[1] < 2:
        return logits.float().sum(dim=(1, 2)) * 0.0

    next_token_logits = logits[:, :-1, :].float()
    next_token_labels = labels[:, 1:]
    valid = next_token_labels.ne(-100) & attention_mask[:, 1:].bool()
    safe_labels = next_token_labels.masked_fill(~valid, 0)
    token_logprobs = F.log_softmax(next_token_logits, dim=-1).gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return token_logprobs.masked_fill(~valid, 0.0).sum(dim=-1)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log probability for each input row."""
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    return _response_logprobs_from_batch(model, batch)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return mean response-token negative log likelihood (not prompt tokens)."""
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    logprobs = _response_logprobs_from_batch(model, batch)
    response_tokens = batch["labels"][:, 1:].ne(-100).sum().to(logprobs.device)
    return -logprobs.sum() / response_tokens.clamp_min(1)


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the mean DPO objective for per-example sequence logps."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    shapes = {
        tuple(value.shape)
        for value in (
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
    }
    if len(shapes) != 1:
        raise ValueError(f"All log-probability tensors must have matching shapes; got {shapes}")

    policy_gap = policy_chosen_logps - policy_rejected_logps
    reference_gap = reference_chosen_logps - reference_rejected_logps
    return -F.logsigmoid(float(beta) * (policy_gap - reference_gap)).mean()


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Compute DPO loss; gradients flow through the policy only."""
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    policy_chosen_logps = _response_logprobs_from_batch(policy_model, batch["chosen"])
    policy_rejected_logps = _response_logprobs_from_batch(policy_model, batch["rejected"])
    with torch.no_grad():
        reference_chosen_logps = _response_logprobs_from_batch(reference_model, batch["chosen"])
        reference_rejected_logps = _response_logprobs_from_batch(reference_model, batch["rejected"])

    policy_device = policy_chosen_logps.device
    return compute_dpo_loss_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps.to(policy_device),
        reference_rejected_logps.to(policy_device),
        beta,
    )