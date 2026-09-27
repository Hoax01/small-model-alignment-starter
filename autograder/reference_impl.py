from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def encode_prompt_response(tokenizer, prompt: str, response: str, max_length: int):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_text = response + (tokenizer.eos_token or "")
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    if len(prompt_ids) >= max_length:
        prompt_ids = prompt_ids[: max_length - 1]
    available = max_length - len(prompt_ids)
    response_ids = response_ids[:available]
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask, labels


def build_reference_batch(tokenizer, prompts, responses, max_length):
    examples = [encode_prompt_response(tokenizer, p, r, max_length) for p, r in zip(prompts, responses)]
    input_ids = pad_sequence(
        [torch.tensor(x[0], dtype=torch.long) for x in examples],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_mask = pad_sequence(
        [torch.tensor(x[1], dtype=torch.long) for x in examples],
        batch_first=True,
        padding_value=0,
    )
    labels = pad_sequence(
        [torch.tensor(x[2], dtype=torch.long) for x in examples],
        batch_first=True,
        padding_value=-100,
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def response_logps_from_batch(model, batch):
    logits = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask")).logits.float()
    labels = batch["labels"][:, 1:].clone()
    logits = logits[:, :-1, :]
    mask = labels.ne(-100)
    labels[~mask] = 0
    token_logps = torch.gather(F.log_softmax(logits, dim=-1), 2, labels.unsqueeze(2)).squeeze(2)
    return (token_logps * mask).sum(dim=1)


def reference_response_logps(model, tokenizer, prompts, responses, max_length):
    return response_logps_from_batch(model, build_reference_batch(tokenizer, prompts, responses, max_length))


def reference_sft_loss(model, tokenizer, prompts, responses, max_length):
    return -reference_response_logps(model, tokenizer, prompts, responses, max_length).mean()


def reference_dpo_loss(policy, reference, tokenizer, prompts, chosen, rejected, beta, max_length):
    pi_chosen = reference_response_logps(policy, tokenizer, prompts, chosen, max_length)
    pi_rejected = reference_response_logps(policy, tokenizer, prompts, rejected, max_length)
    with torch.no_grad():
        ref_chosen = reference_response_logps(reference, tokenizer, prompts, chosen, max_length)
        ref_rejected = reference_response_logps(reference, tokenizer, prompts, rejected, max_length)
    logits = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)
    return -F.logsigmoid(beta * logits).mean()
