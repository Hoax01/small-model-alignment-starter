from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class ToyTokenizer:
    """Small tokenizer that mimics the subset of HF tokenizer behavior used by tests."""

    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-_+/\\n#'\"()[]{}")
        self.vocab = {self.pad_token: self.pad_token_id, self.eos_token: self.eos_token_id}
        for ch in chars:
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = False):
        ids = []
        i = 0
        while i < len(text):
            if text.startswith(self.eos_token, i):
                ids.append(self.eos_token_id)
                i += len(self.eos_token)
                continue
            if text.startswith(self.pad_token, i):
                ids.append(self.pad_token_id)
                i += len(self.pad_token)
                continue
            ch = text[i]
            ids.append(self.vocab.setdefault(ch, len(self.vocab)))
            i += 1
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids, skip_special_tokens: bool = False):
        chars = []
        for idx in ids:
            idx = int(idx)
            if skip_special_tokens and idx in {self.pad_token_id, self.eos_token_id}:
                continue
            chars.append(self.inv_vocab.get(idx, "?"))
        return "".join(chars)

    def __call__(
        self,
        text,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
    ):
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        encoded = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
        if truncation and max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]
        attention = [[1] * len(ids) for ids in encoded]
        if padding:
            pad_to = max(len(ids) for ids in encoded) if max_length is None or padding != "max_length" else max_length
            encoded = [ids + [self.pad_token_id] * (pad_to - len(ids)) for ids in encoded]
            attention = [mask + [0] * (pad_to - len(mask)) for mask in attention]
        if return_tensors == "pt":
            return SimpleNamespace(
                input_ids=torch.tensor(encoded, dtype=torch.long),
                attention_mask=torch.tensor(attention, dtype=torch.long),
            )
        if is_batch:
            return {"input_ids": encoded, "attention_mask": attention}
        return SimpleNamespace(input_ids=encoded[0], attention_mask=attention[0])


class ToyCausalLM(nn.Module):
    """Tiny causal LM with trainable per-token logits.

    Logits are independent of position/input, which is enough to test sequence
    log-probability math and DPO gradients deterministically.
    """

    def __init__(self, vocab_size: int, bias_scale: float = 0.0):
        super().__init__()
        base = torch.linspace(-0.3, 0.3, vocab_size) * bias_scale
        self.logit_bias = nn.Parameter(base.clone())

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch, seq_len = input_ids.shape
        logits = self.logit_bias.view(1, 1, -1).expand(batch, seq_len, -1).contiguous()
        return SimpleNamespace(logits=logits)
