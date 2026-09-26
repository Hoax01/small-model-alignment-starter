from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from qwen_dpo_alignment.modeling import Batch
from qwen_dpo_alignment.prompts import format_prompt


def load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode, encoding="utf-8") as fin:
        if path.suffix in {".jsonl", ".gz"}:
            return [json.loads(line) for line in fin if line.strip()]
        return json.load(fin)


def maybe_sample(rows: list[dict[str, Any]], max_examples: int | None, seed: int) -> list[dict[str, Any]]:
    if max_examples is None or len(rows) <= max_examples:
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), max_examples))
    return [rows[idx] for idx in indices]


def extract_sft_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    for prompt_key, response_key in (
        ("prompt", "response"),
        ("instruction", "output"),
        ("question", "answer"),
        ("input", "output"),
    ):
        if row.get(prompt_key) and row.get(response_key):
            return str(row[prompt_key]), str(row[response_key])

    messages = row.get("messages") or row.get("conversation") or row.get("conversations")
    if isinstance(messages, list):
        user_text = None
        for msg in messages:
            role = str(msg.get("role") or msg.get("from") or "").lower()
            text = msg.get("content") or msg.get("value")
            if text is None:
                continue
            if role in {"user", "human"} and user_text is None:
                user_text = str(text)
            elif role in {"assistant", "gpt", "bot"} and user_text is not None:
                return user_text, str(text)
    return None


def extract_hh_pair(row: dict[str, Any], source: str = "hh") -> dict[str, str] | None:
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not chosen or not rejected:
        return None
    chosen = str(chosen)
    rejected = str(rejected)
    marker = "\n\nAssistant:"
    marker_idx = chosen.find(marker)
    if marker_idx < 0:
        return None
    prompt_with_marker = chosen[: marker_idx + len(marker)]
    if not rejected.startswith(prompt_with_marker):
        return None
    if prompt_with_marker.count("\n\nHuman:") != 1:
        return None
    instruction = prompt_with_marker.split("\n\nHuman:", 1)[1].rsplit(marker, 1)[0].strip()
    chosen_response = chosen[len(prompt_with_marker) :].strip()
    rejected_response = rejected[len(prompt_with_marker) :].strip()
    if not instruction or not chosen_response or not rejected_response:
        return None
    return {
        "instruction": instruction,
        "chosen": chosen_response,
        "rejected": rejected_response,
        "source": source,
    }


def build_lm_example(
    tokenizer,
    instruction: str,
    response: str,
    max_length: int,
    max_prompt_length: int,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer(format_prompt(instruction), add_special_tokens=False).input_ids
    response_ids = tokenizer(response.strip() + tokenizer.eos_token, add_special_tokens=False).input_ids

    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[:max_prompt_length]
    if len(prompt_ids) + len(response_ids) > max_length:
        response_ids = response_ids[: max(1, max_length - len(prompt_ids))]

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class SFTDataset(Dataset):
    def __init__(
        self,
        rows: Iterable[dict[str, Any]],
        tokenizer,
        max_length: int,
        max_prompt_length: int,
    ) -> None:
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        for row in rows:
            pair = extract_sft_pair(row)
            if pair is not None:
                self.examples.append({"instruction": pair[0], "response": pair[1]})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        ex = self.examples[idx]
        return build_lm_example(
            self.tokenizer,
            ex["instruction"],
            ex["response"],
            self.max_length,
            self.max_prompt_length,
        )


class DPODataset(Dataset):
    def __init__(
        self,
        rows: Iterable[dict[str, Any]],
        tokenizer,
        max_length: int,
        max_prompt_length: int,
    ) -> None:
        self.examples = list(rows)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        chosen = build_lm_example(
            self.tokenizer,
            ex["instruction"],
            ex["chosen"],
            self.max_length,
            self.max_prompt_length,
        )
        rejected = build_lm_example(
            self.tokenizer,
            ex["instruction"],
            ex["rejected"],
            self.max_length,
            self.max_prompt_length,
        )
        return {
            "instruction": ex["instruction"],
            "chosen": chosen,
            "rejected": rejected,
            "source": ex.get("source", "unknown"),
        }


def collate_sft(batch: list[dict[str, list[int]]], pad_token_id: int) -> Batch:
    return _collate_lm_items(batch, pad_token_id)


def collate_dpo(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    return {
        "instruction": [ex["instruction"] for ex in batch],
        "source": [ex["source"] for ex in batch],
        "chosen": _collate_lm_items([ex["chosen"] for ex in batch], pad_token_id),
        "rejected": _collate_lm_items([ex["rejected"] for ex in batch], pad_token_id),
    }


def _collate_lm_items(batch: list[dict[str, list[int]]], pad_token_id: int) -> Batch:
    input_ids = pad_sequence(
        [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    attention_mask = pad_sequence(
        [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = pad_sequence(
        [torch.tensor(ex["labels"], dtype=torch.long) for ex in batch],
        batch_first=True,
        padding_value=-100,
    )
    return Batch(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

