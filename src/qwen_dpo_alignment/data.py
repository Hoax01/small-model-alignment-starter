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



def message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        value = message.get("content") or message.get("value") or message.get("text") or ""
        return str(value)
    return str(message or "")


def messages_to_prompt_response(messages: Any) -> tuple[str, str] | None:
    if isinstance(messages, str):
        return None
    if not isinstance(messages, list):
        return None
    user_parts = []
    assistant_parts = []
    seen_assistant = False
    for msg in messages:
        role = ""
        if isinstance(msg, dict):
            role = str(msg.get("role") or msg.get("from") or "").lower()
        text = message_text(msg).strip()
        if not text:
            continue
        if role in {"assistant", "gpt", "bot"}:
            seen_assistant = True
            assistant_parts.append(text)
        elif not seen_assistant and role in {"user", "human"}:
            user_parts.append(text)
        elif not seen_assistant and not role:
            user_parts.append(text)
    prompt = "\n\n".join(user_parts).strip()
    response = "\n\n".join(assistant_parts).strip()
    if prompt and response:
        return prompt, response
    return None


def response_from_messages(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, list):
        assistant_parts = []
        fallback_parts = []
        for msg in value:
            text = message_text(msg).strip()
            if not text:
                continue
            fallback_parts.append(text)
            role = ""
            if isinstance(msg, dict):
                role = str(msg.get("role") or msg.get("from") or "").lower()
            if role in {"assistant", "gpt", "bot"}:
                assistant_parts.append(text)
        response = "\n\n".join(assistant_parts or fallback_parts).strip()
        return response or None
    return None


def prompt_from_value(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    pair = messages_to_prompt_response(value)
    if pair is not None:
        return pair[0]
    return response_from_messages(value)


def extract_ultrafeedback_pair(row: dict[str, Any], source: str = "ultrafeedback") -> dict[str, str] | None:
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not chosen or not rejected:
        return None

    instruction = None
    for key in ("prompt", "instruction", "input", "question"):
        if row.get(key):
            instruction = prompt_from_value(row[key])
            if instruction:
                break

    if instruction is None:
        chosen_pair = messages_to_prompt_response(chosen)
        rejected_pair = messages_to_prompt_response(rejected)
        if chosen_pair is not None and rejected_pair is not None and chosen_pair[0] == rejected_pair[0]:
            instruction = chosen_pair[0]
            chosen_response = chosen_pair[1]
            rejected_response = rejected_pair[1]
        else:
            return None
    else:
        chosen_response = response_from_messages(chosen)
        rejected_response = response_from_messages(rejected)

    if not instruction or not chosen_response or not rejected_response:
        return None
    return {
        "instruction": instruction.strip(),
        "chosen": chosen_response.strip(),
        "rejected": rejected_response.strip(),
        "source": source,
    }


def extract_dpo_pair(row: dict[str, Any], source: str = "unknown") -> dict[str, str] | None:
    if {"instruction", "chosen", "rejected"}.issubset(row):
        instruction = str(row["instruction"]).strip()
        chosen = response_from_messages(row["chosen"])
        rejected = response_from_messages(row["rejected"])
        if instruction and chosen and rejected:
            return {"instruction": instruction, "chosen": chosen, "rejected": rejected, "source": source}
    if "ultrafeedback" in source.lower():
        return extract_ultrafeedback_pair(row, source=source)
    return extract_hh_pair(row, source=source) or extract_ultrafeedback_pair(row, source=source)


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
    prompt_format: str = "plain",
) -> dict[str, list[int]]:
    if prompt_format == "chat_template":
        return build_chat_template_example(tokenizer, instruction, response, max_length)
    if prompt_format != "plain":
        raise ValueError(f"Unsupported prompt_format: {prompt_format}")

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


def build_chat_template_example(
    tokenizer,
    instruction: str,
    response: str,
    max_length: int,
) -> dict[str, list[int]]:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("prompt_format='chat_template' requires a tokenizer with a chat_template.")

    prompt_messages = [{"role": "user", "content": instruction.strip()}]
    full_messages = [*prompt_messages, {"role": "assistant", "content": response.strip()}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    input_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()

    input_ids = list(input_ids)[:max_length]
    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class SFTDataset(Dataset):
    def __init__(
        self,
        rows: Iterable[dict[str, Any]],
        tokenizer,
        max_length: int,
        max_prompt_length: int,
        prompt_format: str = "plain",
    ) -> None:
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.prompt_format = prompt_format
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
            self.prompt_format,
        )


class DPODataset(Dataset):
    def __init__(
        self,
        rows: Iterable[dict[str, Any]],
        tokenizer,
        max_length: int,
        max_prompt_length: int,
        prompt_format: str = "plain",
    ) -> None:
        self.examples = list(rows)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.prompt_format = prompt_format

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
            self.prompt_format,
        )
        rejected = build_lm_example(
            self.tokenizer,
            ex["instruction"],
            ex["rejected"],
            self.max_length,
            self.max_prompt_length,
            self.prompt_format,
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

