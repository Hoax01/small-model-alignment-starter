"""Input loading and normalization for single-turn alignment datasets."""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any


PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction: str) -> str:
    """Use one plain-text prompt template across SFT, DPO, and generation."""
    if not isinstance(instruction, str):
        raise TypeError("instruction must be text")
    if instruction.startswith("### Instruction:\n"):
        return instruction
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


def _content(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        content = value.get("content", value.get("text"))
        if isinstance(content, str):
            return content.strip()
        if isinstance(value.get("messages"), list):
            return _content(value["messages"])
    return None


def _role(message: dict) -> str:
    return str(message.get("role", message.get("from", ""))).lower()


def _message_content(message: dict) -> str | None:
    return _content(message.get("content", message.get("value", message.get("text"))))


def _single_turn(messages: Any) -> tuple[str, str] | None:
    if isinstance(messages, dict) and isinstance(messages.get("messages"), list):
        messages = messages["messages"]
    if not isinstance(messages, list):
        return None
    turns = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = _role(message)
        text = _message_content(message)
        if not text or role in {"system", "developer"}:
            continue
        if role in {"user", "human"}:
            turns.append(("user", text))
        elif role in {"assistant", "gpt", "bot"}:
            turns.append(("assistant", text))
    if len(turns) == 2 and turns[0][0] == "user" and turns[1][0] == "assistant":
        return turns[0][1], turns[1][1]
    return None


def _prompt_text(value: Any) -> str | None:
    if isinstance(value, dict) and isinstance(value.get("messages"), list):
        value = value["messages"]
    direct = _content(value)
    if direct is not None:
        return direct
    if not isinstance(value, list):
        return None
    users = [
        _message_content(message)
        for message in value
        if isinstance(message, dict) and _role(message) in {"user", "human"}
    ]
    assistants = [
        message
        for message in value
        if isinstance(message, dict) and _role(message) in {"assistant", "gpt", "bot"}
    ]
    users = [text for text in users if text]
    # A prompt may include system context and exactly one user message. Reject
    # multi-turn histories in this single-turn training path.
    return users[0] if len(users) == 1 and not assistants else None


def _response_text(value: Any) -> str | None:
    if isinstance(value, dict) and isinstance(value.get("messages"), list):
        value = value["messages"]
    direct = _content(value)
    if direct is not None:
        return direct
    if not isinstance(value, list):
        return None
    assistants = [
        _message_content(message)
        for message in value
        if isinstance(message, dict) and _role(message) in {"assistant", "gpt", "bot"}
    ]
    assistants = [text for text in assistants if text]
    return assistants[0] if len(assistants) == 1 else None


def _parse_hh_transcript(text: str) -> tuple[str, str] | None:
    """Parse only a single Human/Assistant exchange from HH-style text."""
    chunks = re.split(r"\n\s*\n(?=(?:Human|Assistant):)", text.strip())
    turns = []
    for chunk in chunks:
        match = re.match(r"^(Human|Assistant):\s*(.*)$", chunk.strip(), flags=re.DOTALL)
        if not match:
            return None
        role = "user" if match.group(1) == "Human" else "assistant"
        turns.append((role, match.group(2).strip()))
    if len(turns) == 2 and turns[0][0] == "user" and turns[1][0] == "assistant":
        return turns[0][1], turns[1][1]
    return None


def _first_text(row: dict, *keys: str) -> str | None:
    for key in keys:
        text = _content(row.get(key))
        if text:
            return text
    return None


def _first_prompt(row: dict, *keys: str) -> str | None:
    for key in keys:
        text = _prompt_text(row.get(key))
        if text:
            return text
    return None


def _first_response(row: dict, *keys: str) -> str | None:
    for key in keys:
        text = _response_text(row.get(key))
        if text:
            return text
    return None


def normalize_sft_row(row: dict) -> dict[str, str] | None:
    for key in ("messages", "conversations", "conversation"):
        pair = _single_turn(row.get(key))
        if pair:
            return {"prompt": pair[0], "response": pair[1]}

    instruction = _first_prompt(row, "instruction", "prompt", "question", "query")
    response = _first_response(row, "response", "output", "answer", "completion", "chosen")
    if instruction is None or response is None:
        return None
    extra_input = _first_text(row, "input", "context")
    if extra_input:
        instruction = f"{instruction}\n\n{extra_input}"
    return {"prompt": instruction, "response": response}


def normalize_preference_row(row: dict) -> dict[str, str] | None:
    prompt = _first_prompt(row, "prompt", "instruction", "question", "query")
    chosen_raw = row.get("chosen", row.get("chosen_response", row.get("accepted")))
    rejected_raw = row.get("rejected", row.get("rejected_response", row.get("dispreferred")))
    chosen = _response_text(chosen_raw)
    rejected = _response_text(rejected_raw)
    if not chosen or not rejected:
        return None

    # Some HF conversational preference sets put user/assistant messages in
    # each completion and omit a separate prompt column.
    chosen_messages = _single_turn(chosen_raw)
    rejected_messages = _single_turn(rejected_raw)
    if chosen_messages and rejected_messages:
        if chosen_messages[0] != rejected_messages[0]:
            return None
        if prompt is None:
            prompt = chosen_messages[0]
        chosen, rejected = chosen_messages[1], rejected_messages[1]

    chosen_pair = _parse_hh_transcript(chosen)
    rejected_pair = _parse_hh_transcript(rejected)
    if chosen_pair and rejected_pair:
        if chosen_pair[0] != rejected_pair[0]:
            return None
        if prompt is None:
            prompt = chosen_pair[0]
        chosen, rejected = chosen_pair[1], rejected_pair[1]

    if prompt is None:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def _read_local_records(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    if suffix != ".json":
        raise ValueError(f"Unsupported data file extension: {path.suffix}")

    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "examples", "records"):
            if isinstance(payload.get(key), list):
                return payload[key]
        if all(isinstance(value, dict) for value in payload.values()):
            return list(payload.values())
    raise ValueError(f"Expected a list of records in {path}")


def load_records(source: str, split: str | None = None, config: str | None = None) -> list[dict]:
    """Load local JSON/JSONL/CSV or a Hugging Face datasets repository."""
    path = Path(source).expanduser()
    if path.is_file():
        return _read_local_records(path)
    if path.is_dir():
        candidates = [path / "data.jsonl", path / "data.json", path / "train.jsonl"]
        for candidate in candidates:
            if candidate.is_file():
                return _read_local_records(candidate)
        raise FileNotFoundError(f"No data.json, data.jsonl, or train.jsonl found under {path}")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            f"'{source}' is not a local file. Install datasets or pass a local JSON/JSONL/CSV path."
        ) from exc

    dataset = load_dataset(source, config, split=split or "train")
    return [dict(row) for row in dataset]


def select_rows(rows: list[dict], limit: int | None, seed: int = 42) -> list[dict]:
    """Choose a reproducible subset without replacement."""
    if limit is None or limit <= 0 or len(rows) <= limit:
        return rows
    indices = random.Random(seed).sample(range(len(rows)), limit)
    return [rows[index] for index in indices]


def load_sft_rows(
    source: str,
    split: str | None = None,
    config: str | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> list[dict[str, str]]:
    rows = load_records(source, split=split, config=config)
    normalized = [item for row in rows if (item := normalize_sft_row(row))]
    if not normalized:
        raise ValueError(f"No single-turn instruction/response examples found in {source}")
    return select_rows(normalized, limit, seed)


def load_preference_rows(
    source: str,
    split: str | None = None,
    config: str | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> list[dict[str, str]]:
    rows = load_records(source, split=split, config=config)
    normalized = [item for row in rows if (item := normalize_preference_row(row))]
    if not normalized:
        raise ValueError(f"No single-turn prompt/chosen/rejected examples found in {source}")
    return select_rows(normalized, limit, seed)