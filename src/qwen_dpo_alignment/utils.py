from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def write_json(path: str | os.PathLike[str], data: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)


def write_jsonl(path: str | os.PathLike[str], rows: list[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | os.PathLike[str], row: dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def cuda_summary() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable; running on CPU."
    lines = [f"CUDA devices: {torch.cuda.device_count()}"]
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        total_gib = props.total_memory / 1024**3
        lines.append(f"  cuda:{idx}: {props.name}, {total_gib:.1f} GiB")
    return "\n".join(lines)

