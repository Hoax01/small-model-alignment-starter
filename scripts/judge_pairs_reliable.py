"""Resumable Kaggle model judging with per-case recovery for malformed batches."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

from alignment.evaluation import read_records, summarize_blind_judgments
from scripts.judge_pairs import _batch_prompt, _parse_batch


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def summarize(pairs: list[dict], output: Path, model: str, attempts_path: Path) -> dict:
    judgments = read_records(output) if output.exists() else []
    valid_ids = {
        str(row.get("id")) for row in judgments
        if str(row.get("winner", "")).lower() in {"a", "b", "tie"}
    }
    missing = {str(row["id"]) for row in pairs} - valid_ids
    summary = summarize_blind_judgments(pairs, judgments, model, failure_ids=missing)
    attempts_by_id: dict[str, int] = {}
    if attempts_path.exists():
        for item in read_records(attempts_path):
            for item_id in item["ids"]:
                attempts_by_id[item_id] = attempts_by_id.get(item_id, 0) + 1
    summary["judge_case_attempts"] = sum(attempts_by_id.values())
    summary["judge_case_retries"] = sum(max(0, count - 1) for count in attempts_by_id.values())
    summary["cases_retried"] = sum(count > 1 for count in attempts_by_id.values())
    summary["max_attempts_per_case"] = max(attempts_by_id.values(), default=0)
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--env-file", default="/tmp/pa2-benchmarks.env")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.max_retries < 1 or args.sleep_seconds < 5:
        raise ValueError("Batch size and retries must be positive; sleep must be at least 5 seconds")

    from dotenv import load_dotenv

    load_dotenv(args.env_file, override=True)
    import kaggle_benchmarks as kbench

    if args.model not in kbench.llms:
        raise RuntimeError(f"Judge {args.model!r} unavailable; choices: {sorted(kbench.llms)}")
    judge = kbench.llms[args.model]
    pairs = read_records(args.pairs)
    pair_ids = [str(pair["id"]) for pair in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("Pair IDs must be unique")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    failure_path = output.with_name(output.stem + "_failures.jsonl")
    attempts_path = output.with_name(output.stem + "_attempts.jsonl")
    existing = read_records(output) if output.exists() else []
    done = {
        str(row.get("id")) for row in existing
        if str(row.get("winner", "")).lower() in {"a", "b", "tie"}
    }
    pending = [pair for pair in pairs if str(pair["id"]) not in done]
    print(f"Judge {args.model}: {len(done)} saved, {len(pending)} pending.")
    last_request = None
    run_id = uuid.uuid4().hex[:8]

    def request(batch: list[dict], label: str):
        nonlocal last_request
        if last_request is not None:
            delay = args.sleep_seconds - (time.monotonic() - last_request)
            if delay > 0:
                time.sleep(delay)
        last_request = time.monotonic()
        ids = [str(pair["id"]) for pair in batch]
        append(attempts_path, {"ids": ids, "time": time.time()})
        with kbench.chats.new(f"pa2-{run_id}-{label}"):
            raw = str(judge.prompt(_batch_prompt(batch)))
        return _parse_batch(raw, set(ids))

    def save(parsed: dict[str, tuple[str, str]], attempt_count: int):
        for item_id, (winner, reason) in parsed.items():
            append(output, {
                "id": item_id, "winner": winner, "reason": reason,
                "judge_model": args.model, "attempts": attempt_count,
            })
            done.add(item_id)

    fatal_markers = ("401", "402", "403", "unauthorized", "forbidden", "quota exceeded", "invalid api key")
    try:
        for batch_index, start in enumerate(range(0, len(pending), args.batch_size), 1):
            batch = pending[start:start + args.batch_size]
            last_error = None
            parsed = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    parsed = request(batch, f"{batch_index}-batch-{attempt}")
                    save(parsed, attempt)
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    if any(marker in last_error.lower() for marker in fatal_markers):
                        raise RuntimeError(f"Judge access failed: {last_error}") from exc
            if parsed is None:
                append(failure_path, {
                    "ids": [str(pair["id"]) for pair in batch],
                    "phase": "batch", "error": last_error,
                    "attempts": args.max_retries,
                })
                # A malformed batch should not discard otherwise judgeable cases.
                for case_index, pair in enumerate(batch, 1):
                    item_id = str(pair["id"])
                    last_error = None
                    for attempt in range(1, args.max_retries + 1):
                        try:
                            parsed = request([pair], f"{batch_index}-case-{case_index}-{attempt}")
                            save(parsed, attempt)
                            break
                        except Exception as exc:
                            last_error = f"{type(exc).__name__}: {exc}"
                            if any(marker in last_error.lower() for marker in fatal_markers):
                                raise RuntimeError(f"Judge access failed: {last_error}") from exc
                    else:
                        append(failure_path, {
                            "ids": [item_id], "phase": "single",
                            "error": last_error, "attempts": args.max_retries,
                        })
            print(f"Finished batch {batch_index}: {len(done)}/{len(pairs)} valid judgments.")
    finally:
        summary = summarize(pairs, output, args.model, attempts_path)
    if summary["unknown"]:
        raise RuntimeError(f"{summary['unknown']} judgments unresolved; inspect {failure_path} and rerun")


if __name__ == "__main__":
    main()
