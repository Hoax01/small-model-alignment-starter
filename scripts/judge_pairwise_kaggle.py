from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge pairwise JSONL files with Kaggle model-proxy models.")
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--judge_model", default="google/gemini-3.6-flash")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--reuse_judgments", action="store_true")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Skipping malformed JSONL record {path}:{line_number}: {exc}")
    return rows


def append_jsonl(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def normalize_choice(value: str | None) -> str:
    if not value:
        return "UNKNOWN"
    value = value.strip().upper()
    if value in {"A", "RESPONSE A", "MODEL A"}:
        return "A"
    if value in {"B", "RESPONSE B", "MODEL B"}:
        return "B"
    if value in {"TIE", "DRAW", "EQUAL"}:
        return "TIE"
    return "UNKNOWN"


def parse_judge_output(raw_text: str, expected_items: int) -> list[dict]:
    pattern = re.compile(r"(?ms)^ITEM\s+(\d+)\s*$\n(.*?)(?=^ITEM\s+\d+\s*$|\Z)")
    blocks = pattern.findall(raw_text.strip())
    parsed = {}
    for item_number, block in blocks:
        choice_match = re.search(r"(?im)^CHOICE:\s*(A|B|TIE)\s*$", block)
        reason_match = re.search(r"(?im)^REASON:\s*(.+?)\s*$", block)
        parsed[int(item_number)] = {
            "judge_choice": normalize_choice(choice_match.group(1) if choice_match else None),
            "judge_reason": reason_match.group(1).strip() if reason_match else "",
            "raw_block": block.strip(),
            "parse_error": None if choice_match else "missing_choice",
        }
    return [
        parsed.get(
            idx + 1,
            {
                "judge_choice": "UNKNOWN",
                "judge_reason": "",
                "raw_block": "",
                "parse_error": "missing_item",
            },
        )
        for idx in range(expected_items)
    ]


def pair_key(row: dict) -> str:
    return str(row.get("id") or row.get("instruction"))


def compatible_cached(row: dict, pair: dict, judge_model: str) -> bool:
    return (
        row.get("id") == pair.get("id")
        and row.get("instruction") == pair.get("instruction")
        and row.get("response_a") == pair.get("response_a")
        and row.get("response_b") == pair.get("response_b")
        and row.get("generator_a") == pair.get("generator_a")
        and row.get("generator_b") == pair.get("generator_b")
        and row.get("judge_model") == judge_model
        and row.get("judge_choice") in {"A", "B", "TIE", "UNKNOWN"}
    )


def build_batch_prompt(rows: list[dict]) -> str:
    formatted_items = []
    for idx, row in enumerate(rows, start=1):
        reference = row.get("reference_output")
        reference_block = ""
        if reference:
            reference_block = f"""
REFERENCE RESPONSE ({row.get('reference_generator') or 'reference'}):
{reference}
"""
        formatted_items.append(
            f"""
ITEM {idx}
PAIR ID: {row.get('id')}

INSTRUCTION:
{row['instruction']}
{reference_block}
RESPONSE A ({row.get('generator_a', 'A')}):
{row['response_a']}

RESPONSE B ({row.get('generator_b', 'B')}):
{row['response_b']}
""".strip()
        )

    batch_input = "\n\n".join(formatted_items)
    return textwrap.dedent(
        f"""
        You are an impartial pairwise evaluator for instruction-following assistant responses.

        For each item, compare Response A and Response B using the instruction. If a reference response is provided, use it only as helpful context, not as a mandatory exact answer.

        Prefer the response that is more helpful, correct, complete, concise when appropriate, well-formatted, and faithful to the instruction. Penalize hallucinations, contradictions, unsafe content, and failure to follow requested format. Use TIE if both responses are similarly good or similarly bad.

        OUTPUT FORMAT
        For each item, output exactly:
        ITEM <number>
        CHOICE: A|B|TIE
        REASON: <one short sentence>

        Do not output markdown, bullet points, JSON, or extra commentary.

        INPUTS:
        {batch_input}
        """
    ).strip()


def extract_usage(run) -> dict:
    input_tokens = 0
    output_tokens = 0
    if getattr(run, "chat", None) is not None:
        for message in run.chat.messages:
            meta = getattr(message, "_meta", {}) or {}
            input_tokens += meta.get("input_tokens", 0) or 0
            output_tokens += meta.get("output_tokens", 0) or 0
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def main() -> None:
    args = parse_args()
    try:
        import kaggle_benchmarks as kbench
    except ImportError as exc:
        raise ImportError(
            "This script must run in a Kaggle environment with kaggle_benchmarks and model-proxy access."
        ) from exc

    pairs = read_jsonl(args.pairs_path)
    if not pairs:
        raise ValueError(f"No pairs found in {args.pairs_path}")

    os.environ["LLM_DEFAULT"] = args.judge_model
    llm_holder = {"llm": None}
    active_results: list[dict] = []

    def get_llm():
        if llm_holder["llm"] is None:
            print(f"Loading Kaggle judge model: {args.judge_model}")
            llm_holder["llm"] = kbench.kaggle.load_model(args.judge_model)
        return llm_holder["llm"]

    @kbench.task(name="qwen_dpo_pairwise_judge_batch")
    def judge_batch(llm, batch_rows):
        raw_text = str(llm.prompt(build_batch_prompt(batch_rows), temperature=0))
        parsed_rows = parse_judge_output(raw_text, len(batch_rows))
        for pair, parsed in zip(batch_rows, parsed_rows):
            choice = parsed["judge_choice"]
            winner = "tie"
            if choice == "A":
                winner = pair.get("generator_a", "A")
            elif choice == "B":
                winner = pair.get("generator_b", "B")
            active_results.append(
                {
                    **pair,
                    "judge_model": args.judge_model,
                    "judge_choice": choice,
                    "winner": winner,
                    "judge_reason": parsed["judge_reason"],
                    "parse_error": parsed["parse_error"],
                    "raw_block": parsed["raw_block"],
                    "raw_text": raw_text,
                }
            )

    cached_by_key = {}
    output_path = Path(args.output_path)
    if args.reuse_judgments and output_path.is_file():
        for row in read_jsonl(output_path):
            cached_by_key[pair_key(row)] = row

    accepted_cached = []
    missing = []
    pair_by_key = {pair_key(row): row for row in pairs}
    for row in pairs:
        cached = cached_by_key.get(pair_key(row))
        if cached and compatible_cached(cached, row, args.judge_model):
            accepted_cached.append(cached)
        else:
            missing.append(row)

    write_jsonl(output_path, accepted_cached)
    print(f"Loaded {len(accepted_cached)} cached judgments; judging {len(missing)} missing pairs.")

    usage = Counter()
    for start in tqdm(range(0, len(missing), args.batch_size), desc="judging", unit="batch"):
        batch = missing[start : start + args.batch_size]
        active_results.clear()
        run = judge_batch.run(get_llm(), batch)
        usage.update(extract_usage(run))
        batch_rows = list(active_results)
        returned = {pair_key(row) for row in batch_rows}
        expected = {pair_key(row) for row in batch}
        if returned != expected:
            raise RuntimeError(f"Judge returned keys {sorted(returned)}; expected {sorted(expected)}")
        append_jsonl(output_path, batch_rows)

    final_rows = read_jsonl(output_path)
    final_by_key = {pair_key(row): row for row in final_rows if pair_key(row) in pair_by_key}
    ordered = [final_by_key[pair_key(row)] for row in pairs if pair_key(row) in final_by_key]
    write_jsonl(output_path, ordered)

    counts = Counter(row.get("winner", "unknown") for row in ordered)
    parse_errors = sum(1 for row in ordered if row.get("parse_error"))
    print(f"Wrote {len(ordered)} judgments to {output_path}")
    for key, value in counts.most_common():
        print(f"{key}: {value} ({value / max(1, len(ordered)):.3f})")
    print(f"Parse errors: {parse_errors}")
    print(f"Judge input tokens this run: {usage['input_tokens']:,}")
    print(f"Judge output tokens this run: {usage['output_tokens']:,}")


if __name__ == "__main__":
    main()
