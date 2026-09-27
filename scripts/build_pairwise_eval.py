from __future__ import annotations

import argparse
import random

from qwen_dpo_alignment.utils import read_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build randomized pairwise comparisons for local/manual judging.")
    parser.add_argument("--a_outputs", required=True)
    parser.add_argument("--b_outputs", default=None)
    parser.add_argument("--b_from_reference", action="store_true")
    parser.add_argument("--a_name", required=True)
    parser.add_argument("--b_name", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    a_rows = read_json(args.a_outputs)
    if args.b_from_reference:
        b_by_id = {
            row.get("id") or row["instruction"]: {
                "instruction": row["instruction"],
                "output": row.get("reference_output"),
                "generator": row.get("reference_generator", args.b_name),
            }
            for row in a_rows
            if row.get("reference_output")
        }
    else:
        if not args.b_outputs:
            raise ValueError("--b_outputs is required unless --b_from_reference is set.")
        b_by_id = {row.get("id") or row["instruction"]: row for row in read_json(args.b_outputs)}

    pairs = []
    for row in a_rows:
        key = row.get("id") or row["instruction"]
        other = b_by_id.get(key)
        if other is None or not other.get("output"):
            continue
        left_is_a = rng.random() < 0.5
        left = row if left_is_a else other
        right = other if left_is_a else row
        pair = {
            "id": key,
            "instruction": row["instruction"],
            "response_a": left["output"],
            "response_b": right["output"],
            "generator_a": args.a_name if left_is_a else args.b_name,
            "generator_b": args.b_name if left_is_a else args.a_name,
            "answer_key": "A" if left_is_a else "B",
            "a_outputs_generator": args.a_name,
            "b_outputs_generator": args.b_name,
            "reference_output": row.get("reference_output"),
            "reference_generator": row.get("reference_generator"),
        }
        for metadata_key in ("dataset", "category", "expected_behavior"):
            if metadata_key in row:
                pair[metadata_key] = row[metadata_key]
        pairs.append(pair)

    write_jsonl(args.output_path, pairs)
    print(f"Wrote {len(pairs)} pairwise comparisons to {args.output_path}")


if __name__ == "__main__":
    main()

