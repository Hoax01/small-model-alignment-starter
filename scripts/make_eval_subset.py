"""Create a fixed, seeded instruction subset for fair generation comparisons."""

from __future__ import annotations

import argparse

from alignment.evaluation import read_prompt_records, sample_prompts, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Local JSON/JSONL/CSV or HF dataset ID")
    parser.add_argument("--split", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="eval_sets/alpaca_eval_300_seed42.json")
    args = parser.parse_args()

    prompts = read_prompt_records(args.source, split=args.split, config=args.config)
    chosen = sample_prompts(prompts, args.count, args.seed)
    write_json(args.output, chosen)
    print(f"Wrote {len(chosen)} fixed prompts to {args.output} (seed={args.seed}).")


if __name__ == "__main__":
    main()