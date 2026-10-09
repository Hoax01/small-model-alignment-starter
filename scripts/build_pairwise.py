"""Create response-order-randomized DPO-vs-SFT pairwise records."""

from __future__ import annotations

import argparse
from pathlib import Path

from alignment.evaluation import append_jsonl, build_blind_pairs, read_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", required=True, help="SFT generation JSON/JSONL")
    parser.add_argument("--dpo", required=True, help="DPO generation JSON/JSONL")
    parser.add_argument("--sft-name", default="sft")
    parser.add_argument("--dpo-name", default="dpo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pairs = build_blind_pairs(
        read_records(args.sft), read_records(args.dpo),
        args.sft_name, args.dpo_name, args.seed,
    )
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(f"{path} already exists; choose a new pairwise output path.")
    for pair in pairs:
        append_jsonl(path, pair)
    print(f"Wrote {len(pairs)} blinded comparisons to {path}")


if __name__ == "__main__":
    main()