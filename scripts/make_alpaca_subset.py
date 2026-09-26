from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from huggingface_hub import hf_hub_download

from qwen_dpo_alignment.utils import ensure_dir, write_json


DATASET_REPO = "tatsu-lab/alpaca_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed AlpacaEval subset with optional GPT-style references.")
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="data/alpaca_eval")
    parser.add_argument(
        "--reference_generators",
        default="gpt4_turbo,gpt-4-turbo,gpt4,gpt-4,text_davinci_003",
        help="Comma-separated generator name fragments to prefer if using alpaca_eval_all_outputs.json.",
    )
    return parser.parse_args()


def download_json(filename: str):
    path = hf_hub_download(repo_id=DATASET_REPO, filename=filename, repo_type="dataset")
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def load_eval_set() -> list[dict]:
    """Load AlpacaEval prompts without executing the deprecated HF dataset script."""
    rows = download_json("alpaca_eval.json")
    normalized = []
    for idx, row in enumerate(rows):
        if not row.get("instruction"):
            continue
        normalized.append(
            {
                "instruction": row["instruction"],
                "dataset": row.get("dataset", "alpaca_eval"),
                "source_index": idx,
            }
        )
    return normalized


def load_reference_outputs(generator_fragments: list[str]) -> dict[str, dict]:
    """Load GPT-style references keyed by instruction.

    Prefer alpaca_eval_gpt4_baseline.json because it is smaller and already contains
    one GPT-4-family baseline per instruction. If that file is unavailable, fall back
    to alpaca_eval_all_outputs.json and select a matching generator name.
    """
    try:
        baseline = download_json("alpaca_eval_gpt4_baseline.json")
        return {
            row["instruction"]: {
                "instruction": row["instruction"],
                "output": row.get("output"),
                "generator": row.get("generator", "gpt4_baseline"),
            }
            for row in baseline
            if row.get("instruction") and row.get("output")
        }
    except Exception as exc:
        print(f"Could not download alpaca_eval_gpt4_baseline.json: {exc}")

    try:
        all_outputs = download_json("alpaca_eval_all_outputs.json")
    except Exception as exc:
        print(f"Could not download reference outputs: {exc}")
        return {}

    grouped = defaultdict(list)
    for row in all_outputs:
        if row.get("instruction") and row.get("output"):
            grouped[row["instruction"]].append(row)

    refs = {}
    for instruction, rows in grouped.items():
        chosen = None
        for fragment in generator_fragments:
            for row in rows:
                if fragment.lower() in str(row.get("generator", "")).lower():
                    chosen = row
                    break
            if chosen is not None:
                break
        if chosen is not None:
            refs[instruction] = chosen
    return refs


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    eval_set = load_eval_set()
    if args.size > len(eval_set):
        raise ValueError(f"Requested {args.size} examples, but AlpacaEval has {len(eval_set)}.")

    rng = random.Random(args.seed)
    subset = rng.sample(eval_set, args.size)
    refs = load_reference_outputs([x.strip() for x in args.reference_generators.split(",") if x.strip()])

    rows = []
    ref_rows = []
    for idx, row in enumerate(subset):
        instruction = row["instruction"]
        ref = refs.get(instruction)
        out = {
            "id": f"alpacaeval-{args.seed}-{idx:04d}",
            "instruction": instruction,
            "dataset": row.get("dataset", "alpaca_eval"),
            "reference_output": ref.get("output") if ref else None,
            "reference_generator": ref.get("generator") if ref else None,
        }
        rows.append(out)
        if ref is not None:
            ref_rows.append(
                {
                    "instruction": instruction,
                    "output": ref["output"],
                    "generator": ref.get("generator", "reference"),
                    "dataset": out["dataset"],
                }
            )

    subset_path = output_dir / f"alpacaeval_{args.size}_seed{args.seed}.json"
    ref_path = output_dir / f"alpacaeval_{args.size}_seed{args.seed}_references.json"
    write_json(subset_path, rows)
    write_json(ref_path, ref_rows)
    print(f"Wrote {len(rows)} examples to {subset_path}")
    print(f"Wrote {len(ref_rows)} reference outputs to {ref_path}")


if __name__ == "__main__":
    main()
