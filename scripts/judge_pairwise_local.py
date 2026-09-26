from __future__ import annotations

import argparse
import json
from collections import Counter

import torch
from tqdm.auto import tqdm

from qwen_dpo_alignment.modeling import load_causal_lm, load_tokenizer
from qwen_dpo_alignment.utils import write_jsonl


JUDGE_TEMPLATE = """You are an impartial judge. Pick the better assistant response for the instruction.

Judge by helpfulness, correctness, clarity, and instruction following. Ignore response order. If both are similarly good, answer TIE.

Instruction:
{instruction}

Response A:
{response_a}

Response B:
{response_b}

Answer with exactly one token: A, B, or TIE.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge pairwise comparisons with a local instruct model.")
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--judge_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--no_fp16", action="store_true")
    return parser.parse_args()


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def parse_choice(text: str) -> str:
    normalized = text.strip().upper()
    if normalized.startswith("A"):
        return "A"
    if normalized.startswith("B"):
        return "B"
    if normalized.startswith("TIE"):
        return "TIE"
    return "UNKNOWN"


def main() -> None:
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_fp16 = (not args.no_fp16) and device.startswith("cuda")
    tokenizer = load_tokenizer(args.judge_model)
    model = load_causal_lm(args.judge_model, device=device, fp16=use_fp16)
    model.eval()

    rows = read_jsonl(args.pairs_path)
    judged = []
    counts = Counter()
    for row in tqdm(rows, desc="judge"):
        prompt = JUDGE_TEMPLATE.format(**row)
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0, encoded.input_ids.shape[1] :], skip_special_tokens=True)
        choice = parse_choice(text)
        winner = "tie"
        if choice in {"A", "B"}:
            winner = row["generator_a"] if choice == "A" else row["generator_b"]
        counts[winner] += 1
        judged.append({**row, "judge_raw": text.strip(), "judge_choice": choice, "winner": winner})

    write_jsonl(args.output_path, judged)
    total = max(1, len(judged))
    print(f"Wrote {len(judged)} judgments to {args.output_path}")
    for key, value in counts.most_common():
        print(f"{key}: {value} ({value / total:.3f})")


if __name__ == "__main__":
    main()

