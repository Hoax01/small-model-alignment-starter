from __future__ import annotations

import argparse
import time

import torch
from tqdm.auto import tqdm

from qwen_dpo_alignment.modeling import amp_context, load_causal_lm, load_tokenizer
from qwen_dpo_alignment.prompts import format_prompt
from qwen_dpo_alignment.utils import cuda_summary, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AlpacaEval-subset responses.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--subset_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--generator_name", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--no_fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(cuda_summary())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_fp16 = (not args.no_fp16) and device.startswith("cuda")
    tokenizer = load_tokenizer(args.model_name_or_path)
    tokenizer.padding_side = "left"
    model = load_causal_lm(args.model_name_or_path, device=device, fp16=use_fp16)
    model.eval()

    rows = read_json(args.subset_path)
    outputs = []
    start = time.time()
    for start_idx in tqdm(range(0, len(rows), args.batch_size), desc="generate"):
        batch_rows = rows[start_idx : start_idx + args.batch_size]
        prompts = [format_prompt(row["instruction"]) for row in batch_rows]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p
        with torch.no_grad(), amp_context(device, use_fp16):
            generated = model.generate(**encoded, **generation_kwargs)
        prompt_len = encoded.input_ids.shape[1]
        decoded = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        for row, output in zip(batch_rows, decoded):
            outputs.append(
                {
                    "id": row.get("id"),
                    "instruction": row["instruction"],
                    "output": output.strip(),
                    "generator": args.generator_name,
                    "dataset": row.get("dataset", "alpaca_eval"),
                    "reference_output": row.get("reference_output"),
                    "reference_generator": row.get("reference_generator"),
                }
            )
    elapsed = time.time() - start
    write_json(args.output_path, outputs)
    print(f"Wrote {len(outputs)} generations to {args.output_path}")
    print(f"Throughput: {len(outputs) / max(elapsed, 1e-9):.3f} examples/sec")


if __name__ == "__main__":
    main()

