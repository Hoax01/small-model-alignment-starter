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
    parser.add_argument("--device", default=None, help="Generation device, e.g. cuda:0, cuda:1, or cpu. Defaults to cuda:0 when available.")
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N selected prompts. Useful for sanity checks.")
    parser.add_argument("--num_shards", type=int, default=1, help="Split the selected prompts into this many interleaved shards.")
    parser.add_argument("--shard_index", type=int, default=0, help="Generate only this shard index in [0, num_shards).")
    parser.add_argument("--no_fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(cuda_summary())
    if args.num_shards < 1:
        raise ValueError("--num_shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    use_fp16 = (not args.no_fp16) and device.startswith("cuda")
    tokenizer = load_tokenizer(args.model_name_or_path)
    tokenizer.padding_side = "left"
    model = load_causal_lm(args.model_name_or_path, device=device, fp16=use_fp16)
    model.eval()

    rows = read_json(args.subset_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.num_shards > 1:
        rows = rows[args.shard_index :: args.num_shards]
        print(f"Generating shard {args.shard_index}/{args.num_shards} with {len(rows)} prompts on {device}")
    else:
        print(f"Generating {len(rows)} prompts on {device}")
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
            item = {
                "id": row.get("id"),
                "instruction": row["instruction"],
                "output": output.strip(),
                "generator": args.generator_name,
                "dataset": row.get("dataset", "alpaca_eval"),
                "reference_output": row.get("reference_output"),
                "reference_generator": row.get("reference_generator"),
            }
            for metadata_key in ("category", "expected_behavior"):
                if metadata_key in row:
                    item[metadata_key] = row[metadata_key]
            outputs.append(item)
    elapsed = time.time() - start
    write_json(args.output_path, outputs)
    print(f"Wrote {len(outputs)} generations to {args.output_path}")
    print(f"Throughput: {len(outputs) / max(elapsed, 1e-9):.3f} examples/sec")


if __name__ == "__main__":
    main()

