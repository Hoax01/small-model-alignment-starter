"""Train the assignment SFT model from JSON/JSONL/CSV or a HF dataset."""

from __future__ import annotations

import argparse
import json

import torch

from alignment.data import load_sft_rows
from alignment.models import load_causal_lm, load_tokenizer
from alignment.training import SFTConfig, train_sft


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", required=True, help="Local data file or Hugging Face dataset ID")
    parser.add_argument("--validation-source", required=True, help="Local data file or Hugging Face dataset ID")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output-dir", default="outputs/sft")
    parser.add_argument("--train-examples", type=int, default=30000)
    parser.add_argument("--validation-examples", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is visible. Run this stage on Kaggle T4.")
    train_rows = load_sft_rows(
        args.train_source, args.train_split, args.dataset_config, args.train_examples, args.seed
    )
    validation_rows = load_sft_rows(
        args.validation_source, args.validation_split, args.dataset_config,
        args.validation_examples, args.seed + 1,
    )
    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model, dtype=torch.float32)
    print(json.dumps({
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "visible_gpus": torch.cuda.device_count(),
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "prompt_template": "### Instruction:\\n{instruction}\\n\\n### Response:\\n",
    }, indent=2))
    config = SFTConfig(
        output_dir=args.output_dir,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        epochs=args.epochs,
        fp16=not args.no_fp16,
        seed=args.seed,
    )
    result = train_sft(model, tokenizer, train_rows, validation_rows, config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()