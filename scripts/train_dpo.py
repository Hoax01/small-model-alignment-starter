"""Run one from-scratch DPO track using an SFT policy and frozen reference."""

from __future__ import annotations

import argparse
import json

import torch

from alignment.data import load_preference_rows
from alignment.models import load_causal_lm, load_tokenizer
from alignment.training import DPOConfig, train_dpo


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", required=True, help="Local data file or Hugging Face dataset ID")
    parser.add_argument("--validation-source", required=True, help="Local data file or Hugging Face dataset ID")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/dpo")
    parser.add_argument("--track", choices=("hh", "ultrafeedback"), default="hh")
    parser.add_argument("--train-pairs", type=int, default=15000)
    parser.add_argument("--validation-pairs", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--reference-device", default="cuda:1")
    parser.add_argument("--optimizer", choices=("adamw", "rmsprop"), default="adamw")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    for device_name in (args.policy_device, args.reference_device):
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is visible. Run this stage on Kaggle T4.")
    if args.policy_device == args.reference_device and torch.cuda.device_count() > 1:
        print("Note: policy and reference share a device; use cuda:0/cuda:1 to split their memory.")
    train_rows = load_preference_rows(
        args.train_source, args.train_split, args.dataset_config, args.train_pairs, args.seed
    )
    validation_rows = load_preference_rows(
        args.validation_source, args.validation_split, args.dataset_config,
        args.validation_pairs, args.seed + 1,
    )
    tokenizer = load_tokenizer(args.sft_checkpoint)
    policy = load_causal_lm(args.sft_checkpoint, dtype=torch.float32)
    reference_dtype = torch.float16 if args.reference_device.startswith("cuda") and not args.no_fp16 else torch.float32
    reference = load_causal_lm(args.sft_checkpoint, dtype=reference_dtype)
    print(json.dumps({
        "track": args.track,
        "torch": torch.__version__,
        "visible_gpus": torch.cuda.device_count(),
        "train_pairs": len(train_rows),
        "validation_pairs": len(validation_rows),
        "sft_checkpoint": args.sft_checkpoint,
    }, indent=2))
    config = DPOConfig(
        output_dir=args.output_dir,
        policy_device=args.policy_device,
        reference_device=args.reference_device,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta=args.beta,
        epochs=args.epochs,
        fp16=not args.no_fp16,
        optimizer=args.optimizer,
        seed=args.seed,
    )
    result = train_dpo(policy, reference, tokenizer, train_rows, validation_rows, config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()