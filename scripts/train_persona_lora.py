"""TRL DPOTrainer + PEFT LoRA persona-style run (the assignment exception)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from alignment.data import load_preference_rows
from alignment.models import load_tokenizer


MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"


def _truncate_text(tokenizer, text: str, max_tokens: int) -> str:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/persona_lora/ramsay_style_preferences.json",
        help="The provided persona preference JSON file",
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--output-dir", default="outputs/persona_lora")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def main():
    try:
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError("Install the assignment requirements, including trl and peft, before this task.") from exc

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Persona LoRA training is configured for a Kaggle CUDA GPU.")
    tokenizer = load_tokenizer(args.model)
    # Current TRL's DPO collator requires left padding for prompt-completion batches.
    tokenizer.padding_side = "left"

    rows = load_preference_rows(args.data)
    conversational = []
    for row in rows:
        prompt = _truncate_text(tokenizer, row["prompt"], args.max_prompt_tokens)
        conversational.append({
            "prompt": [{"role": "user", "content": prompt}],
            "chosen": [{"role": "assistant", "content": row["chosen"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}],
        })
    dataset = Dataset.from_list(conversational)
    split = dataset.train_test_split(test_size=max(1, round(0.2 * len(dataset))), seed=args.seed)

    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    training_args = DPOConfig(
        output_dir=args.output_dir,
        beta=0.1,
        loss_type="sigmoid",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_length=args.max_length,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=5,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=lora,
    )
    train_result = trainer.train()
    output_dir = Path(args.output_dir)
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(output_dir / "best")
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    (output_dir / "lora_config.json").write_text(json.dumps({
        "base_model": args.model,
        "dataset": args.data,
        "train_rows": len(split["train"]),
        "validation_rows": len(split["test"]),
        "beta": training_args.beta,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "max_prompt_tokens": args.max_prompt_tokens,
        "target_modules": lora.target_modules,
        "rank": lora.r,
        "alpha": lora.lora_alpha,
        "dropout": lora.lora_dropout,
        "best_checkpoint_metric": "eval_loss",
    }, indent=2), encoding="utf-8")
    print(json.dumps(train_result.metrics, indent=2))


if __name__ == "__main__":
    main()