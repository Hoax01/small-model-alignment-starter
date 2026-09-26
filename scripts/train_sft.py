from __future__ import annotations

import argparse
import functools
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from qwen_dpo_alignment.data import SFTDataset, collate_sft, load_json_or_jsonl, maybe_sample
from qwen_dpo_alignment.modeling import amp_context, load_causal_lm, load_tokenizer, make_grad_scaler
from qwen_dpo_alignment.utils import cuda_summary, ensure_dir, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised fine-tune Qwen with a plain PyTorch loop.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset_name", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--train_split", default="train_sft")
    parser.add_argument("--val_split", default="test_sft")
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--val_file", default=None)
    parser.add_argument("--output_dir", default="outputs/sft-qwen-0.5b")
    parser.add_argument("--max_train_examples", type=int, default=20000)
    parser.add_argument("--max_val_examples", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--save_steps", type=int, default=0, help="Save intermediate step checkpoints every N optimizer steps; 0 disables them.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    return parser.parse_args()


def load_rows(dataset_name: str, split: str, path: str | None, max_examples: int | None, seed: int):
    if path:
        rows = load_json_or_jsonl(path)
    else:
        rows = [dict(row) for row in load_dataset(dataset_name, split=split)]
    return maybe_sample(rows, max_examples, seed)


@torch.no_grad()
def evaluate(model, dataloader, device: str, use_fp16: bool) -> float:
    model.eval()
    losses = []
    for batch in tqdm(dataloader, desc="eval", leave=False):
        batch = batch.to(device)
        with amp_context(device, use_fp16):
            loss = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
            ).loss
        losses.append(loss.detach().float().cpu())
    model.train()
    return torch.stack(losses).mean().item()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    write_json(output_dir / "sft_args.json", vars(args))

    print(cuda_summary())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_fp16 = (not args.no_fp16) and device.startswith("cuda")

    tokenizer = load_tokenizer(args.model_name_or_path)
    # Keep trainable weights in fp32. AMP autocast still uses fp16 compute on T4,
    # but GradScaler requires fp32 gradients/parameters to unscale safely.
    model = load_causal_lm(args.model_name_or_path, device=device, fp16=False)
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_rows = load_rows(args.dataset_name, args.train_split, args.train_file, args.max_train_examples, args.seed)
    val_rows = load_rows(args.dataset_name, args.val_split, args.val_file, args.max_val_examples, args.seed + 1)
    train_dataset = SFTDataset(train_rows, tokenizer, args.max_length, args.max_prompt_length)
    val_dataset = SFTDataset(val_rows, tokenizer, args.max_length, args.max_prompt_length)
    print(f"Loaded {len(train_dataset)} SFT train examples and {len(val_dataset)} validation examples.")

    collate = functools.partial(collate_sft, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.per_device_batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_update_steps = max(1, (len(train_loader) * args.num_epochs) // args.gradient_accumulation_steps)
    warmup_steps = int(total_update_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_update_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = make_grad_scaler(device, use_fp16)

    best_val_loss = float("inf")
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()

    for epoch in range(args.num_epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.num_epochs}")
        for micro_step, batch in enumerate(pbar, start=1):
            batch = batch.to(device)
            with amp_context(device, use_fp16):
                loss = model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    labels=batch.labels,
                ).loss
                scaled_loss = loss / args.gradient_accumulation_steps

            scaler.scale(scaled_loss).backward()

            if micro_step % args.gradient_accumulation_steps == 0 or micro_step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                if global_step % args.logging_steps == 0:
                    print(f"step={global_step} train_loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")
                if global_step % args.eval_steps == 0:
                    val_loss = evaluate(model, val_loader, device, use_fp16)
                    print(f"step={global_step} val_loss={val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_dir = output_dir / "best"
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
                        write_json(best_dir / "metrics.json", {"step": global_step, "val_loss": best_val_loss})
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt_dir = output_dir / f"step-{global_step}"
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    if not Path(output_dir / "best").exists():
        model.save_pretrained(output_dir / "best")
        tokenizer.save_pretrained(output_dir / "best")
    print(f"Saved final SFT model to {final_dir}")


if __name__ == "__main__":
    main()

