from __future__ import annotations

import argparse
import functools

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from qwen_dpo_alignment.data import DPODataset, collate_dpo, extract_hh_pair, load_json_or_jsonl, maybe_sample
from qwen_dpo_alignment.modeling import dpo_loss, load_causal_lm, load_tokenizer, sequence_logps
from qwen_dpo_alignment.utils import cuda_summary, ensure_dir, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen with from-scratch DPO.")
    parser.add_argument("--model_name_or_path", required=True, help="SFT checkpoint used for both policy init and reference.")
    parser.add_argument("--dataset_name", default="Anthropic/hh-rlhf")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="test")
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--val_file", default=None)
    parser.add_argument("--output_dir", default="outputs/dpo-qwen-0.5b")
    parser.add_argument("--max_train_examples", type=int, default=10000)
    parser.add_argument("--max_val_examples", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    return parser.parse_args()


def load_hh_rows(dataset_name: str, split: str, path: str | None, max_examples: int | None, seed: int):
    if path:
        raw_rows = load_json_or_jsonl(path)
    else:
        raw_rows = [dict(row) for row in load_dataset(dataset_name, split=split)]
    rows = []
    for row in raw_rows:
        if {"instruction", "chosen", "rejected"}.issubset(row):
            rows.append(row)
        else:
            parsed = extract_hh_pair(row, source=dataset_name)
            if parsed is not None:
                rows.append(parsed)
    return maybe_sample(rows, max_examples, seed)


def logps_for_batch(model, batch, device: str, use_fp16: bool) -> tuple[torch.Tensor, torch.Tensor]:
    chosen = batch["chosen"].to(device)
    rejected = batch["rejected"].to(device)
    with torch.cuda.amp.autocast(enabled=use_fp16):
        chosen_logits = model(input_ids=chosen.input_ids, attention_mask=chosen.attention_mask).logits
        rejected_logits = model(input_ids=rejected.input_ids, attention_mask=rejected.attention_mask).logits
    chosen_logps = sequence_logps(chosen_logits, chosen.labels)
    rejected_logps = sequence_logps(rejected_logits, rejected.labels)
    return chosen_logps, rejected_logps


@torch.no_grad()
def evaluate(policy, reference, dataloader, policy_device: str, ref_device: str, use_fp16: bool, beta: float):
    policy.eval()
    reference.eval()
    losses = []
    accuracies = []
    margins = []
    for batch in tqdm(dataloader, desc="eval", leave=False):
        policy_chosen, policy_rejected = logps_for_batch(policy, batch, policy_device, use_fp16)
        ref_chosen, ref_rejected = logps_for_batch(reference, batch, ref_device, use_fp16)
        loss, metrics = dpo_loss(
            policy_chosen,
            policy_rejected,
            ref_chosen.to(policy_device),
            ref_rejected.to(policy_device),
            beta,
        )
        losses.append(loss.detach().float().cpu())
        accuracies.append(metrics["reward_accuracy"].detach().float().cpu())
        margins.append(metrics["reward_margin"].detach().float().cpu())
    policy.train()
    return {
        "loss": torch.stack(losses).mean().item(),
        "reward_accuracy": torch.stack(accuracies).mean().item(),
        "reward_margin": torch.stack(margins).mean().item(),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    write_json(output_dir / "dpo_args.json", vars(args))

    print(cuda_summary())
    policy_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ref_device = "cuda:1" if torch.cuda.device_count() > 1 else policy_device
    use_fp16 = (not args.no_fp16) and policy_device.startswith("cuda")
    print(f"policy_device={policy_device} ref_device={ref_device}")

    tokenizer = load_tokenizer(args.model_name_or_path)
    policy = load_causal_lm(args.model_name_or_path, device=policy_device, fp16=use_fp16)
    reference = load_causal_lm(args.model_name_or_path, device=ref_device, fp16=use_fp16)
    reference.requires_grad_(False)
    reference.eval()
    if not args.no_gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        policy.config.use_cache = False

    train_rows = load_hh_rows(args.dataset_name, args.train_split, args.train_file, args.max_train_examples, args.seed)
    val_rows = load_hh_rows(args.dataset_name, args.val_split, args.val_file, args.max_val_examples, args.seed + 1)
    train_dataset = DPODataset(train_rows, tokenizer, args.max_length, args.max_prompt_length)
    val_dataset = DPODataset(val_rows, tokenizer, args.max_length, args.max_prompt_length)
    print(f"Loaded {len(train_dataset)} DPO train pairs and {len(val_dataset)} validation pairs.")

    collate = functools.partial(collate_dpo, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.per_device_batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.RMSprop(policy.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    best_accuracy = -1.0
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    policy.train()

    for epoch in range(args.num_epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.num_epochs}")
        for micro_step, batch in enumerate(pbar, start=1):
            policy_chosen, policy_rejected = logps_for_batch(policy, batch, policy_device, use_fp16)
            with torch.no_grad():
                ref_chosen, ref_rejected = logps_for_batch(reference, batch, ref_device, use_fp16)
            loss, metrics = dpo_loss(
                policy_chosen,
                policy_rejected,
                ref_chosen.to(policy_device),
                ref_rejected.to(policy_device),
                args.beta,
            )
            scaler.scale(loss / args.gradient_accumulation_steps).backward()

            if micro_step % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{metrics['reward_accuracy'].item():.3f}")
                if global_step % args.logging_steps == 0:
                    print(
                        f"step={global_step} loss={loss.item():.4f} "
                        f"reward_acc={metrics['reward_accuracy'].item():.3f} "
                        f"margin={metrics['reward_margin'].item():.4f}"
                    )
                if global_step % args.eval_steps == 0:
                    val_metrics = evaluate(policy, reference, val_loader, policy_device, ref_device, use_fp16, args.beta)
                    print(f"step={global_step} val={val_metrics}")
                    if val_metrics["reward_accuracy"] > best_accuracy:
                        best_accuracy = val_metrics["reward_accuracy"]
                        best_dir = output_dir / "best"
                        policy.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
                        write_json(best_dir / "metrics.json", {"step": global_step, **val_metrics})

    final_dir = output_dir / "final"
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final DPO model to {final_dir}")


if __name__ == "__main__":
    main()

