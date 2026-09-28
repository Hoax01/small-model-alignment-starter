from __future__ import annotations

import argparse
import functools

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from qwen_dpo_alignment.data import DPODataset, collate_dpo, extract_dpo_pair, load_json_or_jsonl, maybe_sample
from qwen_dpo_alignment.modeling import amp_context, dpo_loss, load_causal_lm, load_tokenizer, make_grad_scaler, sequence_logps
from qwen_dpo_alignment.utils import append_jsonl, cuda_summary, ensure_dir, seed_everything, write_json


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
    parser.add_argument("--prompt_format", choices=["plain", "chat_template"], default="plain")
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--optimizer", choices=["rmsprop", "adamw"], default="rmsprop")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names to adapt with LoRA.",
    )
    return parser.parse_args()


def resolve_split(dataset_name: str, split: str) -> str:
    if "ultrafeedback_binarized" in dataset_name.lower():
        if split == "train":
            return "train_prefs"
        if split == "test":
            return "test_prefs"
    return split


def load_dpo_rows(dataset_name: str, split: str, path: str | None, max_examples: int | None, seed: int):
    if path:
        raw_rows = load_json_or_jsonl(path)
        source = str(path)
    else:
        resolved_split = resolve_split(dataset_name, split)
        raw_rows = [dict(row) for row in load_dataset(dataset_name, split=resolved_split)]
        source = dataset_name
    rows = []
    for row in raw_rows:
        parsed = extract_dpo_pair(row, source=source)
        if parsed is not None:
            rows.append(parsed)
    return maybe_sample(rows, max_examples, seed)


def maybe_apply_lora(model, args):
    if not args.use_lora:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("LoRA training requires peft. Install requirements.txt or `pip install peft`.") from exc

    target_modules = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.print_trainable_parameters()
    return model


def build_optimizer(parameters, args):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    return torch.optim.RMSprop(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)


def logps_for_batch(model, batch, device: str, use_fp16: bool) -> tuple[torch.Tensor, torch.Tensor]:
    chosen = batch["chosen"].to(device)
    rejected = batch["rejected"].to(device)
    with amp_context(device, use_fp16):
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
    policy_logratios = []
    ref_logratios = []
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
        policy_logratios.append(metrics["policy_logratio"].detach().float().cpu())
        ref_logratios.append(metrics["ref_logratio"].detach().float().cpu())
    policy.train()
    return {
        "loss": torch.stack(losses).mean().item(),
        "reward_accuracy": torch.stack(accuracies).mean().item(),
        "reward_margin": torch.stack(margins).mean().item(),
        "policy_logratio": torch.stack(policy_logratios).mean().item(),
        "ref_logratio": torch.stack(ref_logratios).mean().item(),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    write_json(output_dir / "dpo_args.json", vars(args))
    metrics_path.write_text("", encoding="utf-8")

    print(cuda_summary())
    policy_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ref_device = "cuda:1" if torch.cuda.device_count() > 1 else policy_device
    use_fp16 = (not args.no_fp16) and policy_device.startswith("cuda")
    print(f"policy_device={policy_device} ref_device={ref_device}")

    tokenizer = load_tokenizer(args.model_name_or_path)
    # Full DPO keeps trainable weights in fp32 because GradScaler cannot unscale
    # fp16 trainable gradients. LoRA freezes the base model, so the base can stay
    # fp16 while trainable adapter weights are cast back to fp32.
    policy = load_causal_lm(args.model_name_or_path, device=policy_device, fp16=use_fp16 if args.use_lora else False)
    policy = maybe_apply_lora(policy, args)
    # The frozen reference can be fp16 to save memory. It is only used under no_grad.
    reference = load_causal_lm(args.model_name_or_path, device=ref_device, fp16=use_fp16)
    reference.requires_grad_(False)
    reference.eval()
    if not args.no_gradient_checkpointing:
        if args.use_lora and hasattr(policy, "enable_input_require_grads"):
            policy.enable_input_require_grads()
        policy.gradient_checkpointing_enable()
        policy.config.use_cache = False

    train_rows = load_dpo_rows(args.dataset_name, args.train_split, args.train_file, args.max_train_examples, args.seed)
    val_rows = load_dpo_rows(args.dataset_name, args.val_split, args.val_file, args.max_val_examples, args.seed + 1)
    train_dataset = DPODataset(train_rows, tokenizer, args.max_length, args.max_prompt_length, args.prompt_format)
    val_dataset = DPODataset(val_rows, tokenizer, args.max_length, args.max_prompt_length, args.prompt_format)
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

    optimizer = build_optimizer((param for param in policy.parameters() if param.requires_grad), args)
    scaler = make_grad_scaler(policy_device, use_fp16)
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

            if micro_step % args.gradient_accumulation_steps == 0 or micro_step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                train_metrics = {name: value.detach().float().item() for name, value in metrics.items()}
                train_loss = loss.detach().float().item()
                pbar.set_postfix(loss=f"{train_loss:.4f}", acc=f"{train_metrics['reward_accuracy']:.3f}")
                append_jsonl(
                    metrics_path,
                    {
                        "phase": "train",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "loss": train_loss,
                        "learning_rate": args.learning_rate,
                        **train_metrics,
                    },
                )
                if global_step % args.logging_steps == 0:
                    print(
                        f"step={global_step} loss={train_loss:.4f} "
                        f"reward_acc={train_metrics['reward_accuracy']:.3f} "
                        f"margin={train_metrics['reward_margin']:.4f}"
                    )
                if global_step % args.eval_steps == 0:
                    val_metrics = evaluate(policy, reference, val_loader, policy_device, ref_device, use_fp16, args.beta)
                    append_jsonl(
                        metrics_path,
                        {
                            "phase": "eval",
                            "epoch": epoch + 1,
                            "step": global_step,
                            "learning_rate": args.learning_rate,
                            **{f"val_{name}": value for name, value in val_metrics.items()},
                        },
                    )
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

