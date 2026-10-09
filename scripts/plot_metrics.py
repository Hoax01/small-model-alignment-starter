"""Plot SFT and DPO metrics.jsonl files for the final report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_jsonl(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", default=None, help="SFT metrics.jsonl")
    parser.add_argument("--dpo", action="append", default=[], help="DPO metrics.jsonl (repeatable)")
    parser.add_argument("--labels", action="append", default=[], help="Label matching each --dpo path")
    parser.add_argument("--output-dir", default="outputs/plots")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if args.sft:
        rows = read_jsonl(args.sft)
        epochs = [row["epoch"] for row in rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(epochs, [row["train_loss"] for row in rows], marker="o", label="Train")
        val = [row.get("validation_loss") for row in rows]
        if any(value is not None for value in val):
            ax.plot(epochs, val, marker="o", label="Validation")
        ax.set(title="SFT loss", xlabel="Epoch", ylabel="Response-token NLL")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "sft_loss.png", dpi=180)
        plt.close(fig)

    labels = args.labels + [f"DPO {index + 1}" for index in range(len(args.labels), len(args.dpo))]
    if len(labels) != len(args.dpo):
        raise ValueError("Provide one --labels value per --dpo metrics file, or omit all labels.")
    if args.dpo:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for path, label in zip(args.dpo, labels):
            rows = read_jsonl(path)
            epochs = [row["epoch"] for row in rows]
            axes[0].plot(epochs, [row["train_loss"] for row in rows], marker="o", label=f"{label} train")
            val = [row.get("validation_loss") for row in rows]
            if any(value is not None for value in val):
                axes[0].plot(epochs, val, marker="s", linestyle="--", label=f"{label} validation")
            accuracy = [row.get("reward_accuracy") for row in rows]
            margin = [row.get("reward_margin") for row in rows]
            axes[1].plot(epochs, accuracy, marker="o", label=label)
            axes[2].plot(epochs, margin, marker="o", label=label)
        axes[0].set(title="DPO loss", xlabel="Epoch", ylabel="Loss")
        axes[1].set(title="Validation reward accuracy", xlabel="Epoch", ylabel="Accuracy")
        axes[2].set(title="Validation reward margin", xlabel="Epoch", ylabel="Margin")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "dpo_metrics.png", dpi=180)
        plt.close(fig)

    print(f"Saved plots under {output}")


if __name__ == "__main__":
    main()