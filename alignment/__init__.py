"""Small-model alignment training utilities."""

from .objectives import (
    build_dpo_batch,
    build_sft_batch,
    compute_dpo_loss,
    compute_dpo_loss_from_logps,
    compute_response_logprobs,
    compute_sft_loss,
)

__all__ = [
    "build_dpo_batch",
    "build_sft_batch",
    "compute_dpo_loss",
    "compute_dpo_loss_from_logps",
    "compute_response_logprobs",
    "compute_sft_loss",
]