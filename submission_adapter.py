from __future__ import annotations

from autograder.reference_impl import (
    build_reference_batch,
    reference_dpo_loss,
    reference_response_logps,
    reference_sft_loss,
)


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Instructor reference adapter for the autograder contract.

    Student submissions may organize their implementation however they like, but
    should expose the same functions from this file and call into their own code.
    """

    return build_reference_batch(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    return reference_response_logps(model, tokenizer, prompts, responses, max_length)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    return reference_sft_loss(model, tokenizer, prompts, responses, max_length)


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    return reference_dpo_loss(
        policy_model,
        reference_model,
        tokenizer,
        prompts,
        chosen_responses,
        rejected_responses,
        beta,
        max_length,
    )
