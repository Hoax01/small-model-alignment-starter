from __future__ import annotations

import importlib
import inspect

import pytest
import torch

from autograder.reference_impl import (
    build_reference_batch,
    reference_dpo_loss,
    reference_response_logps,
    reference_sft_loss,
)
from autograder.toy_lm import ToyCausalLM, ToyTokenizer


REQUIRED_FUNCTIONS = {
    "build_sft_batch": ["tokenizer", "prompts", "responses", "max_length"],
    "compute_response_logprobs": ["model", "tokenizer", "prompts", "responses", "max_length"],
    "compute_sft_loss": ["model", "tokenizer", "prompts", "responses", "max_length"],
    "compute_dpo_loss": [
        "policy_model",
        "reference_model",
        "tokenizer",
        "prompts",
        "chosen_responses",
        "rejected_responses",
        "beta",
        "max_length",
    ],
}


@pytest.fixture()
def adapter():
    return importlib.import_module("submission_adapter")


@pytest.fixture()
def tokenizer():
    return ToyTokenizer()


def assert_batch_equal(actual, expected):
    assert set(actual.keys()) == {"input_ids", "attention_mask", "labels"}
    for key in expected:
        assert isinstance(actual[key], torch.Tensor), f"{key} must be a torch.Tensor"
        assert actual[key].dtype == expected[key].dtype, f"{key} has the wrong dtype"
        assert torch.equal(actual[key], expected[key]), f"{key} does not match the reference batch"


def warm_tokenizer(tokenizer, *string_groups):
    for group in string_groups:
        for text in group:
            tokenizer.encode(text, add_special_tokens=False)
    tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)


def test_required_adapter_functions_exist_with_expected_arguments(adapter):
    for name, expected_args in REQUIRED_FUNCTIONS.items():
        fn = getattr(adapter, name, None)
        assert callable(fn), f"submission_adapter.py must define callable {name}"
        observed_args = list(inspect.signature(fn).parameters)
        assert observed_args == expected_args


def test_build_sft_batch_masks_prompt_and_padding_and_preserves_order(adapter, tokenizer):
    prompts = ["Prompt A: ", "Longer prompt B: ", "Short: "]
    responses = ["x", "yz", "answer."]
    max_length = 24

    expected = build_reference_batch(tokenizer, prompts, responses, max_length)
    actual = adapter.build_sft_batch(tokenizer, prompts, responses, max_length)

    assert_batch_equal(actual, expected)
    assert actual["labels"].eq(-100).any(), "prompt/padding positions should be masked with -100"

    first_response_start = len(tokenizer.encode(prompts[0], add_special_tokens=False))
    assert actual["labels"][0, first_response_start].item() != -100
    assert actual["input_ids"][0, first_response_start].item() == actual["labels"][0, first_response_start].item()


def test_build_sft_batch_truncates_without_losing_all_response_tokens(adapter, tokenizer):
    prompts = ["abcdefghijklmnopqrstuvwxyz"]
    responses = ["XYZ"]
    max_length = 10

    batch = adapter.build_sft_batch(tokenizer, prompts, responses, max_length)

    assert batch["input_ids"].shape == (1, max_length)
    assert batch["attention_mask"].sum().item() == max_length
    assert batch["labels"].ne(-100).sum().item() >= 1
    assert batch["labels"][0, :-1].eq(-100).all()


def test_response_logprobs_match_reference_values(adapter, tokenizer):
    prompts = ["p1: ", "p2: ", "p3: "]
    responses = ["a", "bc", "de."]
    warm_tokenizer(tokenizer, prompts, responses)
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=1.25)

    actual = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=18)
    expected = reference_response_logps(model, tokenizer, prompts, responses, max_length=18)

    assert actual.shape == (3,)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_response_logprobs_are_response_only(adapter, tokenizer):
    prompts = ["aaaa: ", "zzzz: "]
    responses = ["b", "b"]
    warm_tokenizer(tokenizer, prompts, responses)
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.75)

    logps = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=20)

    torch.testing.assert_close(logps[0], logps[1], rtol=1e-6, atol=1e-6)


def test_sft_loss_is_negative_mean_response_logprob(adapter, tokenizer):
    prompts = ["task: ", "ask: "]
    responses = ["alpha", "beta"]
    warm_tokenizer(tokenizer, prompts, responses)
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=-0.5)

    actual = adapter.compute_sft_loss(model, tokenizer, prompts, responses, max_length=20)
    expected = reference_sft_loss(model, tokenizer, prompts, responses, max_length=20)

    assert actual.ndim == 0
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_dpo_loss_matches_reference_formula(adapter, tokenizer):
    prompts = ["instruction one: ", "instruction two: ", "instruction three: "]
    chosen = ["good", "clear", "safe"]
    rejected = ["bad", "vague", "risky"]
    warm_tokenizer(tokenizer, prompts, chosen, rejected)
    policy = ToyCausalLM(tokenizer.vocab_size, bias_scale=1.5)
    reference = ToyCausalLM(tokenizer.vocab_size, bias_scale=-0.25)

    actual = adapter.compute_dpo_loss(policy, reference, tokenizer, prompts, chosen, rejected, beta=0.2, max_length=32)
    expected = reference_dpo_loss(policy, reference, tokenizer, prompts, chosen, rejected, beta=0.2, max_length=32)

    assert actual.ndim == 0
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_dpo_loss_backpropagates_policy_but_not_reference(adapter, tokenizer):
    prompts = ["x: ", "y: "]
    chosen = ["a", "a"]
    rejected = ["b", "b"]
    warm_tokenizer(tokenizer, prompts, chosen, rejected)
    policy = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.0)
    reference = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.0)

    loss = adapter.compute_dpo_loss(policy, reference, tokenizer, prompts, chosen, rejected, beta=0.5, max_length=12)
    loss.backward()

    assert policy.logit_bias.grad is not None
    assert policy.logit_bias.grad.abs().sum().item() > 0
    assert reference.logit_bias.grad is None or reference.logit_bias.grad.abs().sum().item() == 0


def test_output_order_is_restored_after_any_internal_sorting(adapter, tokenizer):
    prompts = ["one: ", "two: ", "three: ", "four: "]
    responses = ["a", "bb", "ccc", "dddd"]
    order = [2, 0, 3, 1]
    warm_tokenizer(tokenizer, prompts, responses)
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.6)

    base = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=20)
    shuffled_prompts = [prompts[i] for i in order]
    shuffled_responses = [responses[i] for i in order]
    shuffled = adapter.compute_response_logprobs(model, tokenizer, shuffled_prompts, shuffled_responses, max_length=20)
    restored = torch.empty_like(shuffled)
    for new_idx, old_idx in enumerate(order):
        restored[old_idx] = shuffled[new_idx]

    torch.testing.assert_close(restored, base, rtol=1e-6, atol=1e-6)


def test_tiny_dpo_optimization_prefers_chosen_response(adapter, tokenizer):
    prompts = ["case 1: ", "case 2: ", "case 3: ", "case 4: "]
    chosen = ["a", "a", "a", "a"]
    rejected = ["b", "b", "b", "b"]
    warm_tokenizer(tokenizer, prompts, chosen, rejected)
    policy = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.0)
    reference = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.0)
    optimizer = torch.optim.SGD(policy.parameters(), lr=1.0)

    initial_loss = adapter.compute_dpo_loss(policy, reference, tokenizer, prompts, chosen, rejected, beta=0.5, max_length=16)
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = adapter.compute_dpo_loss(policy, reference, tokenizer, prompts, chosen, rejected, beta=0.5, max_length=16)
        loss.backward()
        optimizer.step()

    final_loss = adapter.compute_dpo_loss(policy, reference, tokenizer, prompts, chosen, rejected, beta=0.5, max_length=16)
    chosen_logps = adapter.compute_response_logprobs(policy, tokenizer, prompts, chosen, max_length=16)
    rejected_logps = adapter.compute_response_logprobs(policy, tokenizer, prompts, rejected, max_length=16)

    assert final_loss.item() < initial_loss.item()
    assert (chosen_logps - rejected_logps).mean().item() > 0
