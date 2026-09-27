# Adapter-Based Autograder

Instructor note: this directory currently contains reference logic for developing the checker. If you want a strict assignment release, publish the adapter contract and a small smoke test, but keep `autograder/reference_impl.py` and the full correctness suite private.


Students may organize their code however they want, but their repository must expose a file at the repository root:

```text
submission_adapter.py
```

The checker imports that file and calls a fixed API. Students should implement these functions by calling into their own code.

Required functions:

```python
def build_sft_batch(tokenizer, prompts, responses, max_length): ...

def compute_response_logprobs(model, tokenizer, prompts, responses, max_length): ...

def compute_sft_loss(model, tokenizer, prompts, responses, max_length): ...

def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts, chosen_responses, rejected_responses, beta, max_length): ...
```

Important contract details:

- `prompts`, `responses`, `chosen_responses`, and `rejected_responses` are lists of strings.
- Returned tensors must preserve the same row order as the input lists.
- Internal sorting/packing is allowed only if outputs are restored to input order.
- Log probabilities and losses must be computed over response tokens only.
- The response should include the tokenizer EOS token when one is available.
- Prompt tokens and padding tokens must not contribute to loss/log probability.
- `compute_dpo_loss` must return a scalar mean loss.
- DPO reference-model computations should not create gradients for the reference model.

Run from a submitted repo with:

```bash
python -m pip install pytest torch
PYTHONPATH=. pytest tests/test_adapter_correctness.py
```

The tests use a toy tokenizer/model, not Hugging Face downloads.
