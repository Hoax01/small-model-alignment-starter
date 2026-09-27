# Project Memory: Qwen DPO Alignment Assignment

Last updated: 2026-09-27

## Repository

GitHub repo:

```text
https://github.com/ahmad903501/qwen-dpo-alignment.git
```

Local path:

```text
/home/ahmed-10xe/Desktop/DPO/qwen-dpo-alignment
```

Main branch is `main`.

## Goal

Build a self-contained assignment around small-model LLM alignment on Kaggle T4x2:

- Explain pretraining, SFT, RLHF, and DPO conceptually.
- Students implement SFT and DPO from scratch.
- Use `Qwen/Qwen2.5-0.5B` as the base model.
- Train on Kaggle T4x2.
- Evaluate with AlpacaEval subset generations and Kaggle model-based pairwise judging.
- Later extension: TRL `DPOTrainer` + LoRA on SmolLM 350M with hyperparameter sweeps.

## Current Repo Shape

Important files:

```text
requirements.txt
pyproject.toml
README.md

src/qwen_dpo_alignment/
  data.py
  modeling.py
  prompts.py
  utils.py

scripts/
  make_alpaca_subset.py
  generate_alpaca.py
  train_sft.py
  train_dpo.py
  build_pairwise_eval.py
  judge_pairwise_kaggle.py

notebooks/
  kaggle_train_only.ipynb
  kaggle_generate_outputs.ipynb
  kaggle_eval_only.ipynb

docs/
  assignment_manual.tex
  README.md
  project_memory.md
  build/assignment_manual.pdf
```

Removed obsolete files:

- `notebooks/kaggle_github_runner.ipynb`
- `notebooks/kaggle_pipeline.ipynb`
- `scripts/judge_pairwise_local.py`

Evaluation is Kaggle-model-only; no local judge path should remain.

## Dependency Policy

Kaggle already has CUDA-compatible PyTorch. Do not reinstall or pin `torch`.

Current `requirements.txt` intentionally excludes `torch` and `numpy`:

```text
transformers>=4.43.0,<5
datasets>=2.18,<5
huggingface_hub
accelerate
tqdm
pyyaml
sentencepiece
safetensors
```

Kaggle install pattern:

```bash
python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors
python -m pip install -q -e . --no-deps
```


## Automated Checker / Adapter Contract

Students should include a root-level `submission_adapter.py`. The checker imports this file rather than assuming a specific project structure. Adapter functions should call into the student's own implementation:

```python
def build_sft_batch(tokenizer, prompts, responses, max_length): ...
def compute_response_logprobs(model, tokenizer, prompts, responses, max_length): ...
def compute_sft_loss(model, tokenizer, prompts, responses, max_length): ...
def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts, chosen_responses, rejected_responses, beta, max_length): ...
```

The public tests live in `tests/test_adapter_correctness.py` and use `autograder/toy_lm.py`, so they do not download Hugging Face models. They check prompt/response masking, EOS handling, response-only log probabilities, SFT loss, DPO formula, reference-model no-grad behavior, output ordering, and a tiny DPO optimization sanity check.

Instructor/reference helpers live in `autograder/reference_impl.py`. The repo's current root `submission_adapter.py` is a reference adapter; students can replace it with one that calls their own code.

## Kaggle Workflow

Use split notebooks:

1. `kaggle_train_only.ipynb`
   - trains SFT and DPO
   - saves model checkpoints under `outputs/`

2. `kaggle_generate_outputs.ipynb`
   - creates AlpacaEval subset
   - generates base/SFT/DPO outputs
   - builds pairwise JSONL files

3. `kaggle_eval_only.ipynb`
   - uses Kaggle model-proxy judging via `kaggle_benchmarks`
   - consumes pairwise JSONL files

Kaggle notebooks are isolated. After training, save a Kaggle notebook version and attach its outputs as an input dataset to the generation notebook. Then use paths like:

```python
SFT_MODEL_PATH = "/kaggle/input/YOUR_TRAINING_OUTPUT_DATASET/outputs/sft-qwen-0.5b/best"
DPO_MODEL_PATH = "/kaggle/input/YOUR_TRAINING_OUTPUT_DATASET/outputs/dpo-qwen-0.5b/best"
```

## Main Training Config

Current student-ready values:

```python
SFT_TRAIN_EXAMPLES = 30000
SFT_VAL_EXAMPLES = 1000

DPO_TRAIN_EXAMPLES = 15000
DPO_VAL_EXAMPLES = 1000

SUBSET_SIZE = 300
```

Model:

```text
Qwen/Qwen2.5-0.5B
```

SFT defaults:

```text
max_length = 512
max_prompt_length = 256
per_device_batch_size = 2
gradient_accumulation_steps = 16
effective batch = 32
lr = 2e-5
weight_decay = 0.1
warmup_ratio = 0.03
cosine schedule
grad_clip = 1.0
epochs = 1
```

DPO defaults:

```text
max_length = 512
max_prompt_length = 256
per_device_batch_size = 1
gradient_accumulation_steps = 32
effective batch = 32
lr = 1e-6
beta = 0.1
optimizer = RMSprop
epochs = 1
policy_device = cuda:0
ref_device = cuda:1 when available
```

## Disk-Saving Decisions

Kaggle disk is limited.

SFT:

- `--save_steps` defaults to `0`.
- `0` means no intermediate step checkpoints.
- still saves:
  - `outputs/sft-qwen-0.5b/best`
  - `outputs/sft-qwen-0.5b/final`

DPO:

- saves:
  - `outputs/dpo-qwen-0.5b/best` when validation reward accuracy improves
  - `outputs/dpo-qwen-0.5b/final`

Expected saved folders for larger run:

```text
outputs/sft-qwen-0.5b/best
outputs/sft-qwen-0.5b/final
outputs/dpo-qwen-0.5b/best
outputs/dpo-qwen-0.5b/final
```

## Important Fixes Already Made

### FP16/AMP Bug

Initial bug:

```text
ValueError: Attempting to unscale FP16 gradients.
```

Cause:

- trainable model weights were loaded in fp16
- GradScaler expects fp32 trainable params/gradients

Fix:

- trainable SFT model loads fp32
- trainable DPO policy loads fp32
- AMP autocast still gives fp16 compute
- DPO frozen reference can be fp16

### Deprecated AMP APIs

Replaced:

```python
torch.cuda.amp.autocast
torch.cuda.amp.GradScaler
```

with:

```python
torch.amp.autocast("cuda")
torch.amp.GradScaler("cuda")
```

via helper functions in `src/qwen_dpo_alignment/modeling.py`:

```python
amp_context(...)
make_grad_scaler(...)
```

### Deprecated Transformers dtype

Replaced:

```python
torch_dtype=...
```

with:

```python
dtype=...
```

### Final Gradient Accumulation

Patched SFT and DPO loops to flush the final partial gradient accumulation step if the number of microbatches is not divisible by `gradient_accumulation_steps`.

### AlpacaEval Loading

`datasets` no longer supports old dataset scripts. `make_alpaca_subset.py` now downloads raw files directly from `tatsu-lab/alpaca_eval`:

- `alpaca_eval.json`
- `alpaca_eval_gpt4_baseline.json`
- fallback: `alpaca_eval_all_outputs.json`

### Local Judge Removed

All local judge paths were removed. Evaluation uses:

```text
scripts/judge_pairwise_kaggle.py
```

with Kaggle Models / `kaggle_benchmarks`.

Default judge:

```text
google/gemini-3.6-flash
```

## Earlier Pilot Results

SFT pilot:

- 2k train examples
- 200 validation examples
- throughput around 2.5 examples/sec
- trained successfully after fp32 trainable-weights fix

DPO pilot:

- 1k train pairs
- 200 validation pairs
- completed in about 8.5 minutes
- validation at step 25:

```text
val_loss ≈ 0.6798
val_reward_accuracy ≈ 0.59
val_reward_margin ≈ 0.0597
```

Interpretation:

- DPO loop is working.
- Signal is modest, as expected for a tiny 1k-pair pilot.

## Runtime Estimate For Larger Run

For:

```python
SFT_TRAIN_EXAMPLES = 30000
DPO_TRAIN_EXAMPLES = 15000
SUBSET_SIZE = 300
```

Estimated on Kaggle T4x2:

```text
SFT training:        ~1.5 hours
DPO training:        ~2.0-2.5 hours
Generation:          ~0.5-1.0 hours
Kaggle judging:      ~0.3-0.5 hours
Total:               ~4.5-5.5 hours
Recommended budget:  6 hours
```

## Metrics Reminder

DPO validation tracks:

### Reward Accuracy

Fraction of examples where:

```text
chosen_reward > rejected_reward
```

Example:

```text
reward_accuracy = 0.59
```

means the policy ranks the preferred response above rejected on 59% of validation pairs.

### Reward Margin

Average:

```text
chosen_reward - rejected_reward
```

Positive margin means the model prefers chosen responses on average. A tiny positive margin means weak but correct-direction preference.

## Model Download / Archiving In Kaggle

Zip trained checkpoints for download:

```python
import shutil
from pathlib import Path

sft_path = Path("/kaggle/working/qwen-dpo-alignment/outputs/sft-qwen-0.5b/best")
shutil.make_archive("/kaggle/working/sft_model_best", "zip", sft_path)

dpo_path = Path("/kaggle/working/qwen-dpo-alignment/outputs/dpo-qwen-0.5b/best")
shutil.make_archive("/kaggle/working/dpo_model_best", "zip", dpo_path)
```

Download:

```text
/kaggle/working/sft_model_best.zip
/kaggle/working/dpo_model_best.zip
```

## Assignment Manual

LaTeX source:

```text
docs/assignment_manual.tex
```

PDF built successfully with local Tectonic:

```text
docs/build/assignment_manual.pdf
```

Local compiler path:

```text
.tools/tectonic/tectonic
```

Build command:

```bash
.tools/tectonic/tectonic -X compile docs/assignment_manual.tex --outdir docs/build
```

System LaTeX was not installed and `sudo apt-get install` was blocked by password prompt. Tectonic was installed locally under `.tools/`, which is ignored by git.

Build produced only minor overfull hbox warnings, no fatal errors.

## Git Notes

Recent important commits:

```text
8551400 Clean stale README run examples
9a67e2b Flush final gradient accumulation step
46cf87a Pass tokenizer regex compatibility flag
ba8ac56 Update AMP and model dtype APIs
dcd66f8 Keep trainable models fp32 under AMP
e1b84fa Reduce checkpoint disk usage for Kaggle runs
```

When Kaggle needs latest code:

```bash
cd /kaggle/working/qwen-dpo-alignment
git pull --ff-only origin main
python -m pip install -q -e . --no-deps
```

## Open/Future Work

- Polish the LaTeX manual further if desired.
- The LaTeX manual now includes the TRL + PEFT/LoRA extension task: "The Idiot Sandwich", using SmolLM 350M Instruct and a curated Gordon Ramsay-style preference dataset.
- Possibly remove `fix_mistral_regex=True` from tokenizer loading for conceptual cleanliness, since the model is Qwen and the warning is probably harmless. It is currently guarded with a fallback and should not affect Qwen behavior.
