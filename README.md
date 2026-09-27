# Qwen2.5-0.5B SFT + From-Scratch DPO on Kaggle T4x2

This repo is a small, script-first implementation of the assignment-style alignment pipeline adapted for Kaggle:

1. Create a fixed AlpacaEval subset.
2. Generate base-model responses.
3. Supervised fine-tune `Qwen/Qwen2.5-0.5B`.
4. Generate SFT responses.
5. Train DPO from scratch on HH preference pairs.
6. Generate DPO responses.
7. Compare models with pairwise judging or reference-output judging.

The training loops are plain PyTorch. They do not use Hugging Face `Trainer` or TRL `DPOTrainer`.

## Kaggle Setup

Use the Kaggle accelerator `GPU T4 x2`. Kaggle already includes a CUDA-compatible PyTorch build, so this repo intentionally does not pin or reinstall `torch`. The install also avoids upgrading NumPy, which can disturb Kaggle preinstalled packages.

If running commands manually:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For notebook-based Kaggle runs after pushing to GitHub, use the split notebooks:

```text
notebooks/kaggle_train_only.ipynb
notebooks/kaggle_generate_outputs.ipynb
notebooks/kaggle_eval_only.ipynb
```

Set `REPO_URL` in the first code cell of each notebook you use.

Check GPUs:

```bash
nvidia-smi
```


## Split Kaggle Notebooks

For cleaner experiments, prefer the split notebooks:

- `notebooks/kaggle_train_only.ipynb`: trains SFT and DPO checkpoints only.
- `notebooks/kaggle_generate_outputs.ipynb`: loads base/SFT/DPO checkpoints and writes AlpacaEval generations plus pairwise JSONL files.
- `notebooks/kaggle_eval_only.ipynb`: consumes pairwise JSONL files and uses Kaggle Models through `kaggle_benchmarks` for judging.


Kaggle-model evaluation uses `scripts/judge_pairwise_kaggle.py`, which expects Kaggle's `kaggle_benchmarks` package and model-proxy access. It does not load a judge model into GPU memory.

Kaggle notebooks are isolated. To use trained checkpoints in a separate generation notebook, save a version of the training notebook and attach its outputs as an input dataset to the generation notebook. Then set `SFT_MODEL_PATH` and `DPO_MODEL_PATH` to the attached `/kaggle/input/.../outputs/.../best` paths.

## 1. Create AlpacaEval Subsets

Smoke test:

```bash
python scripts/make_alpaca_subset.py \
  --size 50 \
  --seed 42 \
  --output_dir data/alpaca_eval
```

Normal pilot:

```bash
python scripts/make_alpaca_subset.py \
  --size 200 \
  --seed 42 \
  --output_dir data/alpaca_eval
```

The script writes:

- `data/alpaca_eval/alpacaeval_300_seed42.json`
- `data/alpaca_eval/alpacaeval_300_seed42_references.json`

If GPT-4/GPT-style reference outputs are found in `alpaca_eval_all_outputs.json`, each subset row includes `reference_output`.

## 2. Generate Base Responses

```bash
python scripts/generate_alpaca.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --subset_path data/alpaca_eval/alpacaeval_300_seed42.json \
  --output_path outputs/eval/base_alpacaeval_300.json \
  --generator_name qwen2.5-0.5b-base \
  --batch_size 8 \
  --max_new_tokens 512
```

## 3. SFT

Pilot run:

```bash
python scripts/train_sft.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --output_dir outputs/sft-pilot \
  --max_train_examples 2000 \
  --max_val_examples 200 \
  --per_device_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --eval_steps 50
```

Larger run:

```bash
python scripts/train_sft.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --output_dir outputs/sft-qwen-0.5b \
  --max_train_examples 30000 \
  --max_val_examples 1000 \
  --per_device_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --eval_steps 250
```

Defaults use `HuggingFaceH4/ultrachat_200k`, split `train_sft` and `test_sft`. You can use local JSON/JSONL with `--train_file` and `--val_file`; supported schemas include `prompt/response`, `instruction/output`, and chat `messages`.

## 4. Generate SFT Responses

```bash
python scripts/generate_alpaca.py \
  --model_name_or_path outputs/sft-qwen-0.5b/best \
  --subset_path data/alpaca_eval/alpacaeval_300_seed42.json \
  --output_path outputs/eval/sft_alpacaeval_300.json \
  --generator_name qwen2.5-0.5b-sft \
  --batch_size 8 \
  --max_new_tokens 512
```

## 5. DPO

DPO loads the trainable policy on `cuda:0` and the frozen reference model on `cuda:1` when two GPUs are available.

Pilot run:

```bash
python scripts/train_dpo.py \
  --model_name_or_path outputs/sft-pilot/best \
  --output_dir outputs/dpo-pilot \
  --max_train_examples 1000 \
  --max_val_examples 200 \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --eval_steps 25
```

Larger run:

```bash
python scripts/train_dpo.py \
  --model_name_or_path outputs/sft-qwen-0.5b/best \
  --output_dir outputs/dpo-qwen-0.5b \
  --max_train_examples 15000 \
  --max_val_examples 1000 \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --eval_steps 250 \
  --beta 0.1 \
  --learning_rate 1e-6
```

Defaults use `Anthropic/hh-rlhf`. You can pass local HH-style JSON/JSONL with `--train_file` and `--val_file`.

## 6. Generate DPO Responses

Run a quick sanity check before full generation:

```bash
python scripts/generate_alpaca.py \
  --model_name_or_path outputs/dpo-qwen-0.5b/best \
  --subset_path data/alpaca_eval/alpacaeval_300_seed42.json \
  --output_path outputs/eval/sanity_dpo_5.json \
  --generator_name qwen2.5-0.5b-dpo-sanity \
  --batch_size 2 \
  --max_new_tokens 256 \
  --limit 5 \
  --device cuda:0
```

For Kaggle T4x2, split generation over both GPUs and merge the shard files in the notebook:

```bash
python scripts/generate_alpaca.py \
  --model_name_or_path outputs/dpo-qwen-0.5b/best \
  --subset_path data/alpaca_eval/alpacaeval_300_seed42.json \
  --output_path outputs/eval/dpo_shard0.json \
  --generator_name qwen2.5-0.5b-dpo \
  --batch_size 8 \
  --max_new_tokens 512 \
  --device cuda:0 \
  --num_shards 2 \
  --shard_index 0 &

python scripts/generate_alpaca.py \
  --model_name_or_path outputs/dpo-qwen-0.5b/best \
  --subset_path data/alpaca_eval/alpacaeval_300_seed42.json \
  --output_path outputs/eval/dpo_shard1.json \
  --generator_name qwen2.5-0.5b-dpo \
  --batch_size 8 \
  --max_new_tokens 512 \
  --device cuda:1 \
  --num_shards 2 \
  --shard_index 1 &

wait
```

## 7. Pairwise Evaluation

Compare DPO vs SFT:

```bash
python scripts/build_pairwise_eval.py \
  --a_outputs outputs/eval/dpo_alpacaeval_300.json \
  --b_outputs outputs/eval/sft_alpacaeval_300.json \
  --a_name dpo \
  --b_name sft \
  --output_path outputs/eval/dpo_vs_sft_pairs.jsonl
```

Judge with Kaggle Models:

```bash
python scripts/judge_pairwise_kaggle.py \
  --pairs_path outputs/eval/dpo_vs_sft_pairs.jsonl \
  --output_path outputs/eval/dpo_vs_sft_judged_kaggle.jsonl \
  --judge_model google/gemini-3.6-flash \
  --batch_size 5 \
  --sleep_seconds 5 \
  --max_retries 3 \
  --reuse_judgments
```

Compare a model against GPT-style reference outputs from the subset:

```bash
python scripts/build_pairwise_eval.py \
  --a_outputs outputs/eval/dpo_alpacaeval_300.json \
  --b_from_reference \
  --a_name dpo \
  --b_name gpt_reference \
  --output_path outputs/eval/dpo_vs_reference_pairs.jsonl
```

## Suggested Workflow

First complete the tiny end-to-end loop:

- AlpacaEval subset: 50
- SFT examples: 2k
- DPO pairs: 1k

Then scale:

- AlpacaEval subset: 300
- SFT examples: 30k
- DPO pairs: 15k

This keeps Kaggle failures cheap while we shake out memory and data issues.

