# Small-Model Alignment: SFT + DPO

This repository implements PA2 using the assignment manual and its public [starter repo](https://github.com/ahmad903501/small-model-alignment-starter). The prompt format is fixed across SFT, both from-scratch DPO runs, and generation:

~~~text
### Instruction:
{instruction}

### Response:
~~~

SFT loss averages over response tokens only. Prompt/padding labels are masked with -100, and EOS is appended before truncation when the tokenizer provides it.

## Kaggle setup

Keep Kaggle's installed PyTorch/CUDA build; do not reinstall torch.

For the complete PA2 run, import `PA2_kaggle_all_in_one.ipynb` into Kaggle, select **T4 x2** and enable Internet, then run its setup and preflight cells. Once both GPUs, the public adapter checks, and the judge ping pass, choose **Save Version → Save & Run All**. This starts a clean background run and saves the generated `/kaggle/working` files when it finishes. The notebook includes dataset downloads, all training tracks, generation, blinded judging, plots, a PDF report, an evidence ZIP, and a final checklist. Check `outputs/final_checklist.json` in the saved version before submitting.

~~~bash
python -m pip install -q -r requirements.txt
python -m pip install -q -e . --no-deps
python -m scripts.check_environment
~~~

Inputs may be local JSON, JSONL, or CSV files, or Hugging Face dataset IDs. SFT rows should contain instruction/output (or prompt/response) columns, or one user/assistant turn. Preference rows should contain prompt/chosen/rejected. Multi-turn rows are filtered unless prepared as single-turn inputs.


## Adapter smoke tests

The starter includes offline public checks for the six submission adapter functions. Run them from the repository root after installing the dependencies:

~~~bash
python -m pytest -q tests/test_public_adapter_smoke.py
~~~

## Train SFT

Defaults: Qwen/Qwen2.5-0.5B, 30,000 train examples, 1,000 validation examples, context length 512, per-device batch 2, accumulation 16, AdamW at 2e-5, cosine decay, 3% warmup, gradient clip 1.0, one epoch.

~~~bash
python -m scripts.train_sft --train-source /kaggle/input/your-data/train.jsonl --validation-source /kaggle/input/your-data/validation.jsonl --output-dir /kaggle/working/outputs/sft
~~~

The loop keeps trainable weights in FP32, uses fp16 autocast and gradient scaling, evaluates response-only validation loss, and saves best/final checkpoints plus metrics.jsonl.

## Train both from-scratch DPO tracks

Start both runs from the same best SFT checkpoint. Defaults: beta 0.1, batch 1, accumulation 32, prompt cap 256, context length 512, fixed learning rate 1e-6, gradient clip 1.0, one epoch. The policy is FP32 on cuda:0; a frozen FP16 reference is placed on cuda:1.

~~~bash
python -m scripts.train_dpo --track hh --train-source /kaggle/input/your-data/hh_train.jsonl --validation-source /kaggle/input/your-data/hh_validation.jsonl --sft-checkpoint /kaggle/working/outputs/sft/best --output-dir /kaggle/working/outputs/dpo_hh
python -m scripts.train_dpo --track ultrafeedback --train-source /kaggle/input/your-data/uf_train.jsonl --validation-source /kaggle/input/your-data/uf_validation.jsonl --sft-checkpoint /kaggle/working/outputs/sft/best --output-dir /kaggle/working/outputs/dpo_ultrafeedback
~~~

Each run logs validation DPO loss, strict reward accuracy, and reward margin. Override device options if your Kaggle runtime differs.

## Persona LoRA-DPO

This track uses TRL DPOTrainer and PEFT LoRA on the supplied 220-row dataset. The script converts records to conversational messages and uses a seeded held-out split.

~~~bash
python -m scripts.train_persona_lora --output-dir /kaggle/working/outputs/persona_lora
~~~

Defaults: SmolLM2-360M-Instruct, rank 8, alpha 16, dropout 0.05, attention projections as targets, learning rate 5e-5, batch 1, accumulation 8, and 3 epochs.

## Generation, blinded pairs, and judging

Create the 300-prompt seed-42 AlpacaEval subset once, then pass that exact file to every compared model. Generate the HH track from eval_sets/hh_rlhf_eval_300_seed42.json and the persona track from eval_sets/persona_style_prompts.json.

~~~bash
python -m scripts.make_eval_subset --source /kaggle/input/alpaca-eval/instructions.json --count 300 --seed 42 --output eval_sets/alpaca_eval_300_seed42.json
python -m scripts.generate --model Qwen/Qwen2.5-0.5B --prompts eval_sets/alpaca_eval_300_seed42.json --output /kaggle/working/outputs/generations/base.jsonl
~~~

Generation writes and flushes JSONL outputs after each batch. Use --resume to skip completed IDs. Keep generation settings equal across each SFT/DPO comparison. For the persona LoRA checkpoint, pass --use-chat-template to format prompts with the instruct model tokenizer template.

Create randomized response-order pairs before judging. Model identity is stored in metadata; only blind comparison prompts are sent to the judge.

~~~bash
python -m scripts.build_pairwise --sft /kaggle/working/outputs/generations/sft.jsonl --dpo /kaggle/working/outputs/generations/dpo.jsonl --sft-name sft --dpo-name dpo --output /kaggle/working/outputs/evaluation/pairs.jsonl
python -m scripts.judge_pairs --pairs /kaggle/working/outputs/evaluation/pairs.jsonl --output /kaggle/working/outputs/evaluation/judgments.jsonl --model google/gemini-3.6-flash --batch-size 5 --sleep-seconds 5 --max-retries 3
~~~

Run the generation and pair-judging commands from the Kaggle notebook. The all-in-one notebook selects an available Kaggle model-proxy judge, uses `scripts.judge_pairs_reliable`, batches five comparisons, waits at least five seconds between requests, retries up to three times, recovers malformed batches one case at a time, resumes completed comparisons, and reports win/loss/tie/unknown plus failure/retry rates.

Generate plots from the logged training metrics with `python -m scripts.plot_metrics`. The all-in-one notebook builds a report from its actual outputs; review the PDF before submission. Full model runs, downloads, and Kaggle judging require Kaggle and have not been run locally.

Kaggle sessions are temporary. Save important outputs under `/kaggle/working`, save notebook versions, and archive checkpoints/generations as zip files or Kaggle output datasets before ending a session.

## Completed Kaggle run and submission evidence

The all-in-one notebook completed the SFT, HH-RLHF DPO, UltraFeedback DPO, and persona LoRA runs on two Tesla T4 GPUs. All seven public adapter checks passed. The final checklist reports all 17 required artifact checks complete, including 300 resolved judgments for each full-model DPO comparison and eight resolved persona judgments.

| Comparison | Wins / ties / losses | Win rate with ties worth half |
| --- | ---: | ---: |
| HH DPO versus SFT | 154 / 68 / 78 | 62.7% |
| UltraFeedback DPO versus SFT | 105 / 70 / 125 | 46.7% |
| Persona LoRA versus base SmolLM2 | 1 / 6 / 1 | 50.0% |

The judge was `google/gemini-3-flash-preview`; all comparisons were resolved, with no unknown judgments. The [five-page final report](report/PA2_final_report.pdf), [evaluation summary](report/evaluation_summary.json), [plots](report/plots), and [final checklist](report/final_checklist.json) are included here. [PA2_evidence.zip](PA2_evidence.zip) contains the fixed evaluation sets, generated responses, blinded pairs, judgments, metrics, logs, plots, report, and source snapshot without large model weights. Best and final full-model checkpoints remain in the saved Kaggle notebook output.

The source commit named in the report is the base commit cloned at the start of the Kaggle job. The notebook applied the later source patches included in this repository and the evidence ZIP before training.
