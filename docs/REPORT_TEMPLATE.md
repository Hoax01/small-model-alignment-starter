# PA2 Final Report Template

The report must be no more than 8 pages including figures, tables, and references. Replace brackets with measured values after the Kaggle experiments. Do not infer response quality from training loss alone.

## 1. Repository, models, and data

- Repository URL: [URL]
- Base checkpoint: Qwen/Qwen2.5-0.5B
- Dataset sources, splits, selected train/validation counts for SFT, HH, UltraFeedback, and persona: [details]
- Fixed evaluation sets and seeds: [details]
- Filtering and prompt formatting choices: [details]
- Paths to configs, checkpoints, metrics, plots, generations, pairwise files, and judgments: [paths]

## 2. Experimental configuration

| Track | Trainable parameters | Optimizer, LR, schedule, warmup, weight decay, clip | Batch, accumulation, effective global batch, precision | Max sequence/prompt length, epochs/steps, seed | Beta and reference |
|---|---|---|---|---|---|
| SFT | [full fine-tune] | [ ] | [ ] | [ ] | - |
| HH DPO | [full fine-tune] | [ ] | [ ] | [ ] | [beta, exact SFT checkpoint] |
| UltraFeedback DPO | [full fine-tune] | [ ] | [ ] | [ ] | [beta, exact SFT checkpoint] |
| Persona LoRA-DPO | [rank, alpha, dropout, target modules] | [ ] | [ ] | [ ] | [beta, exact base/reference] |

Prompt template: ### Instruction: {instruction} ### Response:

Generation settings used for every comparison: temperature [ ], top-p [ ], max new tokens [ ], repetition/length penalties [ ], EOS behavior [ ], prompt/chat template [ ].

## 3. Parameter choices and deviations

| Setting | Guide value | Final value | Reason and evidence | Observed effect |
|---|---|---|---|---|
| [setting] | [value] | [value] | [evidence] | [effect] |

Separate resource-driven changes from performance-driven changes.

## 4. Training behavior

Include labeled SFT training/validation loss and DPO loss, reward accuracy, and reward margin plots. Explain convergence, instability, overfitting, and checkpoint selection.

## 5. DPO-versus-SFT results

| Evaluation set | DPO track | Judge model | N | Wins | Ties | Losses | Unknown | Failures | Failure rate | Win rate excluding ties | Win rate with ties half |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| [HH set] | HH DPO | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| [AlpacaEval seed 42] | UltraFeedback DPO | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| [Persona style prompts] | Persona LoRA-DPO | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

Report win rate with and without ties, invalid judgments, retry counts, and failure rate. Keep prompt order, decoding settings, and judging procedure fixed for each SFT/DPO pair.

## 6. Interpretation and error analysis

State which checkpoint performed better on each set, by how much, and what evidence supports the explanation. Include representative improvements, regressions, and unchanged cases. Separate observations from interpretation.

## 7. Limitations and reproducibility

Describe judge bias, evaluation size, seed variance, failed runs, and compute limits. List paths to configs, metrics, plots, generations, blinded pairs, and judged outputs.