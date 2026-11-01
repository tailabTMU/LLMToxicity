# Toxicity detector evaluation

Unified evaluation of toxicity and safety detectors (Detoxify and LLM guardrails) across multiple datasets (ToxicChat, Aegis, MindGuard, hate-speech-spain, and others).

**Pipeline:** inference notebooks produce scored Parquet files → `detoxify_unified_evaluation_with_f2.py` computes metrics → optional qualitative error analysis.

Datasets and scored Parquet outputs are **not bundled** in this repo. Run the matching inference notebook first (typically in Colab), then point the CLI tools at your saved outputs.

## Repository layout


| Path                                                                                                                                       | Role                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------- |
| `[src/inference/BERT_based_models/](src/inference/BERT_based_models/)`                                                                     | Detoxify scoring notebooks                  |
| `[src/inference/LLM_based_guardrails/](src/inference/LLM_based_guardrails/)`                                                               | Llama Guard / MindGuard guardrail notebooks |
| `[src/quantitative_evaluation/detoxify_unified_evaluation_with_f2.py](src/quantitative_evaluation/detoxify_unified_evaluation_with_f2.py)` | Main metrics CLI                            |
| `[src/quantitative_evaluation/run_evaluation.ipynb](src/quantitative_evaluation/run_evaluation.ipynb)`                                     | Evaluation run examples (Colab)             |
| `[src/qualitative_evaluation/](src/qualitative_evaluation/)`                                                                               | FP/FN and parse-error analysis              |


## Setup

From the repo root:

```bash
pip install -r requirements.txt
```

Inference notebooks require additional packages (`torch`, `transformers`, `datasets`, etc.) installed inside the notebook; these are not listed in `[requirements.txt](requirements.txt)`.

## Step 1 — Inference

Open the notebook for your dataset, run top-to-bottom, and save Parquet outputs locally.

**BERT / Detoxify** (`[src/inference/BERT_based_models/](src/inference/BERT_based_models/)`):


| Notebook                                                    | Dataset                          |
| ----------------------------------------------------------- | -------------------------------- |
| `toxic-chat.ipynb`                                          | ToxicChat                        |
| `nemotron-content-safety_v2.ipynb`                          | Aegis 2.0                        |
| `mind-guard-testset.ipynb`, `mind_guard_testset_bert.ipynb` | MindGuard testset                |
| `hate-speech-spain_dataset.ipynb`                           | hate-speech-spain (hateRADAR-es) |


**LLM guardrails** (`[src/inference/LLM_based_guardrails/](src/inference/LLM_based_guardrails/)`):


| Notebook                              | Dataset                        |
| ------------------------------------- | ------------------------------ |
| `toxic-chat_guardrails.ipynb`         | ToxicChat + Llama Guard        |
| `Aegis_guardrails.ipynb`              | Aegis + guardrails             |
| `mind_guard_testset_guardrails.ipynb` | MindGuard + guardrails         |
| `hate_speech_spain_guardrails.ipynb`  | hate-speech-spain + guardrails |


## Step 2 — Quantitative evaluation

`[detoxify_unified_evaluation_with_f2.py](src/quantitative_evaluation/detoxify_unified_evaluation_with_f2.py)` reads Parquet or CSV with binary `label` (0/1) and score columns (`detox_`*, `guard_unsafe_score`, or `unsafe_score`). It tunes a threshold on a dev set (or an auto-split from `--test`) and reports AUROC, AUPRC, F1, and F2.

Dataset-specific commands are in `[run_evaluation.ipynb](src/quantitative_evaluation/run_evaluation.ipynb)`. Representative examples:

```bash
# ToxicChat (test + dev directories)
python src/quantitative_evaluation/detoxify_unified_evaluation_with_f2.py \
  --test path/to/detoxify_scores_test \
  --dev path/to/detoxify_scores_train \
  --out-dir detoxify_scores_results \
  --target-recall 0.95 \
  --n-balance-repeats 30

# MindGuard (no external dev; auto-split from test)
python src/quantitative_evaluation/detoxify_unified_evaluation_with_f2.py \
  --test path/to/mindguard_detoxify_scores_train_user_message \
  --out-dir mindguard_detoxify_outputs_results \
  --target-recall 0.95 \
  --n-balance-repeats 30 \
  --dev-fraction-when-missing 0.3
```

List all flags:

```bash
python src/quantitative_evaluation/detoxify_unified_evaluation_with_f2.py --help
```

By default, CSV metrics are written under `--out-dir` (default `safety_eval_outputs`). Use `--no-save` to print metrics only.

## Step 3 — Qualitative evaluation

**Common FP/FN** — rows where Detoxify, Llama Guard, and MindGuard-4B all share the same false positive or false negative (`[common_fp_fn.py](src/qualitative_evaluation/common_fp_fn.py)`). Threshold and column args are required; see `--help`.

```bash
python src/qualitative_evaluation/common_fp_fn.py \
  --detoxify-parquet path/to/detoxify.parquet \
  --llama-parquet path/to/llama.parquet \
  --mindguard-parquet path/to/mindguard.parquet \
  --output-dir path/to/output
```

**Parse errors** — count rows where a guardrail label column is `parse_error` (`[exploretory_for_LLM_guardrails.py](src/qualitative_evaluation/exploretory_for_LLM_guardrails.py)`):

```bash
python src/qualitative_evaluation/exploretory_for_LLM_guardrails.py path/to/guardrail.parquet
```

