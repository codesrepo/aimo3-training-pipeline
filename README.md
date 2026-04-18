# AIMO3 Training Pipeline

Code for the AIMO3 (AI Mathematical Olympiad — Progress Prize 3) submission by Mohammad Shadab Alam (Kaggle: [outliar](https://www.kaggle.com/outliar)).

**Final result**: best public LB **42** | model: gpt-oss-120b MXFP4-baked at 700 ORPO steps.

---

## Repository Contents (15 scripts)

```
Stage 0 — Baseline          data_selection.py
Stage 1 — Data / SFT        cluster_high_mismatch.py
                             convert_to_harmony.py
                             sft_lora.py
Stage 2 — Oracle Traces     oracle_traces_aimo.py
                             aimo_gen_cert.py          (optional)
Stage 3 — GRPO              aimo_drgrpo_lora_r4_copy.py   ← used for submission
                             grpo_lora.py                  ← cleaned reference copy
Stage 4 — Evaluation        evaluate_model.py
                             run_evaluation.py             ← calls evaluate_model.py
Stage 5 — Community Data    get_kaggle_datasets.py
                             data_positive_negative_math.py
                             preprocess_training_samples_multi.py
Stage 6 — ORPO              train_dpo.py
Stage 7 — Merge / Bake      merge_lora_v2.py
```

---

## Stage-by-Stage Guide

### Stage 0 — Baseline scoring
**`data_selection.py`**
Runs gpt-oss-120b on a subset of the NVIDIA OpenMathReasoning dataset using Harmony
tool-augmented inference (greedy, T=0.0). Records each problem's predicted answer and
whether it matches the expected answer.

- **Reads**: OpenMathReasoning JSONL (249,341 integer-answer examples)
- **Writes**: `predictions_log_base.jsonl` — 662 scored problems, 53.0% match rate

---

### Stage 1 — Difficulty clustering + SFT warm-start

**`cluster_high_mismatch.py`**
Embeds the 662 scored problems with `all-MiniLM-L6-v2` and clusters them with K-Means
(k=15). Computes the per-cluster mismatch rate and selects clusters above the 0.59 threshold
as "hard". Saves the fitted model so downstream scripts can assign any new problem to a cluster.

- **Reads**: `predictions_log_base.jsonl`
- **Writes**:
  - `high_mismatch_clusters.jsonl` — 2,710 examples from Cluster 3 (68.12% mismatch rate)
  - `kmeans_model.pkl`, `cluster_stats.json` — used by `oracle_traces_aimo.py` to tag traces
  - Diagnostic plots: elbow curve, PCA scatter, mismatch histogram

**`convert_to_harmony.py`**
Parses the raw `generated_solution` text in each cluster record — extracts tool calls, tool
outputs, and the final assistant turn — and re-encodes everything as a Harmony multi-turn
message sequence (system → user → interleaved assistant/tool turns → `\boxed{answer}`).
This format is what the model sees during SFT and inference.

- **Reads**: `high_mismatch_clusters.jsonl`
- **Writes**: `high_mismatch_harmony.jsonl` — 2,710 Harmony-formatted training examples

**`sft_lora.py`**
Supervised fine-tuning warm-start. Trains a LoRA adapter (r=4, α=8) on the Harmony examples
so the model learns the expected tool-use dialogue structure before any reward signal is
introduced. Without this, early GRPO generations are dominated by format errors.

- **Reads**: `high_mismatch_harmony.jsonl`
- **Writes**: SFT LoRA adapter checkpoint
- **Key config**: base = gpt-oss-120b BnB4, LoRA r=4 α=8

---

### Stage 2 — Oracle trace collection

**`oracle_traces_aimo.py`**
Runs gpt-oss-120b via local vLLM on every no-match problem from `predictions_log.jsonl`,
allowing up to 8 tool-augmented attempts per problem. Keeps only traces where the predicted
answer matches the expected answer. Loads `kmeans_model.pkl` at startup to write
`predicted_cluster` and `cluster_mismatch_rate` into each retained trace — this label is
used for difficulty weighting during GRPO. Traces accumulate across training rounds.

- **Reads**: `predictions_log.jsonl`, `kmeans_model.pkl`, `cluster_stats.json`
- **Writes**:
  - `oracle_traces_parsed.jsonl` — 10,002 total correct traces (full OpenMathReasoning run)
  - `aimo_certs_151/173/207/219/243/283.jsonl` — iterative snapshots; 207 was used for GRPO

**`aimo_gen_cert.py`** *(optional post-processing)*
Sends each raw oracle trace through gpt-oss-120b to extract structured fields:
`key_idea`, `proof_skeleton`, `attainment_or_example`, `sanity_checks`, `solved_attempt`.
Useful for human review; the GRPO training can run with or without this step.

- **Reads**: any `aimo_certs_*.jsonl`
- **Writes**: `aimo_certs.jsonl` with structured certificate fields added

---

### Stage 3 — GRPO reinforcement learning

> **Two scripts exist for this stage — see note below.**

**`aimo_drgrpo_lora_r4_copy.py`** ← *this is the script that produced the submission model*
Implements Dr.GRPO (Group Relative Policy Optimisation with per-group advantage
normalisation) in two phases:

| Phase | Steps | Reward emphasis |
|-------|-------|----------------|
| A — Certificate-first | 10 | `\boxed{}` format, protocol headers, exact match, tool consistency, length penalty |
| B — Answer-first | 20 | Exact match (dominant weight), tool ok bonus, tool consistency, length penalty |

The cluster mismatch rate from Stage 2 is used to scale reward — problems from harder
clusters receive stronger signal.

- **Reads**: `aimo_certs_207.jsonl` (207 verified oracle traces)
- **Writes**: GRPO LoRA adapter (`lora_adapter_drgrpo_r4/`)
- **Key config**: LR=5e-6, KL β=0.001, G=2 generations per problem, batch=1, 30 total steps

**`grpo_lora.py`** ← *cleaned reference version; same logic, tidied for readability*
Identical training procedure. Use this as the starting point for re-running or adapting the
GRPO stage. `aimo_drgrpo_lora_r4_copy.py` is preserved as-is to exactly reproduce the
submission.

---

### Stage 4 — Evaluation

**`evaluate_model.py`**
Evaluates any combination of base model / SFT adapter / GRPO adapter on a held-out
50-problem test set drawn from `predictions_log_base.jsonl`. Reports exact-match accuracy
for each configuration by extracting `\boxed{}` answers.

- **Reads**: test problems, optionally SFT and GRPO adapter paths (env vars)
- **Writes**: accuracy report to stdout

**`run_evaluation.py`**
Wrapper that invokes `evaluate_model.py` three times — base only, base + SFT adapter,
base + GRPO adapter — and prints a side-by-side comparison table. Run this to regenerate
the three accuracy numbers reported in the writeup.

- **Depends on**: `evaluate_model.py`
- **Regenerates**: base 53.0% → SFT → GRPO 77.3% accuracy progression

---

### Stage 5 — Community dataset + DPO data preparation

**`get_kaggle_datasets.py`**
Downloads `ycchen/Crystal-Math-Preview`, combines it with the hard-50 benchmark CSV and the
oracle no-match traces, deduplicates on problem text, and filters to integer answers in
[0, 99999]. The result is the candidate pool for ORPO training.

- **Reads**: Kaggle public datasets + `hard_50_math_problems_set_v6.csv`
- **Writes**: `datasets/combined_math_crystal_hard50.jsonl` — 2,664 problems from 11 sources

**`data_positive_negative_math.py`**
Scores every problem in the candidate pool using gpt-oss-120b (MAX_ATTEMPTS=3, parallel
workers via vLLM). Records each attempt with `predicted_answer` and full response text,
labelling it correct or wrong. Designed to be resumable: already-scored problem IDs are
tracked in `parsed_tracking.jsonl`.

- **Reads**: `combined_math_crystal_hard50.jsonl`
- **Writes**: `datasets/scored_combined_math.jsonl` — 2,664 problems × up to 3 attempts each

**`preprocess_training_samples_multi.py`**
Converts scored attempts into ORPO preference pairs in Harmony message format:
- **Chosen**: last correct attempt, re-encoded as a Harmony message list; Python code blocks
  rewritten by vLLM to enforce correctness and add `print(answer)`.
- **Rejected**: each wrong attempt, also as a Harmony message list.
- Multiple (chosen, rejected) rows are emitted per problem when multiple wrong attempts exist.

> **Note on dataset evolution**: this script produced the *early* Harmony-trace format
> (382 pairs). The final submission used `training_samples_v25232026.jsonl` (1,415 pairs)
> which stores chosen/rejected as bare `\boxed{answer}` strings — see `train_dpo.py` note.

- **Reads**: `scored_combined_math.jsonl`
- **Writes**: `datasets/training_samples_multi.jsonl`

---

### Stage 6 — ORPO preference fine-tuning

**`train_dpo.py`**
Applies ORPO (Odds-Ratio Preference Optimisation) on top of the GRPO-trained adapter.
ORPO combines SFT loss on the chosen response with a preference loss in one forward pass —
no reference model required.

**Training signal (final version)**: each pair is `(problem, \boxed{correct}, \boxed{wrong})`.
Only the boxed answer strings are used as chosen/rejected — no solution trace is provided.
The model is calibrated on which numerical answer is preferred; it relies on reasoning
capabilities acquired in Stages 1–3. Earlier dataset versions used full Harmony traces as
chosen/rejected; the answer-only format proved more stable and gave better LB scores.

- **Reads**: `datasets/training_samples_v25232026.jsonl` (1,415 answer-only pairs)
- **Writes**: LoRA adapter checkpoints every 100 steps; best = checkpoint at 700 steps (i700)
- **Key config**: LR=5e-6, β=0.001, 1,000 steps, LoRA r=4 α=8 dropout=0.05, paged AdamW 8-bit

---

### Stage 7 — LoRA merge + MXFP4 baking

**`merge_lora_v2.py`**
Merges the ORPO LoRA adapter into the base MXFP4 weights using Unsloth's
`save_pretrained_merged` with `save_method="mxfp4"`. The merged checkpoint is ~61 GB
(vs ~218 GB for a naïve BF16 merge), fitting within the Kaggle H100 memory budget.
Run on 8× H100 80 GB.

- **Reads**: base MXFP4 model + ORPO LoRA adapter (i700 checkpoint)
- **Writes**: `merged_gpt_oss120b_v25032026/` — uploaded to Kaggle as a dataset

---

## Community Dataset Sources

| Source | Problems scored |
|--------|----------------|
| ODA-Math-460k | 659 |
| Untagged (Crystal-Math-Preview) | 634 |
| DAPO-17K | 561 |
| olympiads-ref-base | 258 |
| Nemotron-Math-V2 | 228 |
| PaCoRe-Train-8k | 133 |
| IMO-AnswerBench | 67 |
| Omni-MATH | 50 |
| OlymMATH | 25 |
| BeyondAIME | 22 |
| MathArena | 18 |
| Putnam-Axiom | 9 |
| **Total** | **2,664** |

---

## Hardware

| Task | Hardware |
|------|----------|
| Stages 0–6 (data + training) | RTX PRO 6000 Blackwell Max-Q (102 GB VRAM), local WSL2 |
| Stage 7 (MXFP4 baking) | 8× NVIDIA H100 80 GB (Kaggle notebook) |
| Inference (Kaggle submission) | 2× NVIDIA H100 80 GB |

## Requirements

```
unsloth
trl
peft
transformers>=4.46
vllm
sentence-transformers
scikit-learn
datasets
```
