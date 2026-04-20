# AIMO3 Training Pipeline

Code for the AIMO3 (AI Mathematical Olympiad — Progress Prize 3) submission by Mohammad Shadab Alam (Kaggle: [outliar](https://www.kaggle.com/outliar)).

**Final result**: best public LB **42** / public rank **1,799 of 4,138** | model: gpt-oss-120b MXFP4-baked at 700 ORPO steps.

---

## Training Pipeline Overview

The pipeline adapts a frozen 120B base model through four successive phases, each targeting a specific failure mode identified during exploration.

![Training Pipeline](pipeline_diagram.png)

| Phase | Why it exists |
|-------|--------------|
| **SFT warm-start** | The base model cannot reliably follow the Harmony tool-use dialogue format on hard problems. SFT exposes it to 2,710 problems it could not solve, teaching the expected structure before any reward signal is introduced. |
| **DRGRPO** | Reinforcement learning on ~30 oracle-verified traces per iteration teaches the model *how to reason* through hard problems. Without this phase the model can produce plausible-looking traces that converge on the wrong answer. Each GRPO iteration took up to 30 min on H100 and longer on the RTX PRO 6000. |
| **ORPO** | For problems the model can already solve, competing incorrect reasoning paths sometimes pull the initial attention weights toward the wrong answer. ORPO presents (correct answer, wrong answer) pairs drawn from 11 competition sources so the model's starting distribution aligns with the right answer before any reasoning begins — without retraining the reasoning chains. |
| **MXFP4 baking** | Merges all adapters and quantises to MXFP4 (~61 GB vs ~218 GB BF16) so the 120B model fits a single Kaggle H100. |

> **Initial explorations** (Stages 0–1): used to probe sensitivity of LB score to prompt format, temperature, and generation count — not a training phase, but guided the hyperparameter choices for all later stages.

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

**Why**: Establishes which problems the base model can already solve (53.0% on the evaluated subset) so we can target fine-tuning exclusively at problems it fails on — avoiding capacity waste on easy examples.

**`data_selection.py`**
Runs gpt-oss-120b on a subset of the NVIDIA OpenMathReasoning dataset using Harmony
tool-augmented inference (greedy, T=0.0). Records each problem's predicted answer and
whether it matches the expected answer.

- **Reads**: OpenMathReasoning JSONL (249,341 integer-answer examples)
- **Writes**: `predictions_log_base.jsonl` — 662 scored problems, 53.0% match rate

---

### Stage 1 — Difficulty clustering + SFT warm-start

**Why**: The base model cannot reliably follow the Harmony multi-turn tool-use format on problems it has not seen in that format. Without an SFT warm-start, early GRPO rollouts are dominated by format errors rather than reasoning errors, and the reward signal collapses. SFT on 2,710 high-difficulty problems (the model's failure region) fixes the format and ensures GRPO can focus on *what to reason*, not *how to format*.

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

**Why**: GRPO requires verified correct trajectories as training examples. Only 30 high-quality oracle traces were used per GRPO iteration — a deliberate choice: each iteration is a single-sample forward pass (batch=1), and even this sparse signal produced a +24.3 pp accuracy gain. Running each iteration on an H100 took up to 30 minutes; on the RTX PRO 6000 Blackwell (local) it took longer.

**`oracle_traces_aimo.py`**
Runs gpt-oss-120b via local vLLM on every no-match problem from `predictions_log.jsonl`,
allowing up to 8 tool-augmented attempts per problem. Keeps only traces where the predicted
answer matches the expected answer. Loads `kmeans_model.pkl` at startup to annotate each retained trace with
`predicted_cluster` and `cluster_mismatch_rate` fields for provenance tracking.
Traces accumulate across training rounds.

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

**Why**: After SFT the model knows the format but still fails on hard reasoning. DRGRPO provides a direct reward signal: generate candidate solutions, compare to the verified answer, and reinforce correct reasoning chains. Only ~30 problems were used per iteration — this is intentional. The GRPO signal is strong and sparse; more problems per step would have required substantially longer per-iteration runtime (already up to 30 min on H100) without clear benefit on the hard problems targeted.

**`aimo_drgrpo_lora_r4_copy.py`** ← *this is the script that produced the submission model*
Implements Dr.GRPO (Group Relative Policy Optimisation with per-group advantage
normalisation) in two phases:

| Phase | Steps | Reward emphasis | Why |
|-------|-------|----------------|-----|
| A — Certificate-first | 10 | `\boxed{}` format, protocol headers, exact match, tool consistency, length penalty | Reinforce correct structure before optimising for answer correctness |
| B — Answer-first | 20 | Exact match (dominant weight), tool ok bonus, tool consistency, length penalty | Shift emphasis to answer correctness once format is stable |

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

**Why**: ORPO needs preference pairs covering the specific error types the model makes when it *can* solve a problem but picks the wrong answer. Community competition datasets (olympiads, AMC/AIME-style) provide diverse examples of these near-miss errors. CrystalMath-Preview (Yi-Chia Chen, [ycchen/Crystal-Math-Preview](https://huggingface.co/datasets/ycchen/Crystal-Math-Preview)) contributed 2,129 problems in raw form; after filtering to integer answers with non-null solutions, answers in [0, 99999], and deduplication, **634 problems** from CrystalMath survived into the 2,664-problem pool. The hard-50 benchmark used for checkpoint selection comes from ZFTurbo's [hard-math-problems-for-aimo-3](https://www.kaggle.com/datasets/zfturbo/hard-math-problems-for-aimo-3) dataset.

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

**Why**: After DRGRPO the model reasons correctly on most hard problems, but on problems it can solve without tool calling it sometimes drifts toward a wrong answer when competing candidates have similar surface plausibility. ORPO aligns the model's initial answer distribution — showing it which numerical answer is correct vs incorrect for 1,415 competition problems — without altering the reasoning chains that DRGRPO built. The answer-only format (no solution trace in chosen/rejected) keeps the preference pairs compact and avoids introducing hallucinated reasoning.

**Checkpoint selection** (hard-50 benchmark + public LB elimination):

| Checkpoint | Steps | Public LB | Hard-50 | Decision |
|------------|-------|-----------|---------|---------|
| i500 | 500 | Lower mean vs i700 | — | Eliminated on public LB score |
| **i700** | **700** | **Best** | **Best score, acceptable runtime** | **Selected** |
| i1000 | 1,000 | Marginal degradation | Slower, no gain | Eliminated |

> **Training note**: i700 was not trained from scratch to 700 steps. The i500 checkpoint was used as the starting point and training continued for an additional 200 steps to produce i700 — each 100-step block saved a checkpoint. The final i500 checkpoint was ruled out first using public LB scores; among remaining checkpoints, i700 outperformed i1000 on the hard-50 benchmark within acceptable runtime, making it the clear choice.

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
- **Key config**: LR=5e-6, β=0.001, 1,000 steps, LoRA r=4 α=8 dropout=0.05, paged AdamW 8-bit, **batch=1 (single sample per iteration)**

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
| CrystalMath-Preview (filtered) | 634 |
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

> CrystalMath-Preview contains 2,129 problems in raw form. After filtering to integer answers with non-null solutions, answers in [0, 99999], deduplication on problem text, and removal of problems with Chinese characters, 634 problems survived into the pool. Credits: Yi-Chia Chen ([ycchen](https://www.kaggle.com/threerabbits)) for [CrystalMath-Preview](https://huggingface.co/datasets/ycchen/Crystal-Math-Preview); ZFTurbo ([zfturbo](https://www.kaggle.com/zfturbo)) for the [hard-50 benchmark](https://www.kaggle.com/datasets/zfturbo/hard-math-problems-for-aimo-3) used for checkpoint selection.

---

## Hardware

| Task | Hardware |
|------|----------|
| Stages 0–6 (data + training) | RTX PRO 6000 Blackwell Max-Q (96 GB VRAM), local WSL2 |
| Stage 7 (MXFP4 baking) | 8× NVIDIA H100 80 GB (Kaggle notebook) |
| Inference (Kaggle submission) | 1× NVIDIA H100 80 GB (tensor-parallel-size 1) |

---

## Acknowledgements

- **Yi-Chia Chen** ([ycchen](https://www.kaggle.com/threerabbits)) — [CrystalMath-Preview](https://huggingface.co/datasets/ycchen/Crystal-Math-Preview): 2,129 verified, high-difficulty contest math problems for RLVR training. 634 problems from this dataset contributed to our ORPO training pool after filtering.
- **ZFTurbo** ([zfturbo](https://www.kaggle.com/zfturbo)) — [hard-math-problems-for-aimo-3](https://www.kaggle.com/datasets/zfturbo/hard-math-problems-for-aimo-3): hard-50 benchmark used throughout for checkpoint evaluation and model selection.

---

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
