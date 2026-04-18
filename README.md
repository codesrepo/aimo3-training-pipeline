# AIMO3 Training Pipeline

Code for the AIMO3 (AI Mathematical Olympiad — Progress Prize 3) submission by Mohammad Shadab Alam (Kaggle: [outliar](https://www.kaggle.com/outliar)).

Final result: best public LB **42**, MXFP4-baked gpt-oss-120b at 700 ORPO steps.

---

## Pipeline Overview

```
OpenMathReasoning (249,341)
        │
        ▼
cluster_high_mismatch.py        ← embed + K-Means (k=15), select Cluster 3 (68.12% mismatch)
        │  high_mismatch_clusters.jsonl (2,710 problems)
        ▼
convert_to_harmony.py           ← parse oracle solutions → Harmony multi-turn message format
        │  high_mismatch_harmony.jsonl
        ▼
sft_lora.py                     ← SFT warm-start on hard examples (LoRA r=4)
        │
        ▼
oracle_traces_aimo.py           ← vLLM multi-attempt inference; keep correct traces; tag with cluster ID
  +                               iterative accumulation: 151→173→207→219→243→283 certs
aimo_gen_cert.py                ← structured certificate extraction (key_idea, proof_skeleton, sanity_checks)
        │  aimo_certs_207.jsonl (207 verified traces used for GRPO)
        ▼
grpo_lora.py                    ← Dr.GRPO two-phase RL (Phase A 10 steps + Phase B 20 steps)
        │
        ▼
get_kaggle_datasets.py          ← download 11 community math datasets → 2,664 problems
        │
        ▼
data_positive_negative_math.py  ← vLLM scoring (MAX_ATTEMPTS=3) → correct + wrong attempts per problem
        │  scored_combined_math.jsonl (2,664 scored problems)
        ▼
preprocess_training_samples_multi.py  ← build DPO pairs: chosen = last correct Harmony trace (LLM python-fix),
        │                               rejected = each wrong Harmony trace
        │  training_samples_multi.jsonl (382 → 856 → 1,416 → 1,415 pairs across iterations)
        ▼
train_dpo.py                    ← ORPO preference fine-tuning (1,000 steps, LoRA r=4, β=0.001)
        │  best checkpoint: i700 (700 steps)
        ▼
[merge_lora + MXFP4 baking]     ← Unsloth save_pretrained_merged, mxfp4 format, ~61 GB
        ▼
Kaggle H100 inference           ← vLLM + Harmony tool-augmented self-consistency
```

---

## Scripts

### Data Preparation

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `cluster_high_mismatch.py` | Embed 662 scored problems with `all-MiniLM-L6-v2`, apply K-Means (k=15), identify high-mismatch clusters (≥0.59 mismatch rate). Saves the fitted model for downstream cluster assignment. | `predictions_log_base.jsonl` (base model scores) | `high_mismatch_clusters.jsonl` (2,710 problems, Cluster 3 at 68.12%), `kmeans_model.pkl`, `cluster_stats.json` |
| `convert_to_harmony.py` | Parse each cluster example's `generated_solution` text — extract tool calls, tool outputs, final answer — and re-encode as a Harmony multi-turn message sequence for SFT. | `high_mismatch_clusters.jsonl` | `high_mismatch_harmony.jsonl` |
| `oracle_traces_aimo.py` | Run gpt-oss-120b via local vLLM on failed problems; keep only traces where predicted answer matches expected. Loads the k-means model to tag each trace with `predicted_cluster` + `cluster_mismatch_rate`. Iteratively accumulates correct traces across rounds. | `predictions_log.jsonl` (no-match problems), `kmeans_model.pkl` | `oracle_traces_parsed.jsonl` (10,002 total), `aimo_certs_{151,173,207,219,243,283}.jsonl` |
| `aimo_gen_cert.py` | Optional LLM pass over raw oracle traces to extract structured certificate fields: `key_idea`, `proof_skeleton`, `attainment_or_example`, `sanity_checks`, `solved_attempt`. | Raw oracle traces | `aimo_certs.jsonl` |
| `get_kaggle_datasets.py` | Download `ycchen/Crystal-Math-Preview` and combine with the hard-50 benchmark and oracle no-match traces. Deduplicates on problem text, filters answers to [0, 99999]. | Kaggle public datasets | `combined_math_crystal_hard50.jsonl` (2,664 problems) |
| `data_positive_negative_math.py` | Score all community problems with gpt-oss-120b (MAX_ATTEMPTS=3, parallel vLLM). Records every attempt with `predicted_answer` and full response text. Resumable via `parsed_tracking.jsonl`. | `combined_math_crystal_hard50.jsonl` | `scored_combined_math.jsonl` |
| `preprocess_training_samples_multi.py` | Build ORPO/DPO preference pairs. Chosen = last correct attempt encoded as Harmony messages with LLM-rewritten Python code. Rejected = each wrong attempt as Harmony messages. Multiple rejected rows per problem when multiple wrong attempts exist. | `scored_combined_math.jsonl` | `training_samples_multi.jsonl` (1,415 pairs in final version) |

### Training

| Script | Purpose | Input | Key hyperparameters |
|--------|---------|-------|---------------------|
| `sft_lora.py` | Supervised fine-tuning warm-start. Teaches Harmony tool-use format before RL to prevent format errors dominating early GRPO reward. | `high_mismatch_harmony.jsonl` (2,710 examples) | LoRA r=4 α=8, base=gpt-oss-120b BnB4 |
| `aimo_drgrpo_lora_r4_copy.py` | Dr.GRPO two-phase RL (actual script used for submission). Phase A (10 steps): certificate-first rewards. Phase B (20 steps): answer-first rewards. Cluster mismatch rate used for difficulty weighting. | `aimo_certs_207.jsonl` (207 oracle traces) | LR=5e-6, β=0.001, G=2, batch=1 |
| `grpo_lora.py` | Cleaned reference version of the GRPO training script. | `aimo_certs_207.jsonl` | Same as above |
| `train_dpo.py` | ORPO preference fine-tuning on top of GRPO adapter. **Training signal**: answer-only pairs — prompt = problem, chosen = `\boxed{correct}`, rejected = `\boxed{wrong}`. No solution trace is provided; model is calibrated on final answer preference. Best checkpoint at 700 steps. | `training_samples_v25232026.jsonl` (1,415 answer-only pairs) | LR=5e-6, β=0.001, 1000 steps, LoRA r=4 α=8 |
| `merge_lora_v2.py` | Merge LoRA adapter into base MXFP4 weights using Unsloth `save_pretrained_merged` with `save_method="mxfp4"`. Produces ~61 GB merged checkpoint. | Base MXFP4 model + LoRA adapter | `merged_gpt_oss120b_v25032026/` |

### Evaluation (Number Regeneration)

| Script | Purpose | Regenerates |
|--------|---------|-------------|
| `data_selection.py` | Score gpt-oss-120b on OpenMathReasoning subset via Harmony vLLM inference. Produces per-problem predictions with match/no-match labels. | `predictions_log_base.jsonl` → 662 problems, 53.0% match rate |
| `evaluate_model.py` | Evaluate base model, SFT adapter, and GRPO adapter on 50-problem test set. Reports exact-match accuracy for each configuration. | Per-adapter accuracy numbers in Section 4 |
| `run_evaluation.py` | Wrapper that runs `evaluate_model.py` three times (base / SFT / GRPO) and prints a comparison table. | All three accuracy numbers in one run |

---

## Community Dataset Sources (Chart 4)

| Source | Problems |
|--------|----------|
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

- **Training**: RTX PRO 6000 Blackwell Max-Q (102 GB VRAM), local WSL2
- **MXFP4 baking**: 8× NVIDIA H100 80 GB
- **Inference (Kaggle)**: 2× NVIDIA H100 80 GB

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
