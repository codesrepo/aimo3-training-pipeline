#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path

# ---- Hard-disable wandb & telemetry BEFORE any HF imports ----
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["TRANSFORMERS_NO_WANDB"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Helps fragmentation in long runs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig


# ----------------------------
# Paths (EDIT THESE)
LOCAL_MODEL_DIR = "/home/malam/wsl-tunix/imo/model/gpt-oss-120b-bnb4"
TRAIN_JSONL     = "/home/malam/wsl-tunix/imo/openmath_data/aimo_certs_207.jsonl"
#SFT_LORA_DIR    = "/content/drive/MyDrive/IT5002/lora_adapter_drgrpo/"
SFT_LORA_DIR = " "
OUT_DIR         = "/home/malam/wsl-tunix/imo/lora_grpo_adapter"

# ----------------------------
# Prompt blocks
# ----------------------------
SYSTEM_BLOCK = (
    "You are an IMO-level mathematical problem solver.\n"
    "Output must end with a non-negative integer in \\boxed{...} where 0<=answer<=99999.\n"
)

CERT_PROTOCOL = (
    "Follow this protocol (be concise):\n"
    "1) Constraints: bullet the key constraints and invariants.\n"
    "2) Plan: 2-4 lines.\n"
    "3) Certificate: state the invariant/lemma/formula/DP/state definition needed to verify.\n"
    "4) Check: if possible, run a small-case or arithmetic check and align it with the boxed answer.\n"
    "5) Final: \\boxed{answer}\n"
    "Avoid long exposition.\n"
)

def build_prompt(problem_text: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_BLOCK}{CERT_PROTOCOL}\n"
        f"<|user|>\n{problem_text}\n"
        f"<|assistant|>\n"
    )


# ----------------------------
# Dataset mapping
# ----------------------------
def _safe_str(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)

def to_row(ex):
    problem = _safe_str(ex.get("problem", "")).strip()
    gt = ex.get("final_answer", None)
    gt = str(gt).strip() if gt is not None else ""
    return {
        "prompt": build_prompt(problem),
        "ground_truth": gt,
    }


# ----------------------------
# Reward function (simple & fast)
# ----------------------------
BOX_RE = re.compile(r"\\boxed\{(\d+)\}")

def _extract_text(c):
    # TRL sometimes provides list-of-dicts; sometimes string
    if isinstance(c, list) and c and isinstance(c[0], dict) and "content" in c[0]:
        return c[0]["content"]
    return str(c)

def reward_fn(prompts, completions, **kwargs):
    # TRL passes dataset columns via kwargs
    gt_list = kwargs.get("ground_truth", [])
    texts = [_extract_text(c) for c in completions]

    rewards = []
    for text, gt in zip(texts, gt_list):
        m = BOX_RE.findall(text)
        if not m:
            rewards.append(-0.2)  # no boxed answer
            continue

        pred = m[-1]
        # exact match reward
        if pred == str(gt).strip():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


# ----------------------------
# Load model (bnb4) + LoRA
# ----------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        device_map={"": 0},              # <- force all on GPU0
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(model, SFT_LORA_DIR, is_trainable=True)
    model.config.use_cache = False
    return model, tokenizer


def main():
    model, tokenizer = load_model_and_tokenizer()

    ds = load_dataset("json", data_files=TRAIN_JSONL, split="train")
    ds = ds.map(to_row, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["prompt"]) > 0 and len(x["ground_truth"]) > 0)

    print("[TRL-GRPO] Dataset size:", len(ds))
    print("[TRL-GRPO] Prompt preview:\n", ds[0]["prompt"][:400])

    # Keep this small first (test end-to-end)
    NUM_GENERATIONS = 2
    MAX_PROMPT_LEN = 256
    MAX_COMP_LEN = 256

    args = GRPOConfig(
        output_dir=OUT_DIR,
        max_steps=10,                      # test run
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        logging_steps=1,
        save_steps=10,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LEN,
        max_completion_length=MAX_COMP_LEN,
        loss_type="grpo",                  # plain grpo first
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
    )

    print("\n" + "=" * 80)
    print("Starting TRL GRPO (no Unsloth patch)...")
    print("=" * 80)
    trainer.train()

    os.makedirs(OUT_DIR, exist_ok=True)
    trainer.model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("Saved GRPO LoRA adapter to:", OUT_DIR)


if __name__ == "__main__":
    main()
