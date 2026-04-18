#!/usr/bin/env python3
"""
LoRA SFT training script for GPT-OSS-120B.

Converts harmony format data (with generated_solution field) to messages format
and trains a LoRA adapter using QLoRA for memory efficiency.
"""

import os
import json
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch

import kagglehub

# Download latest version
#path = kagglehub.model_download("danielhanchen/gpt-oss-120b/transformers/default")
#print(path)
#xxx
# Configuration via environment variables
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "/home/malam/wsl-tunix/imo/data/gpt-oss-120b/transformers/default/1/")
TRAIN_JSONL = os.environ.get("TRAIN_JSONL", "/home/malam/wsl-tunix/imo/openmath_data/high_mismatch_harmony.jsonl")
OUT_DIR = os.environ.get("OUT_DIR", "./lora_sft_adapter")
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "2048"))

print("="*80)
print("LoRA SFT Training Configuration")
print("="*80)
print(f"BASE_MODEL_PATH: {BASE_MODEL_PATH}")
print(f"TRAIN_JSONL: {TRAIN_JSONL}")
print(f"OUT_DIR: {OUT_DIR}")
print(f"MAX_SEQ_LEN: {MAX_SEQ_LEN}")
print("="*80)

# Load tokenizer
print(f"\n[1/6] Loading tokenizer from {BASE_MODEL_PATH}...")
import torch
import json, os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

BASE_MODEL_PATH = "/home/malam/wsl-tunix/imo/data/gpt-oss-120b/transformers/default/1"
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# Tokenizer
tok = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    use_fast=False,            # safer for debugging; can switch to True later
    trust_remote_code=True
)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Model (MXFP4 checkpoint: do NOT pass BitsAndBytesConfig)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    device_map={"": 0},        # IMPORTANT: keep everything on GPU0 (no offload/meta)
    low_cpu_mem_usage=False,   # IMPORTANT: avoid meta-init path
)

# Align special tokens
model.config.pad_token_id = tok.pad_token_id
model.config.bos_token_id = tok.bos_token_id
model.config.eos_token_id = tok.eos_token_id
if getattr(model, "generation_config", None) is not None:
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.bos_token_id = tok.bos_token_id
    model.generation_config.eos_token_id = tok.eos_token_id

# Training stability
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# (optional) sanity: ensure no meta tensors
meta_params = [n for n,p in model.named_parameters() if p.device.type == "meta"]
meta_bufs   = [n for n,b in model.named_buffers()    if b.device.type == "meta"]
print("META params:", len(meta_params), "META bufs:", len(meta_bufs))
assert len(meta_params) == 0 and len(meta_bufs) == 0, "Still have meta tensors; cannot train."

# Configure LoRA
print(f"\n[3/6] Configuring LoRA adapter...")
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# Load and convert dataset
print(f"\n[4/6] Loading dataset from {TRAIN_JSONL}...")
if not Path(TRAIN_JSONL).exists():
    raise FileNotFoundError(f"Training data not found: {TRAIN_JSONL}")

def convert_harmony_to_messages(ex):
    """Convert harmony format to messages format expected by trainer.
    
    Harmony format: {"problem": "...", "expected_answer": "...", "generated_solution": [messages...]}
    Expected format: {"messages": [messages...], "expected_answer": "..."}
    """
    # If already in messages format, return as-is
    if "messages" in ex:
        return ex
    
    # Convert from harmony format
    messages = ex.get("generated_solution", [])
    if not messages:
        # Fallback: create basic structure if generated_solution is missing
        messages = [
            {"role": "system", "content": "You are the top International Mathematical Olympiad (IMO) competitor. The final answer must be a non-negative integer between 0 and 99999. You must place the final integer answer inside \\boxed{}."},
            {"role": "user", "content": ex.get("problem", "")}
        ]
    
    return {
        "messages": messages,
        "expected_answer": str(ex.get("expected_answer", ""))
    }

# Load dataset
ds = load_dataset("json", data_files=TRAIN_JSONL, split="train")
print(f"  ✓ Loaded {len(ds)} examples")

ds = ds.map(convert_harmony_to_messages)

# Normalize tool call schema (remove tool messages and tool_calls for SFT)
def normalize_messages_schema(ex):
    """Normalize messages to format compatible with chat template."""
    messages = ex["messages"]
    normalized = []
    for msg in messages:
        role = msg.get("role")
        # Keep system and user messages as-is
        if role in ["system", "user"]:
            normalized.append({"role": role, "content": msg.get("content", "")})
        # Skip tool messages entirely
        elif role == "tool":
            continue
        # For assistant messages, only keep those with content (skip tool_calls)
        elif role == "assistant":
            # Skip assistant messages with tool_calls (commentary channel)
            if "tool_calls" in msg or msg.get("channel") == "commentary":
                continue
            # Keep assistant messages with content (final answers)
            if "content" in msg and msg.get("content"):
                normalized.append({"role": "assistant", "content": msg.get("content", "")})
    return {"messages": normalized}

ds = ds.map(normalize_messages_schema)

# Convert messages to text using chat template
print(f"\n[5/6] Converting messages to text using chat template...")
def to_text(ex):
    """Convert messages to formatted text using tokenizer chat template."""
    text = tokenizer.apply_chat_template(
        ex["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

# IMPORTANT: only remove columns AFTER to_text has created "text"
ds = ds.map(to_text)
ds = ds.remove_columns([c for c in ds.column_names if c != "text"])

print(f"  ✓ Converted {len(ds)} examples to text format")
print(f"  Sample text length: {len(ds[0]['text'])} characters")

# Training arguments
print(f"\n[6/6] Setting up training arguments...")
args = TrainingArguments(
    output_dir="./sft_runs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=500,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to="none",
    save_total_limit=3,
)
import transformers, accelerate, peft, trl
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("peft:", peft.__version__)
print("trl:", trl.__version__)
# Create trainer
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,  # Changed from tokenizer= for trl >= 0.12.0
    train_dataset=ds,
    formatting_func=lambda ex: ex["text"],
    #max_seq_length=MAX_SEQ_LEN,
    args=args,
    #packing=True,
)

# Train
print(f"\n{'='*80}")
print("Starting SFT Training")
print(f"{'='*80}")
trainer.train()

# Save adapter
print(f"\n{'='*80}")
print("Saving LoRA adapter")
print(f"{'='*80}")
os.makedirs(OUT_DIR, exist_ok=True)
trainer.model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print(f"\n✓ Saved LoRA adapter to: {OUT_DIR}")
print(f"  Files saved:")
for f in Path(OUT_DIR).iterdir():
    if f.is_file():
        print(f"    - {f.name}")

print("\nDone!")
