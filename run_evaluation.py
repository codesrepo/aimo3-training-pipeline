#!/usr/bin/env python3
"""
Wrapper script to run evaluation on base model, SFT adapter, and GRPO adapter.

This script runs the evaluation three times:
1. Base model (no adapter)
2. SFT adapter
3. GRPO adapter

Compares results across all three model configurations.
"""

import os
import subprocess
import sys
from pathlib import Path

# Configuration
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "/data/models/gpt-oss-120b/transformers/default/1")
SFT_LORA_DIR = os.environ.get("SFT_LORA_DIR", "./lora_sft_adapter")
GRPO_LORA_DIR = os.environ.get("GRPO_LORA_DIR", "./lora_grpo_adapter")
EVAL_JSONL = os.environ.get("EVAL_JSONL", "./openmath_data/predictions_log_base.jsonl")
NUM_EXAMPLES = int(os.environ.get("NUM_EXAMPLES", "50"))

EVAL_SCRIPT = Path(__file__).parent / "evaluate_model.py"

print("="*80)
print("Full Model Evaluation Pipeline")
print("="*80)
print(f"BASE_MODEL_PATH: {BASE_MODEL_PATH}")
print(f"SFT_LORA_DIR: {SFT_LORA_DIR}")
print(f"GRPO_LORA_DIR: {GRPO_LORA_DIR}")
print(f"EVAL_JSONL: {EVAL_JSONL}")
print(f"NUM_EXAMPLES: {NUM_EXAMPLES}")
print("="*80)

# Verify eval script exists
if not EVAL_SCRIPT.exists():
    print(f"\n✗ Error: Evaluation script not found: {EVAL_SCRIPT}")
    sys.exit(1)

# Verify eval data exists
if not Path(EVAL_JSONL).exists():
    print(f"\n✗ Error: Evaluation data not found: {EVAL_JSONL}")
    sys.exit(1)

results = {}

# 1. Evaluate base model
print(f"\n{'='*80}")
print("1. Evaluating BASE MODEL")
print(f"{'='*80}")
env = os.environ.copy()
env["BASE_MODEL_PATH"] = BASE_MODEL_PATH
env["LORA_DIR"] = ""  # Empty = base model
env["EVAL_JSONL"] = EVAL_JSONL
env["NUM_EXAMPLES"] = str(NUM_EXAMPLES)

result = subprocess.run(
    [sys.executable, str(EVAL_SCRIPT)],
    env=env,
    capture_output=False
)

if result.returncode == 0:
    print("✓ Base model evaluation completed")
else:
    print("✗ Base model evaluation failed")
    sys.exit(1)

# 2. Evaluate SFT adapter
if Path(SFT_LORA_DIR).exists():
    print(f"\n{'='*80}")
    print("2. Evaluating SFT ADAPTER")
    print(f"{'='*80}")
    env = os.environ.copy()
    env["BASE_MODEL_PATH"] = BASE_MODEL_PATH
    env["LORA_DIR"] = SFT_LORA_DIR
    env["EVAL_JSONL"] = EVAL_JSONL
    env["NUM_EXAMPLES"] = str(NUM_EXAMPLES)
    
    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT)],
        env=env,
        capture_output=False
    )
    
    if result.returncode == 0:
        print("✓ SFT adapter evaluation completed")
    else:
        print("✗ SFT adapter evaluation failed")
else:
    print(f"\n⚠ Warning: SFT adapter directory not found: {SFT_LORA_DIR}")
    print("  Skipping SFT evaluation")

# 3. Evaluate GRPO adapter
if Path(GRPO_LORA_DIR).exists():
    print(f"\n{'='*80}")
    print("3. Evaluating GRPO ADAPTER")
    print(f"{'='*80}")
    env = os.environ.copy()
    env["BASE_MODEL_PATH"] = BASE_MODEL_PATH
    env["LORA_DIR"] = GRPO_LORA_DIR
    env["EVAL_JSONL"] = EVAL_JSONL
    env["NUM_EXAMPLES"] = str(NUM_EXAMPLES)
    
    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT)],
        env=env,
        capture_output=False
    )
    
    if result.returncode == 0:
        print("✓ GRPO adapter evaluation completed")
    else:
        print("✗ GRPO adapter evaluation failed")
else:
    print(f"\n⚠ Warning: GRPO adapter directory not found: {GRPO_LORA_DIR}")
    print("  Skipping GRPO evaluation")

print(f"\n{'='*80}")
print("Evaluation Pipeline Complete")
print(f"{'='*80}")
print("\nTo view detailed results, check the evaluation_results_*.json files in:")
print(f"  {Path(EVAL_JSONL).parent}/")

print("\nDone!")
