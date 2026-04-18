#!/usr/bin/env python3
"""
Evaluation script for base model, SFT adapter, and GRPO adapter.

Evaluates on top 50 records from predictions_log_base.jsonl and reports
accuracy metrics based on exact answer matching from \\boxed{} format.
"""

import os
import json
import re
import torch
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# Configuration via environment variables
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "/data/models/gpt-oss-120b/transformers/default/1")
LORA_DIR = os.environ.get("LORA_DIR", None)  # None for base model, path for adapter
EVAL_JSONL = os.environ.get("EVAL_JSONL", "./openmath_data/predictions_log_base.jsonl")
NUM_EXAMPLES = int(os.environ.get("NUM_EXAMPLES", "50"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))

# Fixed system prompts (matching training format)
SYSTEM_PROMPTS = [
    "You are the top International Mathematical Olympiad (IMO) competitor. The final answer must be a non-negative integer between 0 and 99999. You must place the final integer answer inside \\boxed{}.",
    "Use this tool to execute Python code. The environment is a stateful Jupyter notebook. You must use print() to output results.",
    "You have access to `math`, `numpy` and `sympy` to solve the problem."
]


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the integer from the last \\boxed{...} occurrence in text."""
    matches = re.findall(r"\\boxed\{(\d+)\}", text)
    if not matches:
        return None
    return matches[-1]  # Return last occurrence


def load_eval_data(jsonl_path: str, num_examples: int = 50) -> List[Dict]:
    """Load evaluation data from predictions_log_base.jsonl."""
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            if line.strip():
                ex = json.loads(line)
                examples.append({
                    "problem": ex.get("problem", ""),
                    "expected_answer": str(ex.get("expected_answer", "")),
                    "idx": ex.get("idx", i)
                })
    return examples


def format_prompt(problem: str) -> str:
    """Format problem as a prompt with system messages."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[0]},
        {"role": "system", "content": SYSTEM_PROMPTS[1]},
        {"role": "system", "content": SYSTEM_PROMPTS[2]},
        {"role": "user", "content": problem}
    ]
    return messages


def evaluate_model(model, tokenizer, eval_examples: List[Dict], device: str) -> Dict:
    """Evaluate model on examples and return metrics."""
    correct = 0
    total = len(eval_examples)
    extraction_failures = 0
    results = []
    
    print(f"\nEvaluating on {total} examples...")
    
    for ex in tqdm(eval_examples, desc="Evaluating"):
        problem = ex["problem"]
        expected_answer = str(ex["expected_answer"])
        
        # Format prompt
        messages = format_prompt(problem)
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract completion (text after the prompt)
        prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        completion_tokens = outputs[0][prompt_len:]
        completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        
        # Extract answer
        predicted_answer = extract_boxed_answer(completion_text)
        
        # Check if answer was extracted
        if predicted_answer is None:
            extraction_failures += 1
            is_correct = False
        else:
            is_correct = (predicted_answer == expected_answer)
            if is_correct:
                correct += 1
        
        results.append({
            "idx": ex["idx"],
            "problem": problem[:100] + "..." if len(problem) > 100 else problem,
            "expected_answer": expected_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "extraction_success": predicted_answer is not None,
            "completion": completion_text[:200] + "..." if len(completion_text) > 200 else completion_text
        })
    
    accuracy = correct / total if total > 0 else 0.0
    extraction_rate = (total - extraction_failures) / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "extraction_failures": extraction_failures,
        "extraction_rate": extraction_rate,
        "results": results
    }


def main():
    """Main evaluation function."""
    print("="*80)
    print("Model Evaluation")
    print("="*80)
    
    # Determine model type
    model_type = "base"
    if LORA_DIR and Path(LORA_DIR).exists():
        model_type = "adapter"
        adapter_path = LORA_DIR
    else:
        adapter_path = None
    
    print(f"BASE_MODEL_PATH: {BASE_MODEL_PATH}")
    if adapter_path:
        print(f"LORA_DIR: {adapter_path}")
        print(f"Model Type: LoRA Adapter")
    else:
        print(f"Model Type: Base Model (no adapter)")
    print(f"EVAL_JSONL: {EVAL_JSONL}")
    print(f"NUM_EXAMPLES: {NUM_EXAMPLES}")
    print(f"MAX_NEW_TOKENS: {MAX_NEW_TOKENS}")
    print("="*80)
    
    # Check if eval file exists
    if not Path(EVAL_JSONL).exists():
        print(f"\n✗ Error: Evaluation file not found: {EVAL_JSONL}")
        return
    
    # Load evaluation data
    print(f"\n[1/4] Loading evaluation data from {EVAL_JSONL}...")
    eval_examples = load_eval_data(EVAL_JSONL, NUM_EXAMPLES)
    print(f"  ✓ Loaded {len(eval_examples)} examples")
    
    # Load tokenizer
    print(f"\n[2/4] Loading tokenizer from {BASE_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  ✓ Tokenizer loaded")
    
    # Load model
    print(f"\n[3/4] Loading model...")
    if adapter_path:
        print(f"  Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        print(f"  Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("  ✓ Model with LoRA adapter loaded")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        print("  ✓ Base model loaded")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Evaluate
    print(f"\n[4/4] Running evaluation...")
    metrics = evaluate_model(model, tokenizer, eval_examples, device)
    
    # Print results
    print(f"\n{'='*80}")
    print("Evaluation Results")
    print(f"{'='*80}")
    print(f"Model Type: {model_type.upper()}")
    if adapter_path:
        print(f"Adapter: {adapter_path}")
    print(f"\nMetrics:")
    print(f"  Total Examples: {metrics['total']}")
    print(f"  Correct: {metrics['correct']}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Answer Extraction Rate: {metrics['extraction_rate']:.2%}")
    print(f"  Extraction Failures: {metrics['extraction_failures']}")
    
    # Show sample results (first 5)
    print(f"\n{'='*80}")
    print("Sample Results (first 5)")
    print(f"{'='*80}")
    for i, result in enumerate(metrics['results'][:5]):
        status = "✓" if result['is_correct'] else "✗"
        ext_status = "✓" if result['extraction_success'] else "✗"
        print(f"\nExample {i+1} (idx={result['idx']}):")
        print(f"  Status: {status} | Extraction: {ext_status}")
        print(f"  Expected: {result['expected_answer']} | Predicted: {result['predicted_answer']}")
        print(f"  Problem: {result['problem'][:80]}...")
        print(f"  Completion: {result['completion'][:100]}...")
    
    # Save detailed results
    output_path = Path(EVAL_JSONL).parent / f"evaluation_results_{model_type}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model_type": model_type,
            "adapter_path": str(adapter_path) if adapter_path else None,
            "metrics": {
                "accuracy": metrics['accuracy'],
                "correct": metrics['correct'],
                "total": metrics['total'],
                "extraction_rate": metrics['extraction_rate'],
                "extraction_failures": metrics['extraction_failures']
            },
            "results": metrics['results']
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Detailed results saved to: {output_path}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
