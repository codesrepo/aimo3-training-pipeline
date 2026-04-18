#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) training script for GPT-OSS-120B.

Uses reward-based training where the reward is based on exact answer matching
from \\boxed{...} format. Can optionally start from an SFT adapter.
"""
import os
import re
import json
import io
import math
import contextlib
import signal
from pathlib import Path

import numpy as np
import sympy as sp
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from trl import GRPOTrainer, GRPOConfig
from sentence_transformers import SentenceTransformer

os.environ["TRANSFORMERS_NO_WANDB"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ.pop("CUDA_HOME", None)
os.environ.pop("LD_LIBRARY_PATH", None)
# -----------------------------------------------------------------------------
# Global sentence transformer model for cosine similarity (with safe fallbacks)
# -----------------------------------------------------------------------------
_sentence_model = None

def get_global_sentence_model():
    """Get or create global sentence transformer model."""
    global _sentence_model
    if _sentence_model is None:
        if SentenceTransformer is None:
            return None
        try:
            _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Warning: Could not load SentenceTransformer: {e}")
            _sentence_model = None
            return None
    return _sentence_model


def compute_cosine_similarity(text1, text2):
    """
    Compute cosine similarity between two texts:
      1) SentenceTransformer embeddings (preferred)
      2) HashingVectorizer cosine (fallback)
      3) Jaccard similarity (last resort)
    """
    if not text1 or not text2:
        return 0.0

    # 1) Try SentenceTransformer
    model = get_global_sentence_model()
    if model is not None:
        try:
            embeddings = model.encode([text1, text2], convert_to_numpy=True)
            emb1, emb2 = embeddings[0], embeddings[1]
            dot_product = float(np.dot(emb1, emb2))
            norm1 = float(np.linalg.norm(emb1))
            norm2 = float(np.linalg.norm(emb2))
            if norm1 == 0.0 or norm2 == 0.0:
                return 0.0
            return dot_product / (norm1 * norm2)
        except Exception:
            # fall through to hashing
            pass

    # 2) Last resort: Jaccard similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0



# ============================================================================
# Tool-aware DrGRPO (UPDATED for your new JSONL schema)
# Your new rows look like:
# {
#   "problem": "...",
#   "final_answer": 462,
#   "solved_attempt": 3,
#   "key_idea": "...",
#   "proof_skeleton": [...],
#   "attainment_or_example": "...",
#   "sanity_checks": [...],
#   "failure": {...}
# }
#
# Changes vs your last script:
#  1) Dataset mapping now uses ex["problem"] and ex["final_answer"]
#  2) Optional: include key_idea/proof_skeleton/sanity_checks as "Verifier feedback"
#     (keeps your "repair-style" behavior without needing attempts[])
#  3) Reward func signature + GT alignment fixed (per-prompt group alignment)
#  4) Version-safe GRPOConfig filtering for beta/scale_rewards
# ============================================================================


# ----------------------------
# Paths (yours)
# ----------------------------
LOCAL_MODEL_DIR = "/home/malam/wsl-tunix/imo/model/gpt-oss-120b-bnb4"
TRAIN_JSONL     = "/home/malam/wsl-tunix/imo/openmath_data/aimo_certs_207.jsonl"
SFT_LORA_DIR = "/home/malam/wsl-tunix/imo/home/saved_models/lora_adapter_sft_r4_e5"
OUT_DIR         = "/home/malam/wsl-tunix/imo/lora_grpo_adapter"


# ----------------------------
# Prompt blocks
# ----------------------------
SYSTEM_BLOCK = (
    "You are an IMO-level mathematical problem solver.\n"
    "Output must end with a non-negative integer in \\boxed{...} where 0<=answer<=99999.\n"
)

TOOL_BLOCK = (
    "You may use a python tool call for verification. If you use Python, emit exactly:\n"
    "<tool_call name=\"python\">{\"code\": \"...\"}</tool_call>\n"
    "Then continue.\n"
)

CERT_PROTOCOL = (
    "Follow this protocol (be concise):\n"
    "1) Constraints: bullet the key constraints and invariants.\n"
    "2) Plan: 2-4 lines.\n"
    "3) Certificate: state the invariant/lemma/formula/DP/state definition needed to verify.\n"
    "4) Check: if possible, run a small-case or arithmetic check (python allowed) and align it with the boxed answer.\n"
    "5) Final: \\boxed{answer}\n"
    "Avoid long exposition.\n"
)

def _safe_str(x):
    if x is None: return ""
    if isinstance(x, str): return x
    if isinstance(x, (int, float, bool)): return str(x)
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)

def build_prompt_from_problem(problem_text: str) -> str:
    parts = [
        f"<|system|>\n{SYSTEM_BLOCK}{CERT_PROTOCOL}",
        f"<|system|>\n{TOOL_BLOCK}",
        f"<|user|>\n{problem_text}",
        f"<|assistant|>\n",
    ]
    return "\n".join(parts)

# ----------------------------
# Load model
# ----------------------------
NUM_GENERATIONS = 2
MAX_PROMPT_LEN = 512
MAX_COMP_LEN   = 512

print(f"\n[1/4] Loading tokenizer from {LOCAL_MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("  ✓ Set pad_token to eos_token")
tokenizer.padding_side = "left"

print(f"\n[2/4] Loading base model with 4-bit quantization...")
max_memory = {0: "90GiB"}  # leave headroom for KV cache + optimizer states
base_model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_DIR,
    device_map="cuda:0",     
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    load_in_4bit=True,
    trust_remote_code=True,
    max_memory=max_memory,
)
print("  ✓ Base model loaded")

def _has_lora_adapter(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    return (p / "adapter_config.json").exists() and (
        (p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists()
    )

print(f"\n[3/4] Setting up LoRA adapter...")
if _has_lora_adapter(SFT_LORA_DIR):
    print(f"  Loading SFT adapter from {SFT_LORA_DIR}...")
    model = PeftModel.from_pretrained(base_model, SFT_LORA_DIR, is_trainable=True)
    print("  ✓ Loaded SFT adapter (now trainable)")
else:
    print(f"  WARNING: No SFT adapter found at {SFT_LORA_DIR}. Creating a new LoRA.")
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)
    print("  ✓ Created new LoRA adapter")
model.generation_config.max_new_tokens = MAX_COMP_LEN
model.config.use_cache = False
model.generation_config.max_length     = MAX_COMP_LEN*2+8
model.config.use_cache = False
model.print_trainable_parameters()

# ----------------------------
# NEW dataset mapping for your revised schema
# ----------------------------
def _make_verifier_feedback(ex: dict) -> str:
    # Use the solution metadata as "verifier feedback" (compact).
    # This preserves your "repair-style" prompt WITHOUT needing attempts[].
    key_idea = _safe_str(ex.get("key_idea", ""))
    proof_skel = ex.get("proof_skeleton", [])
    sanity = ex.get("sanity_checks", [])
    attainment = _safe_str(ex.get("attainment_or_example", ""))

    lines = []
    if key_idea:
        lines.append(f"Key idea: {key_idea}")
    if proof_skel:
        lines.append("Proof skeleton:")
        for i, step in enumerate(proof_skel, 1):
            lines.append(f"  {i}. {_safe_str(step)}")
    if attainment:
        lines.append(f"Attainment/example: {attainment}")
    if sanity:
        lines.append("Sanity checks:")
        for s in sanity:
            lines.append(f"  - {_safe_str(s)}")
    return "\n".join(lines).strip()

def to_grpo_row(ex):
    # IMPORTANT: now reading "problem" and "final_answer"
    problem = _safe_str(ex.get("problem", ""))
    gt = ex.get("final_answer", None)

    # ground_truth must be int-like for reward matching
    ground_truth = _safe_str(gt).strip()

    # If you want pure solve (no oracle hints), comment out the feedback block.
    feedback = _make_verifier_feedback(ex)

    if False:
        prompt_parts = [
            f"<|system|>\n{SYSTEM_BLOCK}{CERT_PROTOCOL}",
            f"<|system|>\n{TOOL_BLOCK}",
            f"<|user|>\n{problem}",
            f"<|user|>\nReference outline (for self-correction, do NOT copy verbatim):\n{feedback}\n\nNow produce a short solution following the protocol and end with \\boxed{{answer}}.",
            f"<|assistant|>\n",
        ]
        prompt = "\n".join(prompt_parts)
    else:
        prompt = build_prompt_from_problem(problem)

    return {"prompt": prompt, "ground_truth": ground_truth, "feedback": feedback}


assert Path(TRAIN_JSONL).exists(), f"Missing: {TRAIN_JSONL}"
ds = load_dataset("json", data_files=TRAIN_JSONL, split="train")
ds = ds.map(to_grpo_row, remove_columns=ds.column_names)
ds = ds.filter(lambda x: len(x["prompt"]) > 0 and len(str(x["ground_truth"])) > 0)

print("[DrGRPO] Dataset size:", len(ds))
print("[DrGRPO] Prompt preview:\n", ds[0]["prompt"][:650])

# ----------------------------
# Reward utilities
# ----------------------------
BOX_RE = re.compile(r"\\boxed\{(\d+)\}")
TOOLCALL_RE = re.compile(r"<tool_call\s+name=[\"']python[\"']\s*>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)

BANNED_SUBSTRINGS = [
    "import ", "__", "open(", "os.", "sys.", "subprocess", "socket", "shutil",
    "pathlib", "requests", "urllib", "pip", "apt", "bash", "cuda", "torch",
    "eval(", "exec(", "compile(", "globals(", "locals(", "getattr(", "setattr(",
]

def _extract_text(c):
    if isinstance(c, list) and c and isinstance(c[0], dict) and "content" in c[0]:
        return c[0]["content"]
    return str(c)

def _extract_boxed_int(text: str):
    m = BOX_RE.findall(text)
    return int(m[-1]) if m else None

def _parse_toolcall_code(text: str):
    m = TOOLCALL_RE.search(text)
    if not m:
        return None
    payload = m.group(1).strip()
    try:
        obj = json.loads(payload)
        if isinstance(obj, dict) and "code" in obj:
            return str(obj["code"])
    except Exception:
        return None
    return None

def _run_python_code_safely(code: str, timeout_s: int = 2):
    code_l = code.lower()
    for bad in BANNED_SUBSTRINGS:
        if bad in code_l:
            return (False, "", "banned_token")
    if len(code) > 2000:
        return (False, "", "too_long")

    safe_builtins = {
        "print": print, "range": range, "len": len, "int": int, "float": float,
        "abs": abs, "min": min, "max": max, "sum": sum,
    }
    g = {"__builtins__": safe_builtins, "math": math, "np": np, "numpy": np, "sp": sp, "sympy": sp}
    l = {}
    buf = io.StringIO()

    def _alarm_handler(signum, frame):
        raise TimeoutError("timeout")

    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, g, l)
    except Exception as e:
        return (False, buf.getvalue(), f"err:{type(e).__name__}")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

    return (True, buf.getvalue(), "")

def _last_printed_int(stdout: str):
    toks = re.findall(r"-?\d+", stdout)
    return int(toks[-1]) if toks else None

def has_protocol_headers(text: str) -> int:
    t = text.lower()
    ok = 0
    ok += 1 if "constraints" in t else 0
    ok += 1 if "plan" in t else 0
    ok += 1 if "certificate" in t else 0
    ok += 1 if "check" in t else 0
    return 1 if ok >= 2 else 0

def _strip_boxed(text: str) -> str:
    """Remove \\boxed{...} content from text."""
    return BOX_RE.sub("", text).strip()

def has_generic_certificate(text: str, feedback: str) -> float:
    """
    reward = 5*(sim-0.5)^2 if sim > 0.5 else negative of same expression
    sim computed between (completion minus boxed) and per-sample feedback
    """
    pred = _strip_boxed(text)
    fb = feedback or ""
    sim = compute_cosine_similarity(pred, fb)

    val = 5.0 * (sim - 0.5) ** 2
    return val if sim > 0.5 else -val


# ----------------------------
# Reward func (FIXED signature + correct GT indexing)
# ----------------------------
def make_reward_func(phase: str, num_generations: int):
    assert phase in ["A", "B"]

    if phase == "A":
        w_exact = 0.8
        w_boxfmt = 0.3
        w_headers = 0.25
        w_cert = 0.35
        w_tool_ok = 0.25
        w_tool_consistency = 0.25
        len_pen = 0.00003
    else:
        w_exact = 1.5
        w_boxfmt = 0.35
        w_headers = 0.15
        w_cert = 0.20
        w_tool_ok = 0.20
        w_tool_consistency = 0.30
        len_pen = 0.00005

    def reward_fn(prompts, completions, **kwargs):
        ground_truth = kwargs.get("ground_truth", None)
        feedback_list = kwargs.get("feedback", None)
        fb_list = []
        if feedback_list is not None:
            for x in feedback_list:
                fb_list.append(str(x) if x is not None else "")

        gt_list = []
        if ground_truth is not None:
            for x in ground_truth:
                try:
                    gt_list.append(int(str(x).strip()))
                except Exception:
                    gt_list.append(None)

        texts = [_extract_text(c) for c in completions]
        n = len(texts)
        rewards = [0.0] * n
        per_item_correct = [False] * n
        per_item_len = [len(t) for t in texts]

        for i, text in enumerate(texts):
            group_idx = i // num_generations
            gt = gt_list[group_idx] if group_idx < len(gt_list) else None
            fb = fb_list[group_idx] if group_idx < len(fb_list) else ""

            boxed = _extract_boxed_int(text)
            if boxed is not None:
                rewards[i] += w_boxfmt
            else:
                rewards[i] -= 0.25

            rewards[i] += w_headers * has_protocol_headers(text)
            if fb:
                rewards[i] += has_generic_certificate(text, fb)

            if boxed is not None and gt is not None and boxed == gt:
                rewards[i] += w_exact
                per_item_correct[i] = True

            code = _parse_toolcall_code(text)
            if code is not None:
                ok, stdout, err = _run_python_code_safely(code, timeout_s=2)
                if ok:
                    rewards[i] += w_tool_ok
                    out_int = _last_printed_int(stdout)
                    if out_int is not None and gt is not None and out_int == gt:
                        rewards[i] += 0.7
                    if out_int is not None and boxed is not None and out_int == boxed:
                        rewards[i] += w_tool_consistency
                else:
                    rewards[i] -= 0.5

            rewards[i] -= len_pen * per_item_len[i]

        if num_generations > 1:
            num_groups = n // num_generations
            for g in range(num_groups):
                start = g * num_generations
                end = start + num_generations
                correct_idxs = [k for k in range(start, end) if per_item_correct[k]]
                if correct_idxs:
                    j = min(correct_idxs, key=lambda k: per_item_len[k])
                    rewards[j] += 0.25

        print(rewards)
        return rewards

    return reward_fn

# ----------------------------
# GRPOConfig helper (version-safe)
# ----------------------------
def _bf16_ok() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

def make_grpo_config(**cfg_kwargs):
    allowed = set(getattr(GRPOConfig, "__dataclass_fields__", {}).keys())
    filtered = {k: v for k, v in cfg_kwargs.items() if (not allowed) or (k in allowed)}
    return GRPOConfig(**filtered)

# ----------------------------
# DRGRPO training: 2-stage curriculum
# ----------------------------


TOTAL_STEPS = 30
STEPS_A     = 10
STEPS_B     = TOTAL_STEPS - STEPS_A

common_args = dict(
    output_dir                   = OUT_DIR,
    per_device_train_batch_size  = 1,
    gradient_accumulation_steps  = 2,
    learning_rate                = 5e-6,
    logging_steps                = 5,
    save_steps                   = 10,
    save_total_limit             = 2,
    bf16                         = _bf16_ok(),
    fp16                         = (not _bf16_ok()),
    num_generations              = NUM_GENERATIONS,
    max_prompt_length            = MAX_PROMPT_LEN,
    max_completion_length        = MAX_COMP_LEN,
    loss_type                    = "dr_grpo",
    report_to                    = "none",

    # DR-GRPO knobs (applied only if your TRL exposes them)
    beta                         = 0.001,
    scale_rewards                = False,
)

# ---- Phase A ----

grpo_args_A = make_grpo_config(**common_args, max_steps=STEPS_A)
reward_A = make_reward_func("A", NUM_GENERATIONS)

trainer_A = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=grpo_args_A,
    train_dataset=ds,
    reward_funcs=[reward_A],
)

print("\n" + "="*80)
print("Starting Tool-aware DrGRPO Phase A (certificate-first)...")
print("="*80)
trainer_A.train()

mid_dir = os.path.join(OUT_DIR, "phaseA")
os.makedirs(mid_dir, exist_ok=True)
trainer_A.model.save_pretrained(mid_dir)
tokenizer.save_pretrained(mid_dir)
print("Saved Phase A adapter to:", mid_dir)

# ---- Phase B ----
reward_B = make_reward_func("B", NUM_GENERATIONS)
grpo_args_B = make_grpo_config(**common_args, max_steps=STEPS_B)

trainer_B = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=grpo_args_B,
    train_dataset=ds,
    reward_funcs=[reward_B],
)

print("\n" + "="*80)
print("Starting Tool-aware DrGRPO Phase B (answer-first)...")
print("="*80)
trainer_B.train()

# ----------------------------
# Save final adapter + tokenizer
# ----------------------------
os.makedirs(OUT_DIR, exist_ok=True)
trainer_B.model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("Saved final LoRA adapter to:", OUT_DIR)

