
import os
#os.environ["TORCHDYNAMO_DISABLE"] = "1"
#os.environ["TORCH_COMPILE_DISABLE"] = "1"
#os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
#os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"  # disables Unsloth auto compiler :contentReference[oaicite:1]{index=1}
# critical: disable SDPA flash/mem-efficient kernels globally
#os.environ["PYTORCH_SDP_DISABLE_FLASH"] = "1"
#os.environ["PYTORCH_SDP_DISABLE_MEM_EFFICIENT"] = "1"
#os.environ["PYTORCH_SDP_DISABLE_MATH"] = "0"
# Use FP16 to avoid CUBLAS_STATUS_INVALID_VALUE with bf16 on some GPU/driver combos (ORPO/DPO)
#USE_FP16_FOR_TRAINING = os.environ.get("PREF_TRAIN_USE_FP16", "1") == "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import unsloth
from unsloth import FastLanguageModel
from peft import PeftModel
from pathlib import Path
import torch
# Force SDPA backend selection globally
#torch.backends.cuda.enable_flash_sdp(False)
#torch.backends.cuda.enable_mem_efficient_sdp(False)
#torch.backends.cuda.enable_math_sdp(True)


LOCAL_MODEL_DIR = "/home/malam/wsl-tunix/imo/model/gpt-oss-120b-bnb4"
SFT_LORA_DIR    = "/home/malam/wsl-tunix/aimo3_pivot20Feb2026/dpo_adapter/v25032026checkpoint-500"  # change to your adapter dir
OUT_DIR         = "/home/malam/wsl-tunix/aimo3_pivot20Feb2026/dpo_adapter"
TRAIN_JSONL     = "/home/malam/wsl-tunix/aimo3_pivot20Feb2026/datasets/training_samples_multi.jsonl"
TOTAL_STEPS     = 1000
RANK            = 4
MAX_PROMPT_LEN = 2048
MAX_TOTAL_LEN  = 2048*2 + 128 # total prompt+response; must be int (slice indices in trainer)
# Prefilter: drop rows where tokenized prompt+chosen or prompt+rejected exceeds this (set PREF_MAX_RECORD_TOKENS to override).
MAX_RECORD_TOKENS = int(os.environ.get("PREF_MAX_RECORD_TOKENS", str(MAX_TOTAL_LEN)))

def _has_lora_adapter(path: str) -> bool:
    p = Path(path)
    return p.exists() and (p / "adapter_config.json").exists()

# Keep these consistent with your tokenizer/model setup (ints for slice indices in trainer)
max_seq_length = int(MAX_TOTAL_LEN) + 128  # should be >= max_length you pass to ORPO/DPO
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LOCAL_MODEL_DIR,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- Attach LoRA (either load existing trainable adapter OR create new) ---
if _has_lora_adapter(SFT_LORA_DIR):
    model = PeftModel.from_pretrained(model, SFT_LORA_DIR, is_trainable=True)
else:
    print("[LoRA] No adapter found, creating new LoRA")
    model = FastLanguageModel.get_peft_model(
        model,
        r=RANK,
        lora_alpha=2 * RANK,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        use_gradient_checkpointing=True,
    )
model.config.use_cache = False
#setattr(model.config, "_attn_implementation", "eager")
# Put model into training mode (Unsloth helper)
model = FastLanguageModel.for_training(model)

# Sanity check: ensure we’re NOT full finetuning
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"[LoRA] trainable={trainable:,} / total={total:,} ({100*trainable/total:.4f}%)")


# ============================================================
# CORRECTED ORPO/DPO TRAINING SNIPPET for YOUR NEW PREF JSONL
# (prompt already includes avoid_memo_final block + ends with [Solution])
# chosen = oracle_trace, rejected = long wrong attempt response_text
# ============================================================

import os, re, json
from pathlib import Path
from datetime import datetime

#import torch
from datasets import load_dataset
from transformers import TrainingArguments

# TRL trainers (ORPO preferred, DPO fallback)
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    from trl import ORPOTrainer, ORPOConfig
    _HAS_ORPO = True
except Exception:
    _HAS_ORPO = False
    from trl import DPOTrainer, DPOConfig

try:
    from trl import maybe_apply_chat_template
    _HAS_APPLY_CHAT_TEMPLATE = True
except Exception:
    maybe_apply_chat_template = None
    _HAS_APPLY_CHAT_TEMPLATE = False

# ----------------------------
# MUST match your model setup earlier
# model, tokenizer = FastLanguageModel.from_pretrained(...)
# model = PeftModel.from_pretrained(...) OR FastLanguageModel.get_peft_model(...)
# model = FastLanguageModel.for_training(model)
# ----------------------------

def _bf16_ok() -> bool:
    return torch.cuda.is_available()

def normalize_ws(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _safe_str(x):
    if x is None: return ""
    if isinstance(x, str): return x
    if isinstance(x, (int, float, bool)): return str(x)
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)

def harmony_to_flat_text(harmony) -> str:
    """Fallback: flatten messages to a single string when chat template is not applied."""
    if not isinstance(harmony, list):
        return ""
    parts = []
    for m in harmony:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        if role == "assistant":
            parts.append(content)
        elif role == "tool":
            parts.append("<tool_result>\n" + content + "\n</tool_result>")
    return normalize_ws("\n\n".join(parts)) if parts else ""

def build_prompt_from_problem(problem_text: str) -> str:
    """Build a minimal chat prompt from problem only (fallback when JSONL has no 'prompt' key)."""
    problem_text = normalize_ws(problem_text or "")
    return (
        "<|system|>\n"
        "You are an IMO-level math problem solver. End with a non-negative integer in \\boxed{...}.\n"
        "<|user|>\n"
        f"{problem_text}\n"
        "<|assistant|>\n"
    )

def ensure_assistant_tail(prompt: str) -> str:
    """
    Ensure the prompt ends with <|assistant|> so the model knows where to generate.
    Works with minimal format (problem text only) or legacy prompt ending in [Solution].
    """
    p = normalize_ws(prompt)

    # If prompt already contains assistant tag, good.
    if "<|assistant|>" in p:
        if not p.rstrip().endswith("<|assistant|>"):
            # ensure assistant is last role marker
            if not p.rstrip().endswith("\n<|assistant|>"):
                p = p.rstrip() + "\n<|assistant|>\n"
        return p

    # Otherwise append your chat markers
    return p.rstrip() + "\n<|assistant|>\n"

def to_pref_row(ex: dict) -> dict:
    """
    Supports two formats:
    - Minimal (from preprocess_training_samples): prompt, chosen, rejected as strings (e.g. chosen="\\boxed{5}", rejected="\\boxed{4}"), optional meta.
    - Legacy: chosen/rejected as message lists; kept as-is for TRL maybe_apply_chat_template.
    Ensures prompt ends with <|assistant|> for the model.
    """
    if ex.get("prompt"):
        prompt = ensure_assistant_tail(_safe_str(ex["prompt"]))
    else:
        problem = _safe_str(ex.get("problem", ""))
        prompt = ensure_assistant_tail(build_prompt_from_problem(problem))

    raw_chosen = ex.get("chosen", "")
    raw_rejected = ex.get("rejected", "")
    if isinstance(raw_chosen, list):
        chosen = raw_chosen if raw_chosen else ""
    else:
        chosen = normalize_ws(_safe_str(raw_chosen))
    if isinstance(raw_rejected, list):
        rejected = raw_rejected if raw_rejected else ""
    else:
        rejected = normalize_ws(_safe_str(raw_rejected))

    if isinstance(chosen, str) and (not (chosen or "").strip() or not (rejected or "").strip()):
        return {"prompt": "", "chosen": "", "rejected": ""}
    if isinstance(chosen, list) and (len(chosen) == 0 or len(rejected) == 0):
        return {"prompt": "", "chosen": "", "rejected": ""}

    out = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    if ex.get("meta") is not None:
        out["meta"] = ex["meta"]
    return out


# ----------------------------
# LOAD DATASET: pref JSONL with prompt, chosen, rejected (minimal: boxed-only strings, or legacy message lists), optional meta
# ----------------------------
def _token_count(text) -> int:
    """Tokenize for length; accept str or Harmony list (flatten to str)."""
    if not isinstance(text, str):
        text = harmony_to_flat_text(text) if isinstance(text, list) else (text or "")
    return len(tokenizer.encode(text, add_special_tokens=False))

def _under_max_tokens(row: dict) -> bool:
    n_p = _token_count(row.get("prompt") or "")
    n_c = _token_count(row.get("chosen") or "")
    n_r = _token_count(row.get("rejected") or "")
    return (n_p + n_c <= MAX_RECORD_TOKENS and n_p + n_r <= MAX_RECORD_TOKENS)

def _apply_chat_template_or_flatten(ex: dict) -> dict:
    """If chosen/rejected are message lists, apply tokenizer chat template (TRL); else flatten as fallback.
    Always coerce chosen/rejected to strings so TRL's is_conversational never sees empty lists (IndexError)."""
    if _HAS_APPLY_CHAT_TEMPLATE and maybe_apply_chat_template is not None:
        try:
            ex = maybe_apply_chat_template(ex, tokenizer)
        except Exception:
            pass
    # Always ensure prompt/chosen/rejected are strings (trainer's maybe_apply_chat_template -> is_conversational crashes on empty list)
    if isinstance(ex.get("prompt"), list):
        ex = {**ex, "prompt": normalize_ws(harmony_to_flat_text(ex["prompt"]))}
    elif ex.get("prompt") is not None and not isinstance(ex.get("prompt"), str):
        ex = {**ex, "prompt": str(ex["prompt"])}
    if isinstance(ex.get("chosen"), list):
        ex = {**ex, "chosen": normalize_ws(harmony_to_flat_text(ex["chosen"]))}
    elif not isinstance(ex.get("chosen"), str):
        ex = {**ex, "chosen": str(ex.get("chosen") or "")}
    if isinstance(ex.get("rejected"), list):
        ex = {**ex, "rejected": normalize_ws(harmony_to_flat_text(ex["rejected"]))}
    elif not isinstance(ex.get("rejected"), str):
        ex = {**ex, "rejected": str(ex.get("rejected") or "")}
    return ex

def _valid_row(x) -> bool:
    if not x.get("prompt"):
        return False
    c, r = x.get("chosen"), x.get("rejected")
    if isinstance(c, list):
        return len(c) > 0 and len(r) > 0
    return bool((c or "").strip()) and bool((r or "").strip())


def _nonempty_chosen_rejected_str(x: dict) -> bool:
    """After flatten/template, chosen/rejected must be non-empty strings (must be top-level def for pickling)."""
    c, r = x.get("chosen"), x.get("rejected")
    return bool((c or "").strip()) and bool((r or "").strip())


assert Path(TRAIN_JSONL).exists(), f"Missing: {TRAIN_JSONL}"
print(f"[PREF] MAX_RECORD_TOKENS (prefilter)={MAX_RECORD_TOKENS}  (env PREF_MAX_RECORD_TOKENS to change)")
ds = load_dataset("json", data_files=TRAIN_JSONL, split="train")
ds = ds.shuffle(seed=25)  # explicit shuffle for reproducibility
ds = ds.map(to_pref_row, remove_columns=ds.column_names)
# num_proc=1: filters use global `tokenizer`; multiprocessing can fail to pickle or skip incorrectly.
ds = ds.filter(_valid_row, num_proc=1)
# Apply chat template when chosen/rejected are message lists (model consumes messages via template)
ds = ds.map(_apply_chat_template_or_flatten, desc="apply_chat_template")
ds = ds.filter(_nonempty_chosen_rejected_str, num_proc=1)
n_before = len(ds)
ds = ds.filter(_under_max_tokens, num_proc=1)
n_after = len(ds)
print(
    f"[PREF] Token prefilter: {n_before} -> {n_after} rows "
    f"(dropped {n_before - n_after}; cap={MAX_RECORD_TOKENS} tok: prompt+chosen AND prompt+rejected each)"
)

print("[PREF] Dataset size:", len(ds))
print("[PREF] Prompt preview:\n", ds[0]["prompt"][:650])
print("[PREF] Chosen preview:\n", ds[0]["chosen"][:350])
print("[PREF] Rejected preview:\n", ds[0]["rejected"][:350])


# ============================================================
# IMPORTANT FIXES vs your snippet:
# 1) ORPOTrainer/DPOTrainer expects dataset keys: prompt/chosen/rejected (correct)
# 2) For long reasoning: set max_prompt_length + max_length realistically.
#    Your previous MAX_LENGTH was huge (4096 + 16384 + 128 = 20608 tokens).
#    That will OOM even with preference training if you actually hit it.
#    Set to what your *data actually contains*.
# ============================================================

# Set these to your real distribution.
# If you truly need 16k completions, you must ensure the MODEL's rope/context supports it
# AND accept very slow training. Start with 8k total and scale.


# If your TRL version supports this environment var, it reduces fragmentation
import time, gc
from transformers import TrainerCallback
def _is_oom_error(e: BaseException) -> bool:
    msg = str(e).lower()
    return (
        isinstance(e, torch.OutOfMemoryError)
        or isinstance(e, getattr(torch, "AcceleratorError", ()))
        or ("out of memory" in msg)
        or ("cudaerrormemoryallocation" in msg)
    )

import gc, time, torch

def _is_cuda_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return (
        isinstance(e, (torch.OutOfMemoryError, getattr(torch.cuda, "OutOfMemoryError", torch.OutOfMemoryError)))
        or ("cuda error: out of memory" in msg)
        or ("out of memory" in msg)
        or ("cudaerrormemoryallocation" in msg)
        or ("currentstreamcapturestatusmayinitctx" in msg)
        or ("cudagraph" in msg)
    )
def _is_skippable_runtime(e: BaseException) -> bool:
    msg = str(e).lower()
    return (
        _is_cuda_oom(e)
        or ("out of memory" in msg)
        or ("cudnn" in msg and "error" in msg)
        or ("setstorage:" in msg and "out of bounds" in msg)
        or ("out of bounds for storage" in msg)
        or ("unsloth_zoo/gradient_checkpointing.py" in msg)
    )
def _max_seq_len_from_inputs(inputs: dict) -> int:
    mx = 0
    for _, v in inputs.items():
        if torch.is_tensor(v) and v.dim() == 2:
            mx = max(mx, v.shape[1])
    return mx

class OOMSkipMixin:
    oom_sleep_s = 0.2  # small pause helps after async OOM

    def _ensure_zero_loss(self):
        # Create once while CUDA is healthy; reuse forever.
        if not hasattr(self, "_zero_loss_gpu") or self._zero_loss_gpu is None:
            self._zero_loss_gpu = torch.zeros((), device=self.args.device, requires_grad=True)

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._ensure_zero_loss()

        try:
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        except BaseException as e:
            if not _is_skippable_runtime(e):
                raise

            mx_len = _max_seq_len_from_inputs(inputs)
            err_short = str(e).replace("\n", " ")
            if len(err_short) > 240:
                err_short = err_short[:240] + "..."
            print(f"[SKIP] step={self.state.global_step} max_len={mx_len} err={err_short}")

            # Cleanup grads
            try:
                if getattr(self, "optimizer", None) is not None:
                    self.optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            try:
                model.zero_grad(set_to_none=True)
            except Exception:
                pass

            import gc, time
            gc.collect()

            if torch.cuda.is_available():
                # These may fail if CUDA is in a bad async state; keep them best-effort.
                try: torch.cuda.synchronize()
                except Exception: pass
                try: torch.cuda.empty_cache()
                except Exception: pass
                try: torch.cuda.ipc_collect()
                except Exception: pass

            time.sleep(self.oom_sleep_s)

            return self._zero_loss_gpu
class CUDACleanupCallback(TrainerCallback):
    def __init__(self, sleep_s=1, every_n_steps=1, do_empty_cache=True):
        self.sleep_s = sleep_s
        self.every_n_steps = every_n_steps
        self.do_empty_cache = do_empty_cache

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps != 0:
            return control

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated() / 2**30
            reserv = torch.cuda.memory_reserved() / 2**30
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"[cuda] step={state.global_step} alloc={alloc:.2f}GiB reserved={reserv:.2f}GiB peak={peak:.2f}GiB")

        # Try to reduce Python-side garbage
        gc.collect()

        # Try to return cached blocks to CUDA (helps fragmentation sometimes)
        if torch.cuda.is_available() and self.do_empty_cache:
            torch.cuda.empty_cache()

        if self.sleep_s:
            time.sleep(self.sleep_s)

        return control

try:
    from trl import ORPOTrainer, ORPOConfig
    _HAS_ORPO = True
except Exception:
    _HAS_ORPO = False
    from trl import DPOTrainer, DPOConfig

common_cfg = dict(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,

    learning_rate=5e-6,
    max_steps=TOTAL_STEPS,
    logging_steps=100,
    save_steps=100,
    save_total_limit=2,
    bf16=  _bf16_ok(),
    fp16= (not _bf16_ok()),
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    beta=0.001,
    max_prompt_length=MAX_PROMPT_LEN,
    max_length=MAX_TOTAL_LEN,
    optim="paged_adamw_8bit",
    # Use 1 process for dataset tokenization; 0 DataLoader workers to avoid system OOM/restart (defaults can spawn 24+ processes)
    dataset_num_proc=1,
    dataloader_num_workers=0,

    # IMPORTANT:
    # DO NOT set model_init_kwargs when model is already instantiated.
    # model_init_kwargs=None,   # (optional) leaving it out is best
)
'''
if _HAS_ORPO:
    args = ORPOConfig(**common_cfg)
    trainer = ORPOTrainer(
        model=model,            # instantiated model object ✅
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,
    )
else:
    args = DPOConfig(**common_cfg)
    trainer = DPOTrainer(
        model=model,            # instantiated model object ✅
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,
    )
'''

# 4) NOW create trainer (your block is correct here)
if _HAS_ORPO:
    from trl import ORPOTrainer, ORPOConfig

    class ORPOTrainerSkipOOM(OOMSkipMixin, ORPOTrainer):
        """OOMSkipMixin must be first so training_step runs here; if OOM skips never log, Unsloth may override training_step."""

    args = ORPOConfig(**common_cfg)
    trainer = ORPOTrainerSkipOOM(
        model=model,
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,
    )
else:
    from trl import DPOTrainer, DPOConfig

    class DPOTrainerSkipOOM(OOMSkipMixin, DPOTrainer):
        """OOMSkipMixin must be first so training_step runs here; if OOM skips never log, Unsloth may override training_step."""

    args = DPOConfig(**common_cfg)
    trainer = DPOTrainerSkipOOM(
        model=model,
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,
    )

# 5) callbacks (optional) then train
# trainer.add_callback(...)
# Pre-create a safe "zero loss" tensor BEFORE training (so we don't allocate after an OOM)
if torch.cuda.is_available():
    trainer._zero_loss_gpu = torch.zeros((), device=trainer.args.device, requires_grad=True)

try:
    trainer.train()
except BaseException as e:
    if _is_cuda_oom(e):
        print(f"[OOM] OOM escaped training_step at global_step={trainer.state.global_step}. Exiting gracefully.\n{e}")
        # best-effort cleanup
        import gc, time
        try:
            if getattr(trainer, "optimizer", None) is not None:
                trainer.optimizer.zero_grad(set_to_none=True)
        except Exception:
            pass
        try:
            model.zero_grad(set_to_none=True)
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            try: torch.cuda.synchronize()
            except Exception: pass
            try: torch.cuda.empty_cache()
            except Exception: pass
            try: torch.cuda.ipc_collect()
            except Exception: pass
        # You can still save adapter after this (your code below will run if you don't re-raise).
    else:
        raise
# save adapter
phase_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
mid_dir = os.path.join(OUT_DIR, f"pref_{phase_ts}")
os.makedirs(mid_dir, exist_ok=True)
trainer.model.save_pretrained(mid_dir)
tokenizer.save_pretrained(mid_dir)
print("Saved preference-tuned LoRA adapter to:", mid_dir)
