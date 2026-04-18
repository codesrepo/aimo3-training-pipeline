#!/usr/bin/env python3
"""
Preprocess scored JSONL with multi-turn Harmony-style messages into DPO rows.

Chosen trace: take the first K and last L *assistant blocks* (each block = one assistant
message plus following tool messages until the next assistant). Merge by index (dedup),
preserve order. Output chosen as a Harmony message list (role/content/channel/name), not a
single flattened string; enforce \\boxed{expected} on the last assistant message content.

If any selected assistant content contains a ```python (or ```py) fence, call a local
OpenAI-compatible server by default (vLLM: http://127.0.0.1:8000/v1, model openai/gpt-oss-120b)
to rewrite/fix the code; always ensure a final ans = <expected> and print(ans)
(use deterministic append when --no-llm-python-fix or the request fails). Unauthenticated
vLLM: omit key or use Bearer EMPTY (default when env has no OPENAI_API_KEY / VLLM_API_KEY).

Non-python assistant turns: LLM succinct summaries for both chosen and rejected Harmony traces
(same vLLM). Python: vLLM code fix + ans/print ONLY on chosen; rejected keeps raw ```python``` blocks
(no LLM python rewrite). Rejected is Harmony when the wrong attempt has messages; otherwise one
assistant message with \\boxed{wrong}.

Omitted ids (no output row): missing/invalid gold answer; no messages; exactly one attempt whose
resolved prediction matches gold; or no attempt with a wrong prediction (all correct / no usable
pred — would have been synthetic rejected only).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it: Any, **_kwargs: Any) -> Any:
        return it

BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")
PYTHON_FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)

# Local vLLM (e.g. `vllm serve openai/gpt-oss-120b`). Override with VLLM_BASE_URL / VLLM_MODEL.
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_VLLM_MODEL = "openai/gpt-oss-120b"
DEFAULT_VLLM_TIMEOUT_SEC = 600.0


@dataclass
class PythonFenceCounters:
    """Per-sample counts for ```python``` fences in the selected assistant slice."""

    total: int = 0
    llm_ok: int = 0
    llm_fail: int = 0
    no_llm_mode: int = 0


@dataclass
class SummarizeCounters:
    """Per-sample: compress non-python assistant turns + truncate tool blobs (chosen or rejected)."""

    assistant_seen: int = 0
    skipped_python: int = 0
    skipped_short: int = 0
    summarized_llm_ok: int = 0
    summarized_llm_fail: int = 0
    summarized_heuristic: int = 0
    tool_truncated: int = 0


def normalize_ws(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def canon_answer(x: Any) -> Optional[str]:
    if x is None:
        return None
    x = str(x).strip().replace(" ", "").replace(",", "")
    return x if x else None


def _expected_to_int(expected: Any) -> Optional[int]:
    if expected is None:
        return None
    try:
        s = str(expected).strip().replace(",", "")
        return int(s) if s else None
    except ValueError:
        return None


def build_prompt(problem: str) -> str:
    problem = normalize_ws(problem or "")
    return (
        "You are an IMO-level math problem solver.\n"
        "Solve the problem carefully. End with a non-negative integer in \\boxed{...}.\n\n"
        f"{problem}"
    )


def extract_boxed_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = BOX_RE.search(text)
    if not m:
        return None
    return canon_answer(m.group(1))


def synthetic_rejected_answer(expected: Any) -> Optional[str]:
    n = _expected_to_int(expected)
    if n is None:
        return None
    cands = [
        n + 1,
        max(0, n - 1),
        2 * n,
        max(1, n // 2) if n else 1,
    ]
    for c in cands:
        if c != n and c >= 0:
            return str(int(c))
    return str(n + 1)


def boxed_inner_canon(rejected_boxed: str) -> Optional[str]:
    """Canonical value inside \\boxed{...} for comparison / prompts."""
    return extract_boxed_from_text(rejected_boxed) if rejected_boxed else None


def row_prediction_canon(r: Dict) -> Optional[str]:
    """Canonical prediction from one scored row (predicted_answer or first \\boxed in text)."""
    pred = r.get("predicted_answer")
    if pred is None:
        text = (r.get("response_text") or "") + "\n" + (r.get("model_final") or "")
        pred_str = extract_boxed_from_text(text)
        if pred_str is not None:
            pred = pred_str
    if pred is None:
        return None
    return canon_answer(pred)


def find_first_wrong_attempt_row(rows: List[Dict], expected_raw: Any) -> Optional[Dict]:
    """Same ordering as get_rejected_boxed: first row with wrong canonical prediction."""
    expected_canon = canon_answer(expected_raw)
    for r in rows:
        pred = r.get("predicted_answer")
        if pred is None:
            text = (r.get("response_text") or "") + "\n" + (r.get("model_final") or "")
            pred_str = extract_boxed_from_text(text)
            if pred_str is not None:
                pred = pred_str
        if pred is None:
            continue
        p = canon_answer(pred)
        if p is not None and p != expected_canon:
            return r
    return None


def get_rejected_boxed(rows: List[Dict], expected_raw: Any) -> str:
    expected_canon = canon_answer(expected_raw)
    for r in rows:
        pred = r.get("predicted_answer")
        if pred is None:
            text = (r.get("response_text") or "") + "\n" + (r.get("model_final") or "")
            pred_str = extract_boxed_from_text(text)
            if pred_str is not None:
                pred = pred_str
        if pred is None:
            continue
        p = canon_answer(pred)
        if p is not None and p != expected_canon:
            raw = pred if isinstance(pred, str) else str(pred)
            if "\\boxed{" in raw:
                return raw.strip()
            return f"\\boxed{{{p}}}"
    syn = synthetic_rejected_answer(expected_raw)
    if syn is not None:
        return f"\\boxed{{{syn}}}"
    return "\\boxed{0}"


def assistant_blocks_from_messages(messages: List[Dict]) -> List[List[Dict]]:
    """Split messages into blocks: [assistant, tool?, tool?, ...] per assistant turn."""
    blocks: List[List[Dict]] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        role = (m.get("role") or "").lower()
        if role == "assistant":
            block = [m]
            i += 1
            while i < len(messages):
                nxt = messages[i]
                r2 = (nxt.get("role") or "").lower()
                if r2 == "tool":
                    block.append(nxt)
                    i += 1
                else:
                    break
            blocks.append(block)
        else:
            i += 1
    return blocks


def select_block_indices(n_blocks: int, first_k: int, last_l: int) -> List[int]:
    """Union of first K and last L block indices, sorted, deduped."""
    if n_blocks <= 0:
        return []
    first = list(range(0, min(first_k, n_blocks)))
    lo = max(0, n_blocks - last_l)
    last = list(range(lo, n_blocks))
    seen: Set[int] = set()
    out: List[int] = []
    for idx in first + last:
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return sorted(out)


def _harmony_export_message(m: Dict[str, Any]) -> Dict[str, Any]:
    """One JSON-serializable message dict matching scored JSONL / Harmony style."""
    role = (m.get("role") or "").strip().lower() or "assistant"
    row: Dict[str, Any] = {"role": role, "content": normalize_ws(m.get("content") or "")}
    if m.get("channel") is not None:
        row["channel"] = m["channel"]
    if role == "tool" and m.get("name"):
        row["name"] = m["name"]
    return row


def blocks_to_harmony_chosen(blocks: List[List[Dict]]) -> List[Dict[str, Any]]:
    """Flatten selected assistant blocks into an ordered list of Harmony messages."""
    out: List[Dict[str, Any]] = []
    for block in blocks:
        for m in block:
            out.append(_harmony_export_message(m))
    return out


def ensure_boxed_on_last_assistant(
    harmony_msgs: List[Dict[str, Any]],
    expected_canon: str,
) -> List[Dict[str, Any]]:
    """Mutate/copy so the last assistant message content ends with \\boxed{gold}."""
    last_ai: Optional[int] = None
    for i, m in enumerate(harmony_msgs):
        if (m.get("role") or "").lower() == "assistant":
            last_ai = i
    if last_ai is None:
        return harmony_msgs + [
            {"role": "assistant", "content": f"\\boxed{{{expected_canon}}}", "channel": "final"}
        ]
    fixed: List[Dict[str, Any]] = []
    for i, m in enumerate(harmony_msgs):
        mm = dict(m)
        if i == last_ai:
            mm["content"] = ensure_boxed_suffix(mm.get("content") or "", expected_canon)
        fixed.append(mm)
    return fixed


def _ans_literal_for_code(expected_canon: str, expected_raw: Any) -> str:
    n = _expected_to_int(expected_raw)
    if n is not None:
        return str(n)
    # string fallback for non-int gold
    return repr(expected_canon)


def _append_ans_print(code: str, ans_literal: str) -> str:
    code = (code or "").rstrip()
    has_print = bool(re.search(r"\bprint\s*\(\s*ans\s*\)", code, re.IGNORECASE))
    has_ans = bool(re.search(r"^\s*ans\s*=", code, re.MULTILINE))
    if has_print and has_ans:
        return code + ("\n" if not code.endswith("\n") else "")
    if has_print and not has_ans:
        return code + f"\n\nans = {ans_literal}\n"
    return code + f"\n\nans = {ans_literal}\nprint(ans)\n"


def llm_fix_python_block(
    code: str,
    problem: str,
    ans_literal: str,
    *,
    api_key: Optional[str],
    base_url: str,
    model: str,
    timeout: float,
) -> Optional[str]:
    try:
        import urllib.error
        import urllib.request
    except ImportError:
        return None

    url = base_url.rstrip("/") + "/chat/completions"
    system = (
        "You fix Python snippets used to solve math competition problems. "
        "Return ONLY the corrected Python code, no markdown fences, no explanation. "
        "The code must be runnable and end with assigning the final numerical result to variable `ans` "
        "and then `print(ans)`."
    )
    user = (
        f"Problem context (for math only):\n{problem[:4000]}\n\n"
        f"The correct numerical answer to expose as ans is: {ans_literal}\n\n"
        f"Fix this code:\n```python\n{code}\n```\n\n"
        "Output the full corrected script only."
    )
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
    ).encode("utf-8")
    bearer = (api_key or "").strip() or "EMPTY"
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {bearer}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:python|py)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()
        return text if text else None
    except Exception:
        return None


def llm_summarize_approach(
    reasoning: str,
    problem: str,
    answer_label: str,
    *,
    rejected_trace: bool = False,
    gold_answer: Optional[str] = None,
    max_steps: int = 8,
    api_key: Optional[str] = None,
    base_url: str = "",
    model: str = "",
    timeout: float = 60.0,
) -> Optional[str]:
    """Succinct numbered steps; no new code fences. chosen vs rejected prompts differ."""
    try:
        import urllib.request
    except ImportError:
        return None

    url = base_url.rstrip("/") + "/chat/completions"
    if not rejected_trace:
        system = (
            f"You compress a math solution trace. Output ONLY plain text: at most {max_steps} short numbered lines "
            "(1. ... 2. ...). "
            "Keep: setup, key lemmas/cases, critical equations or labels, and how they imply the final answer. "
            "Drop: filler, repetition, long algebra chains, hedging. "
            "Do NOT use markdown code fences. Do NOT paste Python. Do NOT restate the full problem. "
            f"The correct final answer is {answer_label}; stay consistent with it."
        )
    else:
        ga = gold_answer if gold_answer is not None else "?"
        system = (
            f"You compress a WRONG / dispreferred solution trace. Output ONLY plain text: at most {max_steps} short numbered lines "
            "(1. ... 2. ...). "
            f"The true (gold) answer is {ga}; this trace incorrectly concludes \\boxed{{{answer_label}}}. "
            "Summarize the mistaken setup, wrong lemmas or cases, and how the error leads to that wrong conclusion. "
            "Do not rewrite the math to be correct. "
            "Do NOT use markdown code fences. Do NOT paste new Python. Do NOT restate the full problem."
        )
    user = (
        f"Problem (reference):\n{problem[:2500]}\n\n"
        f"Reasoning to compress:\n{reasoning[:14000]}"
    )
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.25,
        }
    ).encode("utf-8")
    bearer = (api_key or "").strip() or "EMPTY"
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {bearer}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        text = normalize_ws(text)
        return text if text else None
    except Exception:
        return None


def _fallback_trunc_reasoning(text: str, max_chars: int = 900) -> str:
    """If LLM summarize fails, keep start of reasoning with ellipsis."""
    t = normalize_ws(text)
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 20].rstrip() + "\n… [truncated]"


def fix_python_fences_in_text(
    text: str,
    problem: str,
    expected_canon: str,
    expected_raw: Any,
    *,
    use_llm: bool,
    api_key: Optional[str],
    base_url: str,
    model: str,
    timeout: float,
    counters: Optional[PythonFenceCounters] = None,
) -> str:
    ans_literal = _ans_literal_for_code(expected_canon, expected_raw)

    def repl(m: Any) -> str:
        code = m.group(1).strip()
        fixed: Optional[str] = None
        if counters is not None:
            counters.total += 1
        if use_llm:
            fixed = llm_fix_python_block(
                code, problem, ans_literal, api_key=api_key, base_url=base_url, model=model, timeout=timeout
            )
            if counters is not None:
                if fixed is not None:
                    counters.llm_ok += 1
                else:
                    counters.llm_fail += 1
        elif counters is not None:
            counters.no_llm_mode += 1
        if fixed is None:
            fixed = _append_ans_print(code, ans_literal)
        else:
            fixed = _append_ans_print(fixed, ans_literal)
        return f"```python\n{fixed.strip()}\n```"

    return PYTHON_FENCE_RE.sub(repl, text)


def apply_python_fixes_to_blocks(
    blocks: List[List[Dict]],
    problem: str,
    expected_canon: str,
    expected_raw: Any,
    *,
    use_llm: bool,
    api_key: Optional[str],
    base_url: str,
    model: str,
    timeout: float,
    counters: Optional[PythonFenceCounters] = None,
) -> List[List[Dict]]:
    out: List[List[Dict]] = []
    for block in blocks:
        new_block: List[Dict] = []
        for m in block:
            mm = dict(m)
            if (mm.get("role") or "").lower() == "assistant":
                content = mm.get("content") or ""
                if PYTHON_FENCE_RE.search(content):
                    mm["content"] = fix_python_fences_in_text(
                        content,
                        problem,
                        expected_canon,
                        expected_raw,
                        use_llm=use_llm,
                        api_key=api_key,
                        base_url=base_url,
                        model=model,
                        timeout=timeout,
                        counters=counters,
                    )
            new_block.append(mm)
        out.append(new_block)
    return out


def apply_token_saving_pass(
    blocks: List[List[Dict]],
    problem: str,
    trace_answer_label: str,
    *,
    rejected_trace: bool = False,
    gold_answer_for_prompt: Optional[str] = None,
    summarize_non_python: bool,
    use_llm: bool,
    max_steps: int,
    min_chars: int,
    tool_max_chars: int,
    api_key: Optional[str],
    base_url: str,
    model: str,
    timeout: float,
    counters: SummarizeCounters,
) -> List[List[Dict]]:
    """
    Replace long non-python assistant prose with short summaries; truncate tools.
    On rejected traces, python fences are untouched (no prior LLM python pass).
    """
    out: List[List[Dict]] = []
    for block in blocks:
        new_block: List[Dict] = []
        for m in block:
            mm = dict(m)
            role = (mm.get("role") or "").lower()
            if role == "assistant":
                counters.assistant_seen += 1
                content = mm.get("content") or ""
                if PYTHON_FENCE_RE.search(content):
                    counters.skipped_python += 1
                elif summarize_non_python and len(content.strip()) >= min_chars:
                    summary: Optional[str] = None
                    if use_llm:
                        summary = llm_summarize_approach(
                            content,
                            problem,
                            trace_answer_label,
                            rejected_trace=rejected_trace,
                            gold_answer=gold_answer_for_prompt,
                            max_steps=max_steps,
                            api_key=api_key,
                            base_url=base_url,
                            model=model,
                            timeout=timeout,
                        )
                    if summary:
                        mm["content"] = summary
                        counters.summarized_llm_ok += 1
                    else:
                        mm["content"] = _fallback_trunc_reasoning(content)
                        if use_llm:
                            counters.summarized_llm_fail += 1
                        else:
                            counters.summarized_heuristic += 1
                elif len(content.strip()) < min_chars:
                    counters.skipped_short += 1
            elif role == "tool" and tool_max_chars > 0:
                tc = mm.get("content") or ""
                if len(tc) > tool_max_chars:
                    mm["content"] = tc[: tool_max_chars].rstrip() + "\n… [truncated]"
                    counters.tool_truncated += 1
            new_block.append(mm)
        out.append(new_block)
    return out


def ensure_boxed_suffix(text: str, expected_canon: str) -> str:
    t = normalize_ws(text)
    t = re.sub(r"\\boxed\{[^}]*\}\s*$", "", t).rstrip()
    if t:
        return t + f"\n\n\\boxed{{{expected_canon}}}"
    return f"\\boxed{{{expected_canon}}}"


def pick_messages_row(rows: List[Dict]) -> Optional[Dict]:
    for r in rows:
        if r.get("score_match") == "match" and r.get("messages"):
            return r
    for r in rows:
        if r.get("messages"):
            return r
    return None


def build_chosen_multi_turn(
    messages: List[Dict],
    problem: str,
    expected_canon: str,
    expected_raw: Any,
    first_k: int,
    last_l: int,
    *,
    use_llm_python: bool,
    api_key: Optional[str],
    base_url: str,
    model: str,
    timeout: float,
    summarize_non_python: bool,
    max_summary_steps: int,
    summarize_min_chars: int,
    tool_max_chars: int,
) -> Tuple[Optional[List[Dict[str, Any]]], PythonFenceCounters, SummarizeCounters]:
    blocks = assistant_blocks_from_messages(messages)
    if not blocks:
        return None, PythonFenceCounters(), SummarizeCounters()
    idxs = select_block_indices(len(blocks), first_k, last_l)
    selected = [blocks[i] for i in idxs]
    py_count = PythonFenceCounters()
    selected = apply_python_fixes_to_blocks(
        selected,
        problem,
        expected_canon,
        expected_raw,
        use_llm=use_llm_python,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        counters=py_count,
    )
    sum_count = SummarizeCounters()
    selected = apply_token_saving_pass(
        selected,
        problem,
        trace_answer_label=expected_canon,
        rejected_trace=False,
        gold_answer_for_prompt=None,
        summarize_non_python=summarize_non_python,
        use_llm=use_llm_python,
        max_steps=max_summary_steps,
        min_chars=summarize_min_chars,
        tool_max_chars=tool_max_chars,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        counters=sum_count,
    )
    harmony = blocks_to_harmony_chosen(selected)
    harmony = ensure_boxed_on_last_assistant(harmony, expected_canon)
    return harmony, py_count, sum_count


def build_rejected_harmony_multi_turn(
    messages: List[Dict],
    problem: str,
    wrong_canon: str,
    gold_canon: str,
    first_k: int,
    last_l: int,
    *,
    use_llm_summarize: bool,
    api_key: Optional[str],
    base_url: str,
    model: str,
    timeout: float,
    summarize_non_python: bool,
    max_summary_steps: int,
    summarize_min_chars: int,
    tool_max_chars: int,
) -> Tuple[List[Dict[str, Any]], SummarizeCounters]:
    """
    Same slice as chosen (first K + last L assistant blocks). No Python LLM fix — raw code kept.
    Then same token-saving pass as chosen but with rejected-oriented summarize prompts.
    """
    blocks = assistant_blocks_from_messages(messages)
    if not blocks:
        return [], SummarizeCounters()
    idxs = select_block_indices(len(blocks), first_k, last_l)
    selected = [blocks[i] for i in idxs]
    sum_count = SummarizeCounters()
    selected = apply_token_saving_pass(
        selected,
        problem,
        trace_answer_label=wrong_canon,
        rejected_trace=True,
        gold_answer_for_prompt=gold_canon,
        summarize_non_python=summarize_non_python,
        use_llm=use_llm_summarize,
        max_steps=max_summary_steps,
        min_chars=summarize_min_chars,
        tool_max_chars=tool_max_chars,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        counters=sum_count,
    )
    harmony = blocks_to_harmony_chosen(selected)
    harmony = ensure_boxed_on_last_assistant(harmony, wrong_canon)
    return harmony, sum_count


def harmony_single_assistant_boxed(boxed_content: str) -> List[Dict[str, Any]]:
    """Minimal Harmony rejected/chosen fallback: one final assistant with boxed string."""
    s = normalize_ws(boxed_content)
    if "\\boxed{" not in s:
        s = f"\\boxed{{{s}}}"
    return [{"role": "assistant", "content": s, "channel": "final"}]


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess multi-turn scored JSONL: chosen = Harmony message list (first K + last L blocks); optional LLM python fix."
    )
    ap.add_argument(
        "--input",
        default="/home/malam/wsl-tunix/aimo3_pivot20Feb2026/datasets/scored_combined_math.jsonl",
        help="Input JSONL (scored per-attempt, may include messages[])",
    )
    ap.add_argument(
        "--output",
        default="/home/malam/wsl-tunix/aimo3_pivot20Feb2026/datasets/training_samples_multi.jsonl",
        help="Output JSONL",
    )
    ap.add_argument(
        "--combined_math",
        default="/home/malam/wsl-tunix/aimo3_pivot20Feb2026/datasets/combined_math_crystal_hard50.jsonl",
        help="Combined math dataset; unparsed ids get boxed chosen + synthetic rejected (no multi-turn)",
    )
    ap.add_argument(
        "--parsed_tracking",
        default="/home/malam/wsl-tunix/aimo3_pivot20Feb2026/datasets/parsed_tracking.jsonl",
        help="Parsed ids; combined_math skips ids present here",
    )
    ap.add_argument("--first-turns", type=int, default=2, help="Number of assistant blocks to take from the start (default 2).")
    ap.add_argument("--last-turns", type=int, default=3, help="Number of assistant blocks to take from the end (default 3).")
    _default_vllm_base = (
        os.environ.get("VLLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_VLLM_BASE_URL
    )
    _default_fix_model = os.environ.get("VLLM_MODEL") or DEFAULT_VLLM_MODEL
    ap.add_argument(
        "--openai-api-key",
        default=os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY") or "",
        help="Bearer token for the chat server. Empty is OK for local vLLM without --api-key (uses EMPTY).",
    )
    ap.add_argument(
        "--openai-base-url",
        "--vllm-base-url",
        default=_default_vllm_base,
        dest="openai_base_url",
        help="OpenAI-compatible base URL (default: local vLLM http://127.0.0.1:8000/v1).",
    )
    ap.add_argument(
        "--fix-python-model",
        default=_default_fix_model,
        help="Model id served by vLLM/OpenAI (default: openai/gpt-oss-120b).",
    )
    ap.add_argument(
        "--fix-python-timeout",
        type=float,
        default=float(os.environ.get("VLLM_TIMEOUT", str(int(DEFAULT_VLLM_TIMEOUT_SEC)))),
        help="HTTP timeout seconds for python-fix calls (default 600 for large local models).",
    )
    ap.add_argument(
        "--no-llm-python-fix",
        action="store_true",
        help="Never call the API; only append ans/print(ans) to python fences.",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar for scored ids.",
    )
    ap.add_argument(
        "--quiet-python-samples",
        action="store_true",
        help="Do not print one line per id when python fences are detected (summary still printed).",
    )
    ap.add_argument(
        "--max-python-detail-json",
        type=int,
        default=2000,
        help="Max entries in python_samples_detail in the final JSON (0 = no cap).",
    )
    ap.add_argument(
        "--no-summarize-non-python",
        action="store_true",
        help="Keep full assistant text for turns without ```python (no LLM/heuristic compress).",
    )
    ap.add_argument(
        "--summary-max-steps",
        type=int,
        default=8,
        help="Max numbered steps in LLM summary of non-python reasoning (chosen only).",
    )
    ap.add_argument(
        "--summarize-min-chars",
        type=int,
        default=200,
        help="Assistant messages shorter than this are left as-is (chosen only).",
    )
    ap.add_argument(
        "--tool-output-max-chars",
        type=int,
        default=800,
        help="Truncate tool role content longer than this (0 = do not truncate).",
    )
    args = ap.parse_args()

    api_key = (args.openai_api_key or "").strip() or None
    use_llm = not args.no_llm_python_fix
    if use_llm:
        print(
            f"[PREF] Python-fix LLM: base={args.openai_base_url!r} model={args.fix_python_model!r} "
            f"bearer={'set' if api_key else 'EMPTY (vLLM default)'}"
        )
    else:
        print("[PREF] Python-fix LLM: disabled (--no-llm-python-fix)")

    parsed_ids: set = set()
    if Path(args.parsed_tracking).exists():
        with open(args.parsed_tracking, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get("id"):
                        parsed_ids.add(d["id"])
                except Exception:
                    pass
    print(f"[PREF] Parsed (tracked) ids: {len(parsed_ids)}")

    by_id: Dict[str, List[Dict]] = defaultdict(list)
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            by_id[d["id"]].append(d)

    out_rows = []
    skipped_no_messages = 0
    skipped_single_correct_attempt = 0
    skipped_no_wrong_attempt = 0
    scored_ids_total = len(by_id)
    python_agg = PythonFenceCounters()
    summarize_chosen_agg = SummarizeCounters()
    summarize_rejected_agg = SummarizeCounters()
    python_sample_records: List[Dict[str, Any]] = []
    summarize_non_python = not args.no_summarize_non_python

    scored_items = list(by_id.items())
    bar = tqdm(
        scored_items,
        total=len(scored_items),
        desc="Scored ids",
        unit="id",
        disable=args.no_progress,
        file=sys.stderr,
    )
    processed_ok = 0
    for id_, rows in bar:
        rows = sorted(rows, key=lambda x: x.get("attempt", 0))
        if not rows:
            continue
        first = rows[0]
        problem = normalize_ws(first.get("problem") or "")
        if not problem and first.get("messages"):
            for m in first.get("messages", []):
                if (m.get("role") or "").lower() == "user":
                    problem = normalize_ws(m.get("content") or "")
                    break
        if not problem:
            continue
        expected_raw = first.get("answer")
        expected_canon = canon_answer(expected_raw)
        if expected_canon is None:
            continue

        if len(rows) == 1:
            p0 = row_prediction_canon(rows[0])
            if p0 is not None and p0 == expected_canon:
                skipped_single_correct_attempt += 1
                continue

        if find_first_wrong_attempt_row(rows, expected_raw) is None:
            skipped_no_wrong_attempt += 1
            continue

        src = pick_messages_row(rows)
        messages = (src or {}).get("messages") if src else None
        if not messages:
            skipped_no_messages += 1
            continue

        chosen, py_count, sum_count = build_chosen_multi_turn(
            messages,
            problem,
            expected_canon,
            expected_raw,
            args.first_turns,
            args.last_turns,
            use_llm_python=use_llm,
            api_key=api_key,
            base_url=args.openai_base_url,
            model=args.fix_python_model,
            timeout=args.fix_python_timeout,
            summarize_non_python=summarize_non_python,
            max_summary_steps=args.summary_max_steps,
            summarize_min_chars=args.summarize_min_chars,
            tool_max_chars=args.tool_output_max_chars,
        )
        if not chosen:
            skipped_no_messages += 1
            continue

        summarize_chosen_agg.assistant_seen += sum_count.assistant_seen
        summarize_chosen_agg.skipped_python += sum_count.skipped_python
        summarize_chosen_agg.skipped_short += sum_count.skipped_short
        summarize_chosen_agg.summarized_llm_ok += sum_count.summarized_llm_ok
        summarize_chosen_agg.summarized_llm_fail += sum_count.summarized_llm_fail
        summarize_chosen_agg.summarized_heuristic += sum_count.summarized_heuristic
        summarize_chosen_agg.tool_truncated += sum_count.tool_truncated

        if py_count.total > 0:
            python_agg.total += py_count.total
            python_agg.llm_ok += py_count.llm_ok
            python_agg.llm_fail += py_count.llm_fail
            python_agg.no_llm_mode += py_count.no_llm_mode
            rec = {
                "id": id_,
                "fences": py_count.total,
                "llm_ok": py_count.llm_ok,
                "llm_fail": py_count.llm_fail,
                "no_llm_fences": py_count.no_llm_mode,
            }
            python_sample_records.append(rec)
            if not args.quiet_python_samples:
                print(
                    f"[PREF] python id={id_} fences={py_count.total} "
                    f"llm_ok={py_count.llm_ok} llm_fail={py_count.llm_fail} "
                    f"no_llm_fences={py_count.no_llm_mode}",
                    flush=True,
                )

        rejected_str = get_rejected_boxed(rows, expected_raw)
        wrong_canon = boxed_inner_canon(rejected_str) or ""
        if wrong_canon == expected_canon:
            syn = synthetic_rejected_answer(expected_raw)
            rejected_str = f"\\boxed{{{syn}}}" if syn is not None else "\\boxed{0}"
            wrong_canon = boxed_inner_canon(rejected_str) or ""

        wrong_row = find_first_wrong_attempt_row(rows, expected_raw)
        sum_rej = SummarizeCounters()
        if wrong_row and wrong_row.get("messages"):
            rej_harm, sum_rej = build_rejected_harmony_multi_turn(
                wrong_row["messages"],
                problem,
                wrong_canon or "0",
                expected_canon,
                args.first_turns,
                args.last_turns,
                use_llm_summarize=use_llm,
                api_key=api_key,
                base_url=args.openai_base_url,
                model=args.fix_python_model,
                timeout=args.fix_python_timeout,
                summarize_non_python=summarize_non_python,
                max_summary_steps=args.summary_max_steps,
                summarize_min_chars=args.summarize_min_chars,
                tool_max_chars=args.tool_output_max_chars,
            )
            rejected = rej_harm if rej_harm else harmony_single_assistant_boxed(rejected_str)
        else:
            rejected = harmony_single_assistant_boxed(rejected_str)

        summarize_rejected_agg.assistant_seen += sum_rej.assistant_seen
        summarize_rejected_agg.skipped_python += sum_rej.skipped_python
        summarize_rejected_agg.skipped_short += sum_rej.skipped_short
        summarize_rejected_agg.summarized_llm_ok += sum_rej.summarized_llm_ok
        summarize_rejected_agg.summarized_llm_fail += sum_rej.summarized_llm_fail
        summarize_rejected_agg.summarized_heuristic += sum_rej.summarized_heuristic
        summarize_rejected_agg.tool_truncated += sum_rej.tool_truncated

        prompt = build_prompt(problem)
        meta = {
            "domain": "math",
            "source": "aimo3",
            "id": id_,
            "first_turns": args.first_turns,
            "last_turns": args.last_turns,
            "llm_python_fix": use_llm,
            "chosen_format": "harmony",
            "rejected_format": "harmony",
            "summarize_non_python": summarize_non_python,
            "rejected_from_wrong_attempt_messages": bool(wrong_row and wrong_row.get("messages")),
        }
        if py_count.total > 0:
            meta["python_fences"] = py_count.total
            meta["python_llm_ok"] = py_count.llm_ok
            meta["python_llm_fail"] = py_count.llm_fail
        if (
            sum_count.summarized_llm_ok
            + sum_count.summarized_llm_fail
            + sum_count.summarized_heuristic
            + sum_count.tool_truncated
        ) > 0:
            meta["chosen_text_compressed"] = True
            meta["chosen_summary_llm_ok"] = sum_count.summarized_llm_ok
            meta["chosen_summary_llm_fail"] = sum_count.summarized_llm_fail
            meta["chosen_summary_heuristic"] = sum_count.summarized_heuristic
            meta["chosen_tool_truncated"] = sum_count.tool_truncated
        if (
            sum_rej.summarized_llm_ok
            + sum_rej.summarized_llm_fail
            + sum_rej.summarized_heuristic
            + sum_rej.tool_truncated
        ) > 0:
            meta["rejected_text_compressed"] = True
            meta["rejected_summary_llm_ok"] = sum_rej.summarized_llm_ok
            meta["rejected_summary_llm_fail"] = sum_rej.summarized_llm_fail
            meta["rejected_summary_heuristic"] = sum_rej.summarized_heuristic
            meta["rejected_tool_truncated"] = sum_rej.tool_truncated
        out_rows.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "meta": meta,
        })
        processed_ok += 1
        if not args.no_progress:
            bar.set_postfix(written=processed_ok, python_ids=len(python_sample_records))

    added_from_combined = 0
    if Path(args.combined_math).exists():
        with open(args.combined_math, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    id_ = d.get("id")
                    if not id_ or id_ in parsed_ids:
                        continue
                    problem = normalize_ws(d.get("problem") or "")
                    if not problem:
                        continue
                    expected_raw = d.get("answer")
                    expected_canon = canon_answer(expected_raw)
                    if expected_canon is None:
                        continue
                    chosen = f"\\boxed{{{expected_canon}}}"
                    syn = synthetic_rejected_answer(expected_raw)
                    rejected = f"\\boxed{{{syn}}}" if syn is not None else "\\boxed{0}"
                    if canon_answer(rejected) == expected_canon:
                        continue
                    prompt = build_prompt(problem)
                    meta = {"domain": "math", "source": "aimo3", "id": id_}
                    out_rows.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "meta": meta,
                    })
                    added_from_combined += 1
                except Exception as e:
                    print(f"[WARN] combined_math line: {e}")
                    continue
    print(f"[PREF] Added from combined_math (unparsed ids): {added_from_combined}")
    if skipped_no_messages:
        print(f"[PREF] Skipped scored ids (no messages / empty chosen): {skipped_no_messages}")
    if skipped_single_correct_attempt:
        print(f"[PREF] Skipped ids (single attempt, prediction matches gold): {skipped_single_correct_attempt}")
    if skipped_no_wrong_attempt:
        print(f"[PREF] Skipped ids (no wrong attempt; would be synthetic rejected only): {skipped_no_wrong_attempt}")

    print(
        f"[PREF] Python fences (selected turns): samples={len(python_sample_records)} "
        f"fences_total={python_agg.total} llm_ok={python_agg.llm_ok} "
        f"llm_fail={python_agg.llm_fail} no_llm_fences={python_agg.no_llm_mode}",
        flush=True,
    )
    print(
        f"[PREF] Chosen compress (non-python + tools): summarize_on={summarize_non_python} "
        f"assistants_seen={summarize_chosen_agg.assistant_seen} skip_python={summarize_chosen_agg.skipped_python} "
        f"skip_short={summarize_chosen_agg.skipped_short} sum_llm_ok={summarize_chosen_agg.summarized_llm_ok} "
        f"sum_llm_fail={summarize_chosen_agg.summarized_llm_fail} sum_heuristic={summarize_chosen_agg.summarized_heuristic} "
        f"tool_trunc={summarize_chosen_agg.tool_truncated}",
        flush=True,
    )
    print(
        f"[PREF] Rejected compress (non-python + tools; no python LLM fix): summarize_on={summarize_non_python} "
        f"assistants_seen={summarize_rejected_agg.assistant_seen} skip_python={summarize_rejected_agg.skipped_python} "
        f"skip_short={summarize_rejected_agg.skipped_short} sum_llm_ok={summarize_rejected_agg.summarized_llm_ok} "
        f"sum_llm_fail={summarize_rejected_agg.summarized_llm_fail} sum_heuristic={summarize_rejected_agg.summarized_heuristic} "
        f"tool_trunc={summarize_rejected_agg.tool_truncated}",
        flush=True,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.max_python_detail_json == 0:
        detail_json = python_sample_records
        detail_omitted = 0
    elif len(python_sample_records) > args.max_python_detail_json:
        detail_json = python_sample_records[: args.max_python_detail_json]
        detail_omitted = len(python_sample_records) - args.max_python_detail_json
        print(
            f"[PREF] python_samples_detail truncated in JSON: showing {len(detail_json)} of {len(python_sample_records)}",
            flush=True,
        )
    else:
        detail_json = python_sample_records
        detail_omitted = 0

    print(json.dumps({
        "input": args.input,
        "output": args.output,
        "first_turns": args.first_turns,
        "last_turns": args.last_turns,
        "llm_python_fix": use_llm,
        "skipped_no_messages": skipped_no_messages,
        "skipped_single_correct_attempt": skipped_single_correct_attempt,
        "skipped_no_wrong_attempt": skipped_no_wrong_attempt,
        "scored_ids_in_input": scored_ids_total,
        "num_records": len(out_rows),
        "from_scored": len(out_rows) - added_from_combined,
        "from_combined_math_unparsed": added_from_combined,
        "python_samples_with_fences": len(python_sample_records),
        "python_fences_total": python_agg.total,
        "python_llm_ok": python_agg.llm_ok,
        "python_llm_fail": python_agg.llm_fail,
        "python_no_llm_fences": python_agg.no_llm_mode,
        "python_samples_detail": detail_json,
        "python_samples_detail_omitted": detail_omitted,
        "summarize_non_python": summarize_non_python,
        "summary_max_steps": args.summary_max_steps,
        "summarize_min_chars": args.summarize_min_chars,
        "tool_output_max_chars": args.tool_output_max_chars,
        "chosen_compress_assistants_seen": summarize_chosen_agg.assistant_seen,
        "chosen_compress_skipped_python": summarize_chosen_agg.skipped_python,
        "chosen_compress_skipped_short": summarize_chosen_agg.skipped_short,
        "chosen_summarize_llm_ok": summarize_chosen_agg.summarized_llm_ok,
        "chosen_summarize_llm_fail": summarize_chosen_agg.summarized_llm_fail,
        "chosen_summarize_heuristic": summarize_chosen_agg.summarized_heuristic,
        "chosen_tool_truncated": summarize_chosen_agg.tool_truncated,
        "rejected_compress_assistants_seen": summarize_rejected_agg.assistant_seen,
        "rejected_compress_skipped_python": summarize_rejected_agg.skipped_python,
        "rejected_compress_skipped_short": summarize_rejected_agg.skipped_short,
        "rejected_summarize_llm_ok": summarize_rejected_agg.summarized_llm_ok,
        "rejected_summarize_llm_fail": summarize_rejected_agg.summarized_llm_fail,
        "rejected_summarize_heuristic": summarize_rejected_agg.summarized_heuristic,
        "rejected_tool_truncated": summarize_rejected_agg.tool_truncated,
    }, indent=2))


if __name__ == "__main__":
    main()
