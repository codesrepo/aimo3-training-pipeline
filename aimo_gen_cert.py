#!/usr/bin/env python3
"""
Certificate extraction from your base dataset using a locally hosted OpenAI-compatible LLM.

Usage:
  python extract_certs.py --input /path/to/base.jsonl --output certs.jsonl
  python extract_certs.py --input /path/to/base.json  --output certs.jsonl

Assumptions about input (matches your sample):
Each record is a JSON object with keys like:
  - "problem" (string)
  - "expected_answer" (number/string)
  - "attempts" (list of objects with "attempt", "predicted_answer", "response_text", optional "reflection_avoid_memo")
  - optional: "avoid_memo_final", "oracle_trace", "solved_attempt"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from openai import OpenAI
from tqdm import tqdm

# --- Your local GPT-OSS config (as requested)
BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "openai/gpt-oss-120b"


# ----------------------------
# Config
# ----------------------------
@dataclass
class CFG:
    temperature: float = 0.0
    max_output_tokens: int = 8192 
    retries: int = 3
    retry_backoff_s: float = 1.5
    request_timeout_s: float = 120.0
    # If your server doesn't support chat.completions, set True to use legacy completions API.
    force_legacy_completions: bool = False


CERT_SCHEMA_DESCRIPTION = r"""
Output MUST be exactly one valid JSON object, nothing else. No markdown, no ```json```, no explanation before or after.

CRITICAL: Reply with ONLY the raw JSON object. Do not wrap in code blocks or add any text.

{
  "problem": string,               // stable id (we provide it, keep as-is)
  "final_answer": string|number,      // must equal expected_answer
  "solved_attempt": number|null,      // attempt index that solved (1-based), or null if unknown
  "key_idea": string,                 // 1-2 sentences
  "proof_skeleton": [string, ...],    // 4-12 bullet-like steps, each a short sentence
  "attainment_or_example": string,    // equality case / example / "N/A"
  "sanity_checks": [string, ...],     // 0-5 items, may be empty

  "failure": {                        // required even if no wrong attempt; use "none"
    "primary_error_type": string,     // one of: misread_problem | counting_ordering | boundary_condition |
                                      // unjustified_assumption | off_by_one | invalid_domain |
                                      // wrong_geometry_config | missing_case | computational_infeasible |
                                      // algebra_slip | none
    "error_localization": string,     // exact mistaken step in words
    "minimal_fix": string             // smallest correction
  }
}

Rules:
- If there is at least one wrong attempt before the solved attempt, analyze ONLY the earliest wrong attempt.
- Prefer oracle_trace for correctness if present. If avoid_memo_final is truncated mid-sentence, ignore it.
- Keep it concise, verifiable, and aligned with the record.
- Do not invent facts not supported by the record.
- Output ONLY the JSON object. No preamble, no "Here is the JSON:", no ```.
""".strip()


def stable_problem(problem_text: str) -> str:
    h = problem_text
    return h


def looks_truncated(s: Optional[str]) -> bool:
    if not s:
        return False
    s = s.strip()
    # Heuristic: ends mid-word or with obvious truncation markers
    if len(s) < 40:
        return False
    if re.search(r"(Adjus|Systematic$|betwee$|Revised\s*$)", s):
        return True
    # Ends without punctuation and last token is short -> often truncated
    if s[-1] not in ".!?]})\"" and len(s.split()[-1]) <= 6:
        return True
    return False


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []

    # Detect JSONL vs JSON list/object
    if raw[0] == "[":
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list at top-level.")
        return data

    # JSONL - handle cases where multiple JSON objects are on the same line
    records = []
    decoder = json.JSONDecoder()
    for line_num, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        # Parse all JSON objects from this line (handles concatenated objects)
        idx = 0
        while idx < len(line):
            # Skip whitespace
            while idx < len(line) and line[idx].isspace():
                idx += 1
            if idx >= len(line):
                break
            try:
                obj, new_idx = decoder.raw_decode(line, idx=idx)
                records.append(obj)
                idx = new_idx
            except json.JSONDecodeError as e:
                print(f"[WARNING] Line {line_num}: Failed to parse JSON at position {idx}: {e}")
                print(f"  Context: {repr(line[max(0, idx-50):idx+50])}")
                break  # Skip this line
    return records


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_card_from_cert(cert: dict) -> dict:
    """
    Build a compact hint card from a certificate. Keeps the original problem.
    """
    card = {
        "problem": cert.get("problem"),
        "id": cert.get("id") or (cert.get("problem", "")[:40].replace(" ", "_")),
        "a": cert.get("final_answer"),
        "idea": (cert.get("key_idea") or "").strip(),
        "steps": cert.get("proof_skeleton") or [],
        "pitfall": "",
    }

    # Extract pitfall from failure dict (preferred: minimal_fix, else error_localization)
    fail = cert.get("failure") or {}
    if isinstance(fail, dict):
        pit = fail.get("minimal_fix") or fail.get("error_localization") or ""
        card["pitfall"] = " ".join(str(pit).split())

    # Normalize steps to a short list of strings
    steps = card["steps"]
    if isinstance(steps, str):
        steps = [steps]
    elif not isinstance(steps, list):
        steps = [str(steps)]

    # Keep only first 4 steps and collapse whitespace
    card["steps"] = [" ".join(str(s).split()) for s in steps[:4] if str(s).strip()]

    return card


def cert_has_hint_content(cert: dict) -> bool:
    """True if the cert yields at least one of idea, steps, or pitfall non-empty."""
    card = make_card_from_cert(cert)
    has_idea = bool((card.get("idea") or "").strip())
    has_steps = bool(card.get("steps"))
    has_pitfall = bool((card.get("pitfall") or "").strip())
    return has_idea or has_steps or has_pitfall


def build_extraction_payload(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construct the content we feed to the LLM. We strip obviously-truncated avoid_memo_final.
    """
    problem = rec.get("problem", "")
    attempts = rec.get("attempts", []) or []
    expected_answer = rec.get("expected_answer", None)
    oracle_trace = rec.get("oracle_trace", None)

    avoid_memo_final = rec.get("avoid_memo_final", None)
    if looks_truncated(avoid_memo_final):
        avoid_memo_final = None

    solved_attempt = rec.get("solved_attempt", None)

    return {
        "problem": problem,
        "expected_answer": expected_answer,
        "solved_attempt": solved_attempt,
        "attempts": [
            {
                "attempt": a.get("attempt", None),
                "predicted_answer": a.get("predicted_answer", None),
                "response_text": a.get("response_text", None),
                "reflection_avoid_memo": a.get("reflection_avoid_memo", None),
            }
            for a in attempts
        ],
        "avoid_memo_final": avoid_memo_final,
        "oracle_trace": oracle_trace,
    }


def compute_solved_attempt_index(rec: Dict[str, Any]) -> Optional[int]:
    """
    If record has solved_attempt, trust it. Else infer from attempts matching expected_answer.
    Attempts are 1-based in your sample.
    """
    if isinstance(rec.get("solved_attempt", None), int):
        return rec["solved_attempt"]

    expected = rec.get("expected_answer", None)
    attempts = rec.get("attempts", []) or []
    for a in attempts:
        if a.get("predicted_answer", None) == expected:
            # attempt field in your sample is already 1-based
            if isinstance(a.get("attempt", None), int):
                return a["attempt"]
    return None


def fix_common_json_issues(text: str) -> str:
    """Try to fix common JSON syntax issues."""
    # Replace single quotes with double quotes (but be careful with apostrophes in strings)
    # Only replace at boundaries, not inside words
    text = re.sub(r"'(\w+)':", r'"\1":', text)  # Keys: 'key': -> "key":
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)  # String values: : 'value' -> : "value"
    
    # Remove trailing commas more aggressively
    text = re.sub(r",\s*([}\]])", r"\1", text)
    
    return text


def repair_truncated_json(raw: str) -> str:
    """
    Attempt to repair JSON truncated mid-output (e.g. hit max_tokens).
    Closes unclosed strings, then arrays, then objects.
    """
    start = raw.find("{")
    if start == -1:
        return raw
    s = raw[start:]
    out: List[str] = []
    i = 0
    in_string = None  # '"' or "'"
    escape = False
    stack: List[str] = []  # '[' or '{'

    while i < len(s):
        c = s[i]
        if escape:
            escape = False
            out.append(c)
            i += 1
            continue
        if c == "\\" and in_string:
            escape = True
            out.append(c)
            i += 1
            continue
        if in_string:
            if c == in_string:
                in_string = None
            out.append(c)
            i += 1
            continue
        if c in ('"', "'"):
            in_string = c
            out.append(c)
            i += 1
            continue
        if c == "{":
            stack.append("}")
            out.append(c)
            i += 1
            continue
        if c == "}":
            if stack and stack[-1] == "}":
                stack.pop()
            out.append(c)
            i += 1
            continue
        if c == "[":
            stack.append("]")
            out.append(c)
            i += 1
            continue
        if c == "]":
            if stack and stack[-1] == "]":
                stack.pop()
            out.append(c)
            i += 1
            continue
        out.append(c)
        i += 1

    # Truncated: close unclosed string, then brackets in reverse order
    if in_string:
        out.append(in_string)
    while stack:
        out.append(stack.pop())

    return "".join(out)


def json_extract_best_effort(text: str) -> Dict[str, Any]:
    """
    Robust JSON extraction:
      - Try direct json.loads
      - Strip ```json ... ``` or ``` ... ``` code fences
      - Take substring between first '{' and matching last '}' (respect nesting)
      - Relax trailing commas before ] or }
    """
    if not text or not isinstance(text, str):
        raise ValueError("Model did not return valid JSON.")

    raw = text.strip()

    # Remove markdown code fences (```json ... ``` or ``` ... ```)
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL | re.IGNORECASE)
    if m:
        raw = m.group(1).strip()

    # Strip common prefixes (e.g. "Here is the JSON:", "Answer:")
    for prefix in (
        r"^(?:here'?s?|here is)\s+the\s+json\s*:?\s*",
        r"^answer\s*:?\s*",
        r"^json\s*:?\s*",
        r"^output\s*:?\s*",
        r"^response\s*:?\s*",
    ):
        raw = re.sub(prefix, "", raw, flags=re.IGNORECASE)
        raw = raw.strip()

    # Try direct parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
        # Model returned a JSON string containing the object
        if isinstance(obj, str):
            obj2 = json.loads(obj)
            if isinstance(obj2, dict):
                return obj2
    except Exception:
        pass

    # Find JSON object: first '{' then matching '}' (simple brace count)
    start = raw.find("{")
    if start == -1:
        raise ValueError("Model did not return valid JSON.")
    depth = 0
    in_string = None
    escape = False
    end = -1
    i = start
    while i < len(raw):
        c = raw[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == "\\" and in_string:
            escape = True
            i += 1
            continue
        if in_string:
            if c == in_string:
                in_string = None
            i += 1
            continue
        if c in ('"', "'"):
            in_string = c
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
        i += 1

    if end != -1:
        sub = raw[start : end + 1]
        # Relax trailing commas before ] or }
        sub = re.sub(r",\s*([}\]])", r"\1", sub)
        # Fix common issues: unescaped newlines in strings, single quotes
        sub = fix_common_json_issues(sub)
        try:
            return json.loads(sub)
        except Exception:
            pass

    # Fallback: first '{' to last '}'
    m2 = raw.rfind("}")
    if m2 > start:
        sub = raw[start : m2 + 1]
        sub = re.sub(r",\s*([}\]])", r"\1", sub)
        sub = fix_common_json_issues(sub)
        try:
            return json.loads(sub)
        except Exception:
            pass

    # Truncated JSON: model hit max_tokens mid-output. Repair by closing strings/arrays/objects.
    try:
        repaired = repair_truncated_json(raw)
        repaired = fix_common_json_issues(repaired)
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Last resort: try to find any JSON-like structure and fix it
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, raw, re.DOTALL)
    for match in reversed(matches):  # Try longest matches first
        try:
            fixed = fix_common_json_issues(match)
            obj = json.loads(fixed)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    raise ValueError("Model did not return valid JSON.")


def validate_certificate(obj: Dict[str, Any], problem: str, expected_answer: Any) -> Dict[str, Any]:
    """
    Enforce required keys and normalize a bit.
    Fill missing keys (e.g. from truncated model output) with defaults.
    """
    required_top = [
        "problem",
        "final_answer",
        "solved_attempt",
        "key_idea",
        "proof_skeleton",
        "attainment_or_example",
        "sanity_checks",
        "failure",
    ]
    defaults = {
        "problem": problem,
        "final_answer": expected_answer,
        "solved_attempt": None,
        "key_idea": "",
        "proof_skeleton": [],
        "attainment_or_example": "N/A",
        "sanity_checks": [],
        "failure": {"primary_error_type": "none", "error_localization": "", "minimal_fix": ""},
    }
    for k in required_top:
        if k not in obj:
            obj[k] = defaults[k]

    obj["problem"] = problem
    obj["final_answer"] = expected_answer  # hard-enforce exact expected answer

    if not isinstance(obj["proof_skeleton"], list):
        obj["proof_skeleton"] = [str(obj["proof_skeleton"])] if obj.get("proof_skeleton") else []

    if not isinstance(obj["sanity_checks"], list):
        obj["sanity_checks"] = []

    # failure object normalization
    if not isinstance(obj["failure"], dict):
        obj["failure"] = {"primary_error_type": "none", "error_localization": "", "minimal_fix": ""}

    for fk in ["primary_error_type", "error_localization", "minimal_fix"]:
        obj["failure"].setdefault(fk, "")

    # If user has no wrong attempts, allow "none"
    allowed = {
        "misread_problem",
        "counting_ordering",
        "boundary_condition",
        "unjustified_assumption",
        "off_by_one",
        "invalid_domain",
        "wrong_geometry_config",
        "missing_case",
        "computational_infeasible",
        "algebra_slip",
        "none",
    }
    if obj["failure"]["primary_error_type"] not in allowed:
        obj["failure"]["primary_error_type"] = "none"

    return obj


def call_llm_chat(client: OpenAI, system: str, user: str, cfg: CFG) -> str:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=cfg.temperature,
        max_tokens=cfg.max_output_tokens,
        timeout=cfg.request_timeout_s,
    )
    return completion.choices[0].message.content or ""


def call_llm_legacy_completions(client: OpenAI, prompt: str, cfg: CFG) -> str:
    # Works on servers implementing /v1/completions
    completion = client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        temperature=cfg.temperature,
        max_tokens=cfg.max_output_tokens,
        stream=False,
        timeout=cfg.request_timeout_s,
    )
    return completion.choices[0].text or ""


def extract_one_certificate(client: OpenAI, rec: Dict[str, Any], cfg: CFG) -> Optional[Dict[str, Any]]:
    problem = rec.get("problem", "")
    expected = rec.get("expected_answer", None)
    pid = stable_problem(problem)

    solved_attempt = compute_solved_attempt_index(rec)
    payload = build_extraction_payload(rec)

    system_msg = (
        "You are a precise information extraction engine. "
        "You must respond with ONLY a single valid JSON object—no other text, no markdown, no code blocks. "
        "Do not wrap the JSON in ``` or add any explanation. Output the raw JSON object only."
    )

    user_msg = (
        f"{CERT_SCHEMA_DESCRIPTION}\n\n"
        f"problem (must be preserved): {pid}\n\n"
        f"RECORD_JSON:\n{json.dumps({**payload, 'problem': pid, 'solved_attempt': solved_attempt}, ensure_ascii=False)}"
    )

    last_err: Optional[Exception] = None
    last_output: Optional[str] = None
    for attempt in range(cfg.retries):
        try:
            if cfg.force_legacy_completions:
                prompt = f"System:\n{system_msg}\n\nUser:\n{user_msg}\n"
                out = call_llm_legacy_completions(client, prompt, cfg)
            else:
                out = call_llm_chat(client, system_msg, user_msg, cfg)

            last_output = out  # Save for debugging
            obj = json_extract_best_effort(out)
            obj = validate_certificate(obj, pid, expected)
            # prefer our solved_attempt inference if model returns nonsense
            obj["solved_attempt"] = solved_attempt if solved_attempt is not None else obj.get("solved_attempt", None)
            return obj
        except Exception as e:
            last_err = e
            if attempt < cfg.retries - 1:
                print(f"[WARNING] Attempt {attempt + 1}/{cfg.retries} failed for {pid}: {e}")
                if last_output:
                    print(f"[DEBUG] Raw output (first 500 chars): {last_output[:500]}")
            time.sleep(cfg.retry_backoff_s * (attempt + 1))

    # On final failure, return None instead of raising (caller will skip)
    error_msg = f"Failed to extract certificate for {pid}: {last_err}"
    if last_output:
        error_msg += f"\nRaw model output (first 1000 chars):\n{last_output[:1000]}"
        # Try to save problematic output to a debug file
        debug_file = f"/tmp/aimo_cert_debug_{pid}.txt"
        try:
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(f"Problem ID: {pid}\n")
                f.write(f"Error: {last_err}\n")
                f.write(f"Raw output:\n{last_output}\n")
            print(f"[DEBUG] Saved problematic output to {debug_file}")
        except Exception:
            pass
    print(f"[SKIP] {error_msg}")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default = '/home/malam/wsl-tunix/imo/openmath_data/oracle_traces_no_match.jsonl',help="Path to base dataset (.jsonl or .json list)")
    ap.add_argument("--output", default = '/home/malam/wsl-tunix/imo/openmath_data/aimo_certs.jsonl',help="Path to write certificates (jsonl)")
    ap.add_argument("--max-records", type=int, default=0, help="Process only first N records (0 = all)")
    ap.add_argument("--force-legacy", action="store_true", help="Use /v1/completions instead of chat.completions")
    ap.add_argument("--timeout", type=float, default=120.0, help="Request timeout seconds")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1)
    args = ap.parse_args()

    cfg = CFG(
        temperature=args.temperature,
        retries=args.retries,
        request_timeout_s=args.timeout,
        force_legacy_completions=args.force_legacy,
    )

    client = OpenAI(
        base_url=BASE_URL,
        api_key="sk-local",
        timeout=cfg.request_timeout_s,
    )

    records = load_records(args.input)
    if args.max_records and args.max_records > 0:
        records = records[: args.max_records]

    out_rows: List[Dict[str, Any]] = []
    skipped_count = 0
    for rec in tqdm(records, desc="Extracting certificates"):
        cert = extract_one_certificate(client, rec, cfg)
        if cert is None:
            skipped_count += 1
            continue
        out_rows.append(cert)

    # Keep only certs that have at least one of idea, steps, or pitfall non-empty
    filtered_certs = [c for c in out_rows if cert_has_hint_content(c)]
    dropped_by_filter = len(out_rows) - len(filtered_certs)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    save_jsonl(args.output, filtered_certs)

    # Build hint cards from filtered certificates (same dir as certs)
    hint_cards_path = os.path.join(out_dir, "aimo_hint_cards.jsonl")
    hint_cards = [make_card_from_cert(cert) for cert in filtered_certs]
    save_jsonl(hint_cards_path, hint_cards)

    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Total records processed: {len(records)}")
    print(f"  Certificates extracted (before filter): {len(out_rows)}")
    print(f"  Dropped (no idea/steps/pitfall): {dropped_by_filter}")
    print(f"  Skipped (invalid JSON): {skipped_count}")
    print(f"  Wrote {len(filtered_certs)} certificates to {args.output}")
    print(f"  Wrote {len(hint_cards)} hint cards to {hint_cards_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
