#!/usr/bin/env python3
"""
Score math problems with tool calling (Harmony encoding).
Input: combined_math_crystal_hard50.jsonl (id, source, answer, problem, solution, datatype).
Output: scored_combined_math.jsonl. Parsed IDs tracked in parsed_tracking.jsonl for resume.
"""

import json
import time
import re
import os
import sys
import argparse
from typing import Any
import queue
import contextlib
import threading
from pathlib import Path
import httpx
# Optional dependencies:
# - `datasets` is only required for --mode score_dataset
# - `tqdm` is optional (we fall back to plain iteration)
try:
    from datasets import load_dataset, Dataset  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore
    Dataset = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **_kwargs):  # type: ignore
        return x

from concurrent.futures import ThreadPoolExecutor, as_completed

# Add imo directory to path for imports
imo_dir = os.path.dirname(os.path.abspath(__file__))
if imo_dir not in sys.path:
    sys.path.insert(0, imo_dir)

try:
    from openai import OpenAI  # type: ignore
    from openai_harmony import (  # type: ignore
        HarmonyEncodingName,
        load_harmony_encoding,
        SystemContent,
        ReasoningEffort,
        ToolNamespaceConfig,
        Author,
        Message,
        Role,
        TextContent,
        Conversation,
    )
    from jupyter_client import KernelManager, BlockingKernelClient  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    HarmonyEncodingName = None  # type: ignore
    load_harmony_encoding = None  # type: ignore
    SystemContent = None  # type: ignore
    ReasoningEffort = None  # type: ignore
    ToolNamespaceConfig = None  # type: ignore
    Author = None  # type: ignore
    Message = None  # type: ignore
    Role = None  # type: ignore
    TextContent = None  # type: ignore
    Conversation = None  # type: ignore
    KernelManager = None  # type: ignore
    BlockingKernelClient = None  # type: ignore

# GPT-OSS model configuration
BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "openai/gpt-oss-120b"

# Paths: input = combined_math_crystal_hard50.jsonl; output and tracking in same dir
SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR / "datasets"
INPUT_JSONL = DATASETS_DIR / "combined_math_crystal_hard50.jsonl"
PARSED_TRACKING_JSONL = DATASETS_DIR / "parsed_tracking.jsonl"
OUTPUT_JSONL = DATASETS_DIR / "scored_combined_math.jsonl"

# Configuration
BATCH_SIZE = 100000
GPU_BATCH_SIZE = 1
MAX_ATTEMPTS = 3  # Max attempts per problem; stop when prediction matches answer

# Configuration class (matching notebook)
class CFG:
    system_prompt = (
        'You are the top International Mathematical Olympiad (IMO) competitor. '
        'The final answer must be a non-negative integer between 0 and 99999. '
        'You must place the final integer answer inside \\boxed{}.'
    )
    
    tool_prompt = (
        'Use this tool to execute Python code. '
        'The environment is a stateful Jupyter notebook. '
        'You must use print() to output results.'
    )
    
    preference_prompt =  (
        'Please reason step by step and use the python tool to solve the math problem. '
        'For extremely large numbers, find patterns from small cases instead of direct computation. '
        'Finally, Return only the verified final answer in \\boxed{}, where the answer is an integer in [0, 99999]. Never guess.'
    )   
    # API settings
    served_model_name = 'gpt-oss'
    temperature = 1.0
    top_logprobs = 1
    min_p = 0.0
    seed = 42
    context_tokens = 32768
    buffer_tokens = 512
    turns = 128
    jupyter_timeout = 300.0
    sandbox_timeout = 10.0
    session_timeout = 1800

# AIMO3Template class (from notebook)
class AIMO3Template:
    def __init__(self):
        pass

    def get_system_content(self, system_prompt: str, tool_config: ToolNamespaceConfig) -> SystemContent:
        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(
        self,
        system_prompt: str,
        user_input: str,
        tool_config: ToolNamespaceConfig
    ) -> list[Message]:
        system_content = self.get_system_content(system_prompt, tool_config)
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
        user_content = TextContent(text=user_input)
        user_message = Message.from_role_and_content(Role.USER, user_content)
        return [system_message, user_message]

# AIMO3Sandbox class (from notebook)
class AIMO3Sandbox:
    _port_lock = threading.Lock()
    _next_port = 50000

    def __init__(self, timeout: float = 300.0):
        self._default_timeout = timeout
        self._km = None
        self._client = None
        self._owns_kernel = False

        with AIMO3Sandbox._port_lock:
            start_port = AIMO3Sandbox._next_port
            AIMO3Sandbox._next_port += 5

        ports = list(range(start_port, start_port + 5))
        env = os.environ.copy()

        self._km = KernelManager()
        self._km.transport = 'tcp'
        self._km.ip = '127.0.0.1'
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]

        self._km.start_kernel(env=env, extra_arguments=['--Application.log_level=CRITICAL'])

        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True

        self.execute(
            'import math\n'
            'import numpy\n'
            'import sympy\n'
            'import itertools\n'
            'import collections\n'
            'import mpmath\n'
            'mpmath.mp.dps = 64\n'
        )

    def _format_error(self, traceback: list[str]) -> str:
        clean_lines = []
        for frame in traceback:
            clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)
            if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
                continue
            clean_lines.append(clean_frame)
        return ''.join(clean_lines)

    def execute(self, code: str, timeout: float | None = None) -> str:
        client = self._client
        effective_timeout = timeout or self._default_timeout
        
        msg_id = client.execute(
            code,
            store_history=True,
            allow_stdin=False,
            stop_on_error=False
        )

        stdout_parts = []
        stderr_parts = []
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed > effective_timeout:
                self._km.interrupt_kernel()
                return f'[ERROR] Execution timed out after {effective_timeout} seconds'

            try:
                msg = client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue

            if msg.get('parent_header', {}).get('msg_id') != msg_id:
                continue

            msg_type = msg.get('msg_type')
            content = msg.get('content', {})

            if msg_type == 'stream':
                text = content.get('text', '')
                if content.get('name') == 'stdout':
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == 'error':
                traceback_list = content.get('traceback', [])
                stderr_parts.append(self._format_error(traceback_list))
            elif msg_type in {'execute_result', 'display_data'}:
                data = content.get('data', {})
                text = data.get('text/plain')
                if text:
                    stdout_parts.append(text if text.endswith('\n') else f'{text}\n')
            elif msg_type == 'status':
                if content.get('execution_state') == 'idle':
                    break

        stdout = ''.join(stdout_parts)
        stderr = ''.join(stderr_parts)

        if stderr:
            return f'{stdout.rstrip()}\n{stderr}' if stdout else stderr

        return stdout if stdout.strip() else '[WARN] No output. Use print() to see results.'

    def close(self):
        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()

        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)
            with contextlib.suppress(Exception):
                self._km.cleanup_resources()

    def reset(self):
        self.execute(
            '%reset -f\n'
            'import math\n'
            'import numpy\n'
            'import sympy\n'
            'import itertools\n'
            'import collections\n'
            'import mpmath\n'
            'mpmath.mp.dps = 64\n'
        )

    def __del__(self):
        self.close()

# AIMO3Tool class (from notebook)
class AIMO3Tool:
    def __init__(self, local_jupyter_timeout: float, tool_prompt: str, sandbox=None):
        self._local_jupyter_timeout = local_jupyter_timeout
        self._tool_prompt = tool_prompt
        self._jupyter_session = sandbox
        self._owns_session = sandbox is None
        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def _ensure_session(self):
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = AIMO3Sandbox(timeout=self._local_jupyter_timeout)

    @property
    def instruction(self) -> str:
        return self._tool_prompt

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name='python',
            description=self.instruction,
            tools=[]
        )

    def _ensure_last_print(self, code: str) -> str:
        lines = code.strip().split('\n')
        if not lines:
            return code
        last_line = lines[-1].strip()
        if 'print' in last_line or 'import' in last_line:
            return code
        if not last_line or last_line.startswith('#'):
            return code
        lines[-1] = 'print(' + last_line + ')'
        return '\n'.join(lines)

    def _make_response(self, output: str, channel: str | None = None) -> Message:
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name='python')
        message = Message(author=author, content=[content]).with_recipient('assistant')
        if channel:
            message = message.with_channel(channel)
        return message

    def process_sync_plus(self, message: Message) -> list[Message]:
        self._ensure_session()
        raw_script = message.content[0].text
        final_script = self._ensure_last_print(raw_script)

        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final_script)
            except TimeoutError as exc:
                output = f'[ERROR] {exc}'

        return [self._make_response(output, channel=message.channel)]

def is_integer_answer(answer):
    """Check if answer can be parsed as an integer."""
    if answer is None:
        return False
    try:
        int(str(answer).strip())
        return True
    except (ValueError, AttributeError):
        return False

def extract_integer_from_text(text):
    """Extract integer from text (handles boxed format, etc.)"""
    if text is None:
        return None
    
    try:
        return int(str(text).strip())
    except (ValueError, AttributeError):
        pass
    
    boxed_pattern = r'\\boxed\s*\{\s*([0-9,]+)\s*\}'
    matches = re.findall(boxed_pattern, str(text))
    if matches:
        try:
            return int(matches[-1].replace(',', ''))
        except ValueError:
            pass
    
    numbers = re.findall(r'\b(\d+)\b', str(text))
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            pass
    
    return None

def _parse_expected_answer_to_int(expected_answer):
    if expected_answer is None:
        return None
    if isinstance(expected_answer, int):
        return expected_answer
    try:
        s = str(expected_answer).strip()
        s = s.replace(",", "")
        return int(s)
    except Exception:
        return None


def _msg_text(msg) -> str:
    if not hasattr(msg, "content") or not msg.content:
        return ""
    c0 = msg.content[0]
    return c0.text if hasattr(c0, "text") else str(c0)


def extract_trace(conversation, Role):
    """
    Returns:
      assistant_commentary: str
      assistant_final: str
      assistant_all: str
      tool_messages: list[dict]  (python outputs)
    """
    assistant_commentary_parts = []
    assistant_final_parts = []
    assistant_all_parts = []
    tool_messages = []

    for msg in conversation.messages:
        author = getattr(msg, "author", None)
        role = getattr(author, "role", None) if author else None
        channel = getattr(msg, "channel", None)
        recipient = getattr(msg, "recipient", None)

        text = _msg_text(msg).strip()
        if not text:
            continue

        if role == Role.ASSISTANT:
            assistant_all_parts.append(text)
            if channel == "final":
                assistant_final_parts.append(text)
            else:
                # Treat everything not-final as commentary/reasoning
                assistant_commentary_parts.append(text)

        elif role == Role.TOOL:
            # Tool outputs are messages authored by TOOL (name='python')
            tool_messages.append({
                "tool": getattr(author, "name", None) or "tool",
                "channel": channel,
                "recipient": recipient,
                "output": text
            })

    return {
        "assistant_commentary": "\n\n".join(assistant_commentary_parts).strip(),
        "assistant_final": "\n\n".join(assistant_final_parts).strip(),
        "assistant_all": "\n\n".join(assistant_all_parts).strip(),
        "tool_messages": tool_messages,
    }


def extract_quality_tags(model_final: str, tool_calls: list, tool_called: bool) -> list:
    tags = []
    boxed_only = False
    if model_final:
        # boxed-only if final is just a boxed number and no other content
        boxed_only = bool(re.fullmatch(r"\s*\\boxed\s*\{\s*\d+\s*\}\s*", model_final))
    if boxed_only:
        tags.append("boxed_only")
    if not tool_called:
        tags.append("no_tool_called")
    if any(tc.get("error") for tc in tool_calls):
        tags.append("tool_error")
    return tags


# Phrases that indicate meta or answer-reveal in hint summarizer output - parse only the actual hint
_HINT_RAMBLE_PHRASES = (
    "the user now",
    "the user asks",
    "the user asked",
    "output only 2-4 short sentences",
    "start directly with the first hint",
    "no preamble",
    "only output the hint",
    "we need to produce a hint",
    "the answer for the problem is",
    "the answer is indeed",
    "we are not providing the answer",
    "avoid revealing",
    "the user wants",
    "we have to generate",
    "we should ensure",
    "does not contain the numeric answer",
    "it's fine",
    "thus final",
    "correct answer",
    "final numerical",
    "do not include or reveal",
    "the guideline says",
    "comply with user",
    "our answer currently",
    "not the answer",
    "hint does not contain",
    "violate any rule",
)


def _is_hint_ramble(text: str) -> bool:
    """True if segment is meta/answer-reveal from hint summarizer; keep only actual hint content."""
    if not (text or "").strip():
        return True
    lower = text.lower().strip()
    if any(p in lower for p in _HINT_RAMBLE_PHRASES):
        return True
    if "\\boxed{" in text:
        return True
    return False


def _extract_hint_only(raw: str) -> str:
    """From summarizer LLM output, keep only the hint sentences; drop ramble at sentence level so blocks keep valuable hint content."""
    if not (raw or "").strip():
        return ""
    raw = raw.strip()
    # Remove any \boxed{...} that leaked
    raw = re.sub(r"\\boxed\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "", raw)
    parts = []
    for block in re.split(r"\n\n+", raw):
        block = block.strip()
        if not block:
            continue
        # Within block: drop only sentences that are ramble, keep the rest (block may have both ramble and hint)
        sentences = re.split(r"(?<=[.!?])\s+", block)
        kept = [s.strip() for s in sentences if s.strip() and not _is_hint_ramble(s.strip())]
        if kept:
            parts.append(" ".join(kept))
    return "\n\n".join(parts).strip()[:1500] if parts else ""


def _response_text_solution_only(trace: dict) -> str:
    """Build response_text: commentary + final only. Ramble is removed only from the hint summary, so response_text stays clean."""
    commentary = (trace.get("assistant_commentary") or "").strip()
    final = (trace.get("assistant_final") or "").strip()
    return "\n\n".join(filter(None, [commentary, final])) or ""


def _strip_final_boxed(text: str) -> str:
    """Remove the final \\boxed{...} from solution text so the answer is not revealed."""
    if not text or not isinstance(text, str):
        return text
    text = text.strip()
    # Remove last \boxed{...} at end (inner content: digits or simple chars; for nested use .*? from end)
    text = re.sub(r"\s*\\boxed\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*$", "", text)
    return text.strip()


# Trim last N chars for mathematical-principle block (drop trailing answer)
GUIDELINES_TRIM_TAIL = 20


def _solution_to_guidelines(solution_text: str, expected_answer: Any = None) -> str:
    """Deterministic: remove expected answer from solution, trim last 20 chars, prepend 'mathematical principle hidden in LLM'. Numbers kept."""
    s = (solution_text or "").strip()
    if not s:
        return ""
    if expected_answer is not None:
        s = _replace_expected_answer_with_placeholder(s, expected_answer)
    s = s[:-GUIDELINES_TRIM_TAIL].strip() if len(s) > GUIDELINES_TRIM_TAIL else s
    if not s:
        return ""
    return "mathematical principle hidden in LLM\n\n" + s


def _replace_expected_answer_with_placeholder(text: str, expected_answer: Any) -> str:
    """Replace every occurrence of the expected answer in text with the phrase 'expected_answer' (before passing to summarizer)."""
    if not text or expected_answer is None:
        return (text or "").strip()
    try:
        ans = str(expected_answer).strip()
    except Exception:
        return (text or "").strip()
    if not ans:
        return (text or "").strip()
    s = (text or "").strip()
    # \\boxed{42} -> \\boxed{expected_answer}
    s = re.sub(r"\\boxed\s*\{\s*" + re.escape(ans) + r"\s*\}", "\\\\boxed{expected_answer}", s, flags=re.IGNORECASE)
    # Standalone expected-answer value (word boundary) -> expected_answer
    s = re.sub(r"\b" + re.escape(ans) + r"\b", "expected_answer", s)
    return s


def _strip_expected_answer_from_hint(text: str, expected_answer: Any) -> str:
    """Final resort: use regex to remove expected_answer from hint (no final answer in hint)."""
    if not text or expected_answer is None:
        return (text or "").strip()
    s = (text or "").strip()
    try:
        ans = str(expected_answer).strip()
    except Exception:
        return s
    if not ans:
        return s
    # Remove \boxed{expected_answer} (with optional spaces inside braces)
    s = re.sub(r"\\boxed\s*\{\s*" + re.escape(ans) + r"\s*\}", "", s, flags=re.IGNORECASE)
    # Remove phrases like "the answer is 42", "answer: 42", "final answer is 42"
    s = re.sub(
        r"(?i)(the\s+)?(final\s+)?answer\s*(is|:)\s*" + re.escape(ans) + r"\.?\s*",
        "",
        s,
    )
    s = re.sub(r"\n\n+", "\n\n", s).strip()
    return s


def _sanitize_hint_text(text: str) -> str:
    """Remove code fences, JSON/echo blobs, and banned-fragment lines from hint text."""
    t = (text or "").strip()
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    # Leading "analysis..." / "The user asks..." blobs
    if re.match(r"^\s*(analysis|the user asks)\b", t, flags=re.IGNORECASE):
        t = re.sub(r"^\s*(analysis|the user asks).*?\n\s*\n", "", t, flags=re.IGNORECASE | re.DOTALL).strip()
        t = re.sub(r"(?is).*?\bOutput\s+ONLY\b.*?\n\s*\n", "", t).strip()
    banned_fragments = [
        "output only", "start directly", "no preamble", "no meta",
        "the user asks", "we need to", "we should", "thus final",
        "solution_approach_hint_used", "system_prompt", "user_input",
    ]
    lines = [ln.strip() for ln in t.splitlines()]
    kept = []
    for ln in lines:
        if not ln:
            continue
        low = ln.lower()
        if any(bf in low for bf in banned_fragments):
            continue
        if re.match(r'^\s*".+?"\s*:\s*".*"\s*,?\s*$', ln):
            continue
        kept.append(ln)
    t = "\n".join(kept).strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def _take_2_to_4_sentences(text: str) -> str:
    """Keep up to 4 sentences; truncate if model gave more."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return ""
    return " ".join(parts[:4]).strip()


def _looks_like_meta_or_echo(text: str) -> bool:
    """True if text still looks like prompt echo or meta, not a real hint."""
    if not (text or "").strip():
        return True
    low = text.lower()
    if any(x in low for x in ["the user asks", "output only", "system_prompt", "user_input", "analysis"]):
        return True
    if low.count("{") + low.count("}") >= 2:
        return True
    if re.match(r'^\s*["{]', text):
        return True
    return False


META_START_RE = re.compile(
    r"^\s*(analysis\b|the user asks\b|\"?solution_approach_hint_used\"?\s*:)",
    re.IGNORECASE,
)


def strip_leading_meta_block(text: str) -> str:
    """Drop leading meta/echo block (analysis, the user asks, solution_approach_hint_used, etc.)."""
    t = (text or "").lstrip()
    if not META_START_RE.search(t):
        return t
    t2 = re.sub(r"(?is)^\s*(analysis\b.*?)(\n\s*\n+)", "", t, count=1).lstrip()
    if META_START_RE.search(t2) or "output only" in t2.lower():
        t2 = re.sub(r"(?is)^.*?\n\s*\n+", "", t2, count=1).lstrip()
    if "output only" in (t or "").lower() and "\n\n" in (t or ""):
        t2 = t.split("\n\n")[-1].strip()
    return t2.strip()


_HINT_BANNED = (
    "analysis",
    "the user asks",
    "output only",
    "no preamble",
    "no meta",
    "we need to",
    "we should",
    "thus final",
    "system_prompt",
    "user_input",
    "solution_approach_hint_used",
)


def keep_non_meta_sentences(text: str, max_sentences: int = 4) -> str:
    """When model jams meta inline with no blank line: keep only sentences that don't contain BANNED fragments."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    sents = re.split(r"(?<=[.!?])\s+", t)
    clean = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        low = s.lower()
        if any(b in low for b in _HINT_BANNED):
            continue
        clean.append(s)
        if len(clean) >= max_sentences:
            break
    return " ".join(clean).strip()


_SUMMARY_BANNED = [
    "the user asks", "output only", "no preamble", "no meta", "do not repeat",
    "system_prompt", "user_input", "solution_approach_hint_used",
    "use this hint", "hint", "solution approach",
]


def _strip_instruction_echo(text: str) -> str:
    """Strip code fences, 'Use this hint' / 'HINT (solution approach):' headers, and banned-fragment lines."""
    t = (text or "").strip()
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    if re.match(r"^\s*(analysis\b|the user asks\b)", t, flags=re.IGNORECASE):
        t = re.sub(r"(?is)^\s*(analysis\b.*?)(\n\s*\n+)", "", t, count=1).strip()
    t = re.sub(r"(?im)^\s*use this hint.*$", "", t).strip()
    t = re.sub(r"(?im)^\s*hint\s*\(.*?\)\s*:\s*$", "", t).strip()
    t = re.sub(r"(?im)^\s*hint\s*:\s*$", "", t).strip()
    t = re.sub(r"(?im)^\s*solution approach\s*:\s*$", "", t).strip()
    lines = []
    for ln in t.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        low = ln.lower()
        if any(b in low for b in _SUMMARY_BANNED):
            continue
        if re.match(r'^\s*".+?"\s*:\s*', ln):
            continue
        lines.append(ln)
    t = " ".join(lines).strip()
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\\boxed\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "", t).strip()
    return t


def _looks_bad_summary(text: str) -> bool:
    """True if summary still looks like instruction echo or useless."""
    if not (text or "").strip():
        return True
    low = text.lower()
    if any(b in low for b in _SUMMARY_BANNED):
        return True
    if low.count("{") + low.count("}") >= 2:
        return True
    if len(text.strip()) < 20:
        return True
    return False


def summarize_solution_approach(
    solution_text_no_answer: str, problem: str, expected_answer: Any = None
) -> str:
    """
    Ask LLM to summarize the following text. Solution is already stripped of expected_answer by caller.
    Return the raw output as-is, no filtering.
    """
    if not (solution_text_no_answer or "").strip():
        return ""

    template = AIMO3Template()
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()
    tool_config = ToolNamespaceConfig(name="none", description="No tools.", tools=[])

    user_input = "Summarize the following text.\n\n" + solution_text_no_answer[:8000]

    messages = template.apply_chat_template(
        system_prompt="Summarize the given text.",
        user_input=user_input,
        tool_config=tool_config,
    )
    conversation = Conversation.from_messages(messages)
    client = OpenAI(base_url=BASE_URL, api_key="sk-local", timeout=CFG.session_timeout)

    try:
        prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        max_tokens = min(500, CFG.context_tokens - len(prompt_ids) - CFG.buffer_tokens)
        if max_tokens < 50:
            return ""
        completion = client.completions.create(
            model=MODEL_NAME,
            prompt=prompt_ids,
            max_tokens=max_tokens,
            temperature=0.2,
            seed=CFG.seed,
            stream=False,
            extra_body={"min_p": CFG.min_p, "stop_token_ids": stop_token_ids, "return_token_ids": True},
        )
        choice = completion.choices[0] if completion.choices else None
        if not choice:
            return ""
        raw = ""
        if getattr(choice, "text", None):
            raw = (choice.text or "").strip()
        elif getattr(getattr(choice, "message", None), "content", None):
            raw = (choice.message.content or "").strip()
        if not raw and getattr(choice, "token_ids", None):
            new_messages = encoding.parse_messages_from_completion_tokens(choice.token_ids, Role.ASSISTANT)
            if new_messages:
                last = new_messages[-1]
                if hasattr(last, "content") and last.content:
                    raw = (last.content[0].text if hasattr(last.content[0], "text") else str(last.content[0])).strip()
        return raw.strip() if raw else ""
    except Exception as e:
        print(f"[WARNING] summarize_solution_approach failed: {e}")
    return ""


def score_example_with_tools(example, extra_instruction: str = ""):
    """Score a single example using local GPT-OSS model with tool calling."""
    try:
        # Extract problem text
        question = example.get("question", example.get("problem", ""))
        if not question:
            return None
        
        # Check if API server is reachable
        try:
            import httpx
            test_client = httpx.Client(timeout=5.0)
            response = test_client.get(f"{BASE_URL.replace('/v1', '')}/health", timeout=5.0)
            test_client.close()
        except Exception as e:
            print(f"[WARNING] API server health check failed: {e}. Continuing anyway...")
        
        # Initialize sandbox
        sandbox = AIMO3Sandbox(timeout=CFG.jupyter_timeout)
        
        # Create AIMO3Tool instance
        local_tool = AIMO3Tool(
            local_jupyter_timeout=CFG.jupyter_timeout,
            tool_prompt=CFG.tool_prompt,
            sandbox=sandbox
        )
        
        # Setup template and encoding
        template = AIMO3Template()
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        stop_token_ids = encoding.stop_tokens_for_assistant_actions()
        
        # Build messages using template with tool_config
        extra = (extra_instruction or "").strip()
        user_text = f"{question} {CFG.preference_prompt}"
        if extra:
            user_text += "\n\n" + extra
        messages = template.apply_chat_template(
            CFG.system_prompt,
            user_text,
            local_tool.tool_config
        )
        
        conversation = Conversation.from_messages(messages)
        
        # Initialize OpenAI client
        client = OpenAI(
            base_url=BASE_URL,
            api_key="sk-local",
            timeout=CFG.session_timeout
        )
        
        answer = None
        response_text = None
        tool_called = False  # Track if tool was called
        tool_calls = []
        max_turns = CFG.turns
        
        # Multi-turn generation loop with tool calling
        for turn in range(max_turns):
            # Render conversation for completion
            prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
            max_tokens = CFG.context_tokens - len(prompt_ids)
            
            if max_tokens < CFG.buffer_tokens:
                break
            
            try:
                # Call API with explicit timeout wrapper
                print(f"    [Turn {turn + 1}/{max_turns}] Calling API (timeout={CFG.session_timeout}s)...")
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"API call timed out after {CFG.session_timeout}s")
                
                # Set up timeout (Unix only)
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(CFG.session_timeout))
                
                try:
                    completion = client.completions.create(
                        model=MODEL_NAME,
                        prompt=prompt_ids,
                        max_tokens=max_tokens,
                        temperature=CFG.temperature,
                        logprobs=CFG.top_logprobs,
                        seed=CFG.seed,
                        stream=False,
                        timeout=CFG.session_timeout,
                        extra_body={
                            "min_p": CFG.min_p,
                            "stop_token_ids": stop_token_ids,
                            "return_token_ids": True
                        }
                    )
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                
                print(f"    [Turn {turn + 1}/{max_turns}] API call completed")
                
                # Extract token_ids from response
                completion_tokens = completion.choices[0].token_ids if hasattr(completion.choices[0], 'token_ids') else []
                if not completion_tokens:
                    break
                
                # Parse messages from completion (detects tool calls)
                new_messages = encoding.parse_messages_from_completion_tokens(completion_tokens, Role.ASSISTANT)
                conversation.messages.extend(new_messages)
                last_message = new_messages[-1]
                
                # Check if we got a final answer
                if last_message.channel == 'final':
                    answer_text = last_message.content[0].text
                    response_text = answer_text
                    pattern = r'\\boxed\s*\{\s*([0-9,]+)\s*\}'
                    matches = re.findall(pattern, answer_text)
                    if matches:
                        try:
                            answer = int(matches[-1].replace(',', ''))
                            if 0 <= answer <= 99999:
                                break
                        except ValueError:
                            pass
                
                # Check if it's a tool call (recipient == "python")
                message_recipient = getattr(last_message, 'recipient', None)
                if message_recipient == 'python':
                    tool_called = True

                    tool_code = _msg_text(last_message)
                    tool_event = {
                        "code": tool_code,
                        "output": None,
                        "error": False,
                        "turn": turn + 1,
                        "channel": getattr(last_message, "channel", None),
                    }

                    tool_responses = local_tool.process_sync_plus(last_message)
                    conversation.messages.extend(tool_responses)

                    # Capture output text from the tool response(s)
                    out_text = ""
                    if tool_responses:
                        out_text = _msg_text(tool_responses[-1])

                    tool_event["output"] = out_text
                    # Basic error detection
                    if out_text.startswith("[ERROR]") or "Traceback" in out_text or out_text.strip().startswith("Error"):
                        tool_event["error"] = True

                    tool_calls.append(tool_event)
                    continue
                
                # If no tool call and no final answer, continue
                if last_message.channel != 'final':
                    continue
                    
            except TimeoutError as e:
                print(f"    [ERROR] Turn {turn + 1} timed out: {e}")
                break
            except Exception as e:
                print(f"    [ERROR] Error in turn {turn + 1}: {e}")
                break
            
            # Small delay between turns
            if turn < max_turns - 1:
                time.sleep(0.05)
        
        # Cleanup sandbox
        sandbox.reset()
        sandbox.close()
        
        trace = extract_trace(conversation, Role)
        model_final = trace["assistant_final"]
        model_commentary = trace["assistant_commentary"]
        # response_text: only solution-related output, exclude instruction echo / ramble
        response_text = _response_text_solution_only(trace) or model_final or ""

        # Fallback: try to extract answer from any message if not yet set
        if answer is None:
            for msg in reversed(conversation.messages):
                text = _msg_text(msg)
                pattern = r'\\boxed\s*\{\s*([0-9,]+)\s*\}'
                matches = re.findall(pattern, text)
                if matches:
                    try:
                        answer = int(matches[-1].replace(',', ''))
                        if 0 <= answer <= 99999:
                            break
                    except ValueError:
                        pass

        quality_tags = extract_quality_tags(model_final=model_final, tool_calls=tool_calls, tool_called=tool_called)

        # Build a training-ready chat transcript (messages list)
        messages_out = []
        for msg in conversation.messages:
            author = getattr(msg, "author", None)
            role = getattr(author, "role", None) if author else None
            text = _msg_text(msg).strip()
            if not text:
                continue

            if role == Role.SYSTEM:
                messages_out.append({"role": "system", "content": text})
            elif role == Role.USER:
                messages_out.append({"role": "user", "content": text})
            elif role == Role.ASSISTANT:
                messages_out.append({
                    "role": "assistant",
                    "content": text,
                    "channel": getattr(msg, "channel", None)
                })
            elif role == Role.TOOL:
                messages_out.append({
                    "role": "tool",
                    "name": getattr(author, "name", "python"),
                    "content": text
                })

        return {
            "answer": answer if answer is not None else None,
            "response_text": response_text,
            "model_final": model_final,
            "model_commentary": model_commentary,
            "tool_called": tool_called,
            "tool_calls": tool_calls,
            "quality_tags": quality_tags,
            "messages": messages_out,
        }

    except Exception as e:
        print(f"Error scoring example: {e}")
        return {
            "answer": None,
            "response_text": f"Error: {str(e)}",
            "tool_called": False,
            "model_final": "",
            "model_commentary": "",
            "tool_calls": [],
            "quality_tags": [],
            "messages": [],
        }

def main():
    if OpenAI is None or load_harmony_encoding is None or KernelManager is None:
        raise RuntimeError(
            "Missing optional dependencies for tool-calling execution. "
            "Install: `pip install openai openai-harmony jupyter_client`."
        )

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_JSONL}")

    # Load input: combined_math_crystal_hard50.jsonl (id, source, answer, problem, solution, datatype)
    examples = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(examples)} examples from {INPUT_JSONL}")

    # Load parsed-tracking: ids we have already scored (skip on re-run)
    parsed_ids = set()
    if PARSED_TRACKING_JSONL.exists():
        with open(PARSED_TRACKING_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    pid = obj.get("id")
                    if pid is not None:
                        parsed_ids.add(pid)
                except json.JSONDecodeError:
                    continue
        print(f"Resume: {len(parsed_ids)} already parsed (tracked in {PARSED_TRACKING_JSONL})")

    to_process = [ex for ex in examples if ex.get("id") not in parsed_ids]
    print(f"To process: {len(to_process)} (skipping {len(examples) - len(to_process)} already done)")

    if not to_process:
        print("Nothing to do.")
        return

    # Optional cap
    parser = argparse.ArgumentParser(description="Score combined math dataset with tool calling")
    parser.add_argument("--max-examples", type=int, default=None, help="Cap number of examples to score")
    args = parser.parse_args()
    if args.max_examples is not None:
        to_process = to_process[: args.max_examples]
        print(f"Capped to {len(to_process)} examples")

    print("=" * 80)
    print("Scoring combined math dataset (tool calling)")
    print("=" * 80)
    print(f"Output: {OUTPUT_JSONL}")
    print(f"Tracking: {PARSED_TRACKING_JSONL}")

    processed = 0
    matches = 0
    for idx, row in enumerate(tqdm(to_process, desc="Scoring")):
        ex_id = row.get("id")
        problem = row.get("problem", row.get("question", ""))
        if not problem:
            continue
        raw_answer = row.get("answer")
        try:
            expected_answer = int(str(raw_answer).strip().replace(",", "")) if raw_answer is not None else None
        except (ValueError, TypeError):
            expected_answer = None
        example = {"problem": problem, "expected_answer_int": expected_answer}

        # Attempt 2+: deterministic block = solution with expected answer removed, last 20 chars trimmed, prefixed by "mathematical principle hidden in LLM" (numbers kept)
        solution_approach_hint = None
        raw_solution = row.get("solution")
        if raw_solution and isinstance(raw_solution, str) and raw_solution.strip():
            solution_no_answer = _strip_final_boxed(raw_solution)
            if solution_no_answer:
                solution_approach_hint = _solution_to_guidelines(solution_no_answer, expected_answer)
                if not (solution_approach_hint or "").strip():
                    solution_approach_hint = None

        problem_matched = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            commentary_used = None
            solution_approach_hint_used = None
            extra_instruction = ""
            # Attempt 2+: silently augment with solution approach summary (no "hint" / "Use this hint" wrapper)
            if attempt >= 2 and solution_approach_hint:
                extra_instruction = "\n\n" + solution_approach_hint.strip()
                solution_approach_hint_used = solution_approach_hint.strip()
                commentary_used = extra_instruction

            try:
                result = score_example_with_tools(example, extra_instruction=extra_instruction)
            except Exception as e:
                print(f"\n[ERROR] id={ex_id} attempt={attempt}: {e}")
                result = None

            if isinstance(result, dict):
                predicted_answer = result.get("answer")
                response_text = result.get("response_text", "")
                tool_called = result.get("tool_called", False)
                model_final = result.get("model_final", "")
                model_commentary = result.get("model_commentary", "")
                tool_calls = result.get("tool_calls", [])
                quality_tags = result.get("quality_tags", [])
                messages = result.get("messages", [])
            else:
                predicted_answer = None
                response_text = "" if result is None else str(result)
                tool_called = False
                model_final = model_commentary = ""
                tool_calls = []
                quality_tags = []
                messages = []

            is_match = False
            if predicted_answer is not None and expected_answer is not None:
                is_match = str(predicted_answer) == str(expected_answer)
            score_match = "match" if is_match else "no match"
            if is_match:
                problem_matched = True

            out_entry = {
                "id": ex_id,
                "source": row.get("source"),
                "answer": raw_answer,
                "problem": problem,
                "solution": row.get("solution"),
                "datatype": row.get("datatype"),
                "attempt": attempt,
                "commentary_used": commentary_used,
                "solution_approach_hint_used": solution_approach_hint_used,
                "predicted_answer": predicted_answer,
                "response_text": response_text,
                "score_match": score_match,
                "tool_called": tool_called,
                "model_final": model_final,
                "model_commentary": model_commentary,
                "tool_calls": tool_calls,
                "quality_tags": quality_tags,
                "messages": messages,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(OUTPUT_JSONL, "a", encoding="utf-8") as f_out:
                f_out.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
                f_out.flush()

            if is_match:
                break

        if problem_matched:
            matches += 1
        processed += 1
        with open(PARSED_TRACKING_JSONL, "a", encoding="utf-8") as f_track:
            f_track.write(json.dumps({"id": ex_id}, ensure_ascii=False) + "\n")
            f_track.flush()

    pct = (100 * matches / processed) if processed else 0
    print(f"\nDone. Processed {processed}, matches {matches} ({pct:.1f}%)")
    print(f"Output: {OUTPUT_JSONL}")
    print(f"Tracking: {PARSED_TRACKING_JSONL}")


if __name__ == "__main__":
    main()
