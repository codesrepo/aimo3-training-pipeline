#!/usr/bin/env python3
"""
Dataset preparation script for OpenMathReasoning.
Iteratively selects examples where model score matches expected answer.
Uses tool calling with Harmony encoding (like imo-v15012026.ipynb).
"""

import json
import time
import re
import os
import sys
import argparse
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

# Configuration
BATCH_SIZE = 100000  # Top 100k per iteration
GPU_BATCH_SIZE = 1  # Process 8 examples concurrently on GPU
LOCAL_DATA_DIR = Path("/home/malam/wsl-tunix/imo/openmath_data")
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_LOG_PATH = LOCAL_DATA_DIR / "predictions_log.jsonl"
ORACLE_TRACES_OUT_PATH = LOCAL_DATA_DIR / "oracle_traces_no_match.jsonl"
ALL_EXAMPLES_PATH = LOCAL_DATA_DIR / "all_examples_with_clusters.jsonl"
MASTER_EXAMPLES_PATH = LOCAL_DATA_DIR / "master_examples_not_in_predictions.jsonl"
PARSED_TRACKING_PATH = LOCAL_DATA_DIR / "oracle_traces_parsed.jsonl"


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

def extract_quality_tags(model_final: str, tool_calls: list[dict], tool_called: bool) -> list[str]:
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

def load_no_match_problems_from_predictions_log(predictions_log_path: Path) -> list[dict]:
    """Read predictions_log.jsonl and keep only score_match == 'no match'.

    Output rows have only:
      - problem
      - expected_answer (int when parseable; else original value)
    """
    rows: list[dict] = []
    no_match = 0
    total = 0
    if not predictions_log_path.exists():
        raise FileNotFoundError(f"Missing predictions log at: {predictions_log_path}")

    with open(predictions_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("score_match") != "no match":
                continue
            no_match += 1
            problem = obj.get("problem", "")
            expected_raw = obj.get("expected_answer")
            expected_int = _parse_expected_answer_to_int(expected_raw)
            rows.append(
                {
                    "problem": problem,
                    "expected_answer": expected_int if expected_int is not None else expected_raw,
                }
            )

    # User requested sanity check: should be 40+
    print(f"[Input] predictions_log total={total} no_match={no_match}")
    if no_match < 40:
        raise RuntimeError(
            f"Expected >= 40 'no match' records, but found {no_match}. "
            f"Check {predictions_log_path}."
        )
    return rows


def _collect_problem_set_from_predictions_log(predictions_log_path: Path) -> set[str]:
    """Collect all 'problem' values from predictions_log.jsonl (for exclusion when building master)."""
    seen: set[str] = set()
    if not predictions_log_path.exists():
        return seen
    with open(predictions_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = obj.get("problem")
            if isinstance(p, str) and p:
                seen.add(p)
    return seen


def build_master_examples_file(
    all_examples_path: Path,
    predictions_log_path: Path,
    master_path: Path,
) -> int:
    """Build master JSONL from all_examples_with_clusters: filter expected_answer in [0,99999],
    exclude examples whose problem is in predictions_log. Return count written."""
    problems_in_log = _collect_problem_set_from_predictions_log(predictions_log_path)
    print(f"[Build master] Excluding {len(problems_in_log)} problems present in predictions_log")

    master_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    total = 0
    filtered_int = 0
    excluded_in_log = 0

    with open(master_path, "w", encoding="utf-8") as out:
        with open(all_examples_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Building master from all_examples"):
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                problem = obj.get("problem")
                if not isinstance(problem, str) or not problem:
                    continue
                raw = obj.get("expected_answer")
                v = _parse_expected_answer_to_int(raw)
                if v is None or v < 0 or v > 99999:
                    filtered_int += 1
                    continue
                if problem in problems_in_log:
                    excluded_in_log += 1
                    continue
                record = {"problem": problem, "expected_answer": v}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    print(f"[Build master] total={total} | kept (int 0–99999, not in log)={written} | filtered_int={filtered_int} | excluded_in_log={excluded_in_log}")
    return written


def load_problems_from_master(master_path: Path) -> list[dict]:
    """Load problem/expected_answer rows from master JSONL."""
    rows: list[dict] = []
    if not master_path.exists():
        return rows
    with open(master_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            problem = obj.get("problem")
            if not isinstance(problem, str) or not problem:
                continue
            rows.append({"problem": problem, "expected_answer": obj.get("expected_answer")})
    return rows


def load_parsed_problems(parsed_path: Path) -> set[str]:
    """Load set of problem strings already processed (from parsed-tracking JSONL)."""
    seen: set[str] = set()
    if not parsed_path.exists():
        return seen
    with open(parsed_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = obj.get("problem")
            if isinstance(p, str) and p:
                seen.add(p)
    return seen


def generate_oracle_mismatch_trace(
    problem: str,
    expected_answer,
    failed_attempt_texts: list[str],
    successful_attempt_text: str,
) -> str:
    """Ask an oracle LLM to compare failed vs successful attempt precisely.

    Output format requirement (user):
      since the problem... approach-1 failed because... approach-2 works here because...
    """
    template = AIMO3Template()
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    tool_config = ToolNamespaceConfig(
        name="none",
        description="No tools available.",
        tools=[],
    )

    failures = "\n\n".join(
        f"ATTEMPT {i+1} (model output):\n{(t or '').strip()}" for i, t in enumerate(failed_attempt_texts)
    )
    user_input = (
        "You are an IMO math oracle and error analyst.\n"
        "Task: Compare a failed attempt vs a later successful attempt and pinpoint the *exact* wrong step:\n"
        "- Identify the incorrect assumption / formula / function use / missing case.\n"
        "- Provide two approaches:\n"
        "  - approach-1: summarize what was tried and why it failed (be very precise)\n"
        "  - approach-2: summarize what worked and why it works here (key lemma/invariant/case split)\n"
        "- Conclude with the correct final integer in \\boxed{}.\n\n"
        "You MUST write the final output in exactly this template:\n"
        "since the problem ...\n"
        "approach-1 failed because ...\n"
        "approach-2 works here because ...\n"
        "final: \\boxed{...}\n\n"
        f"PROBLEM:\n{problem}\n\n"
        f"EXPECTED_ANSWER:\n{expected_answer}\n\n"
        f"FAILED_ATTEMPTS:\n{failures}\n"
        f"\nSUCCESSFUL_ATTEMPT:\n{(successful_attempt_text or '').strip()}\n"
    )

    messages = template.apply_chat_template(
        system_prompt="You are a precise mathematical oracle and error analyst.",
        user_input=user_input,
        tool_config=tool_config,
    )
    conversation = Conversation.from_messages(messages)

    client = OpenAI(
        base_url=BASE_URL,
        api_key="sk-local",
        timeout=httpx.Timeout(connect=30.0, read=1800.0, write=1800.0, pool=30.0),
    )

    response_text = ""
    max_turns = 16
    for _ in range(max_turns):
        prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        max_tokens = CFG.context_tokens - len(prompt_ids)
        if max_tokens < CFG.buffer_tokens:
            break

        completion = client.completions.create(
            model=MODEL_NAME,
            prompt=prompt_ids,
            max_tokens=max_tokens,
            temperature=0.2,
            logprobs=CFG.top_logprobs,
            seed=CFG.seed,
            stream=False,
            extra_body={
                "min_p": CFG.min_p,
                "stop_token_ids": stop_token_ids,
                "return_token_ids": True,
            },
        )

        completion_tokens = completion.choices[0].token_ids if hasattr(completion.choices[0], "token_ids") else []
        if not completion_tokens:
            break
        new_messages = encoding.parse_messages_from_completion_tokens(completion_tokens, Role.ASSISTANT)
        conversation.messages.extend(new_messages)
        last_message = new_messages[-1]
        if last_message.channel == "final":
            response_text = last_message.content[0].text
            break

    # Fallback to any last assistant message
    if not response_text:
        for msg in reversed(conversation.messages):
            if getattr(msg, "author", None) and getattr(msg.author, "role", None) == Role.ASSISTANT:
                if hasattr(msg, "content") and msg.content:
                    response_text = msg.content[0].text if hasattr(msg.content[0], "text") else str(msg.content[0])
                    break
    return response_text.strip()

def generate_reflection_avoidance(problem: str, expected_answer, predicted_answer, failed_attempt_text: str, avoid_so_far: str) -> str:
    """Independent reflection LLM: summarize what didn't work and revise the approach, suggesting like, to use small numbers to find the pattern then easily generalize to predict correctly.

    Returns a short string that can be appended into the next attempt prompt with revised guidance.
    """
    template = AIMO3Template()
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    tool_config = ToolNamespaceConfig(
        name="none",
        description="No tools available.",
        tools=[],
    )

    user_input = (
        "You are a reflection coach for competition-math solving.\n"
        "Given a problem, a failed attempt with its predicted answer, and the expected answer, "
        "produce a revised approach memo for the next attempt.\n"
        "suggesting like, to use small numbers to find the pattern then easily generalize to predict correctly\n"
        "Requirements:\n"
        "- Identify what went wrong: name the mistaken assumption/formula/step pattern.\n"
        "- Revise the approach: suggest how to correct the method to reach the expected answer.\n"
        "- Be specific and actionable: provide concrete guidance (e.g., 'use X theorem instead of Y', 'check case Z', 'apply invariant W').\n"
        "- Provide 3-6 bullet points.\n"
        "- Do NOT solve the problem fully. For Norwegian you can suggest: Assume 1 is a divisor, Factor Constraints, Divide by \(n\) rearrange, Bound the Variables, Test Cases: Try \(a=2\) and \(a=3\) to find the smallest integer \(n\) that satisfies the condition \(d_1+d_2+d_3=n\)\n"
        "- Keep it under 1500 characters.\n\n"
        f"PROBLEM:\n{problem}\n\n"
        f"LAST_ATTEMPT_PREDICTED_ANSWER:\n{predicted_answer}\n\n"
        f"EXPECTED_ANSWER:\n{expected_answer}\n\n"
        f"FAILED_ATTEMPT_OUTPUT:\n{(failed_attempt_text or '').strip()}\n\n"
    )
    if (avoid_so_far or "").strip():
        user_input += f"PREVIOUS_AVOIDANCE_NOTES (do not repeat these mistakes):\n{avoid_so_far.strip()}\n\n"
    user_input += (
        "Return a revised approach memo that:\n"
        "1. Summarizes what went wrong in the last attempt\n"
        "2. Provides specific guidance on how to revise the approach to predict the correct answer\n"
        "3. Suggests alternative strategies, theorems, or case splits that might work\n"
    )

    messages = template.apply_chat_template(
        system_prompt="You are a precise reflection coach.",
        user_input=user_input,
        tool_config=tool_config,
    )
    conversation = Conversation.from_messages(messages)

    client = OpenAI(
        base_url=BASE_URL,
        api_key="sk-local",
        timeout=CFG.session_timeout,
    )

    response_text = ""
    max_turns = 8
    for _ in range(max_turns):
        prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        max_tokens = CFG.context_tokens - len(prompt_ids)
        if max_tokens < CFG.buffer_tokens:
            break

        completion = client.completions.create(
            model=MODEL_NAME,
            prompt=prompt_ids,
            max_tokens=max_tokens,
            temperature=0.1,
            logprobs=CFG.top_logprobs,
            seed=CFG.seed,
            stream=False,
            extra_body={
                "min_p": CFG.min_p,
                "stop_token_ids": stop_token_ids,
                "return_token_ids": True,
            },
        )

        completion_tokens = completion.choices[0].token_ids if hasattr(completion.choices[0], "token_ids") else []
        if not completion_tokens:
            break
        new_messages = encoding.parse_messages_from_completion_tokens(completion_tokens, Role.ASSISTANT)
        conversation.messages.extend(new_messages)
        last_message = new_messages[-1]
        if last_message.channel == "final":
            response_text = last_message.content[0].text
            break

    if not response_text:
        for msg in reversed(conversation.messages):
            if getattr(msg, "author", None) and getattr(msg.author, "role", None) == Role.ASSISTANT:
                if hasattr(msg, "content") and msg.content:
                    response_text = msg.content[0].text if hasattr(msg.content[0], "text") else str(msg.content[0])
                    break

    return (response_text or "").strip()[:1500]

def run_oracle_traces_from_no_match_predictions(
    predictions_log_path: Path,
    out_path: Path,
    max_examples: int | None = None,
):
    problems = load_no_match_problems_from_predictions_log(predictions_log_path)
    if max_examples is not None:
        problems = problems[:max_examples]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Writing oracle traces to: {out_path}")

    # Resume support: if output exists, skip already-written problems
    already_done: set[str] = set()
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    p = obj.get("problem")
                    if isinstance(p, str) and p:
                        already_done.add(p)
            if already_done:
                print(f"[Resume] Found {len(already_done)} existing oracle records; will skip them.")
        except Exception as e:
            print(f"[Resume] Warning: could not read existing output for resume: {e}")

    solved_first_try = 0
    solved_second_try = 0
    solved_third_try = 0
    oracle_traces_written = 0
    skipped_all_failed = 0

    # Append so we save after each record and can resume safely.
    with open(out_path, "a", encoding="utf-8") as f_out:
        for i, row in enumerate(tqdm(problems, desc="Generating oracle traces")):
            if i<6:
                continue
            problem = row["problem"]
            expected_answer = row["expected_answer"]
            expected_int = _parse_expected_answer_to_int(expected_answer)

            if problem in already_done:
                continue

            # Build a minimal example compatible with score_example_with_tools
            example = {"problem": problem, "expected_answer_int": expected_int}

            attempt_texts: list[str] = []
            attempt_answers: list[int | None] = []
            success_at = None
            had_failure_before_success = False
            failed_texts_before_success: list[str] = []
            successful_text: str | None = None
            avoid_memo = ""
            per_attempt_reflection: list[str] = []

            for attempt_idx in range(3):
                extra = ""
                if avoid_memo.strip():
                    extra = (
                        "Previous attempt(s) failed. Use the revised approach guidance below to predict correctly.\n"
                        f"REVISED_APPROACH_GUIDANCE:\n{avoid_memo.strip()}\n"
                    )
                res = score_example_with_tools(example, extra_instruction=extra)
                predicted = res.get("answer") if isinstance(res, dict) else None
                response_text = res.get("response_text", "") if isinstance(res, dict) else ""
                attempt_texts.append(response_text)
                attempt_answers.append(predicted)

                is_match = False
                if predicted is not None and expected_int is not None:
                    is_match = str(predicted) == str(expected_int)

                if is_match:
                    success_at = attempt_idx + 1
                    successful_text = response_text
                    if success_at == 1:
                        solved_first_try += 1
                    elif success_at == 2:
                        solved_second_try += 1
                    else:
                        solved_third_try += 1
                    print(
                        f"\n[OK] idx={i} solved in {success_at} attempt(s): predicted={predicted} expected={expected_int}"
                    )
                    break
                else:
                    had_failure_before_success = True
                    failed_texts_before_success.append(response_text)
                    print(
                        f"\n[WRONG] idx={i} attempt={attempt_idx+1} predicted={predicted} expected={expected_int}"
                    )
                    # Reflection after a failure: build a revised approach memo for the next attempt
                    if attempt_idx < 2:
                        memo = generate_reflection_avoidance(
                            problem=problem,
                            expected_answer=expected_answer,
                            predicted_answer=predicted,
                            failed_attempt_text=response_text,
                            avoid_so_far=avoid_memo,
                        )
                        per_attempt_reflection.append(memo)
                        if memo:
                            avoid_memo = (avoid_memo + "\n\n" + memo).strip() if avoid_memo else memo.strip()

            # User correction:
            # - Oracle call must happen only after we observed a fail AND then a pass.
            # - If all 3 fail, do NOT call oracle and skip the record.
            if success_at is None:
                skipped_all_failed += 1
                continue

            if success_at == 1:
                # Got it right immediately; nothing to analyze/train on for "fail->pass" traces.
                continue

            if not had_failure_before_success or not failed_texts_before_success or successful_text is None:
                # Shouldn't happen for success_at>1, but keep safe.
                continue

            oracle_trace = generate_oracle_mismatch_trace(
                problem=problem,
                expected_answer=expected_answer,
                failed_attempt_texts=failed_texts_before_success,
                successful_attempt_text=successful_text,
            )

            record = {
                "problem": problem,
                "expected_answer": expected_answer,
                "attempts": [
                    {
                        "attempt": j + 1,
                        "predicted_answer": attempt_answers[j],
                        "response_text": attempt_texts[j],
                        "reflection_avoid_memo": per_attempt_reflection[j] if j < len(per_attempt_reflection) else None,
                    }
                    for j in range(len(attempt_texts))
                ],
                "avoid_memo_final": avoid_memo,
                "solved_attempt": success_at,
                "oracle_trace": oracle_trace,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            try:
                os.fsync(f_out.fileno())
            except Exception:
                pass
            oracle_traces_written += 1

    print("\n" + "=" * 80)
    print("Oracle traces run complete")
    print("=" * 80)
    print(f"Total problems: {len(problems)}")
    print(f"Solved in 1 attempt: {solved_first_try}")
    print(f"Solved in 2 attempts: {solved_second_try}")
    print(f"Solved in 3 attempts: {solved_third_try}")
    print(f"Oracle traces written (fail->pass only): {oracle_traces_written}")
    print(f"Skipped (all 3 failed): {skipped_all_failed}")


def run_oracle_traces_from_master(
    master_path: Path,
    out_path: Path,
    parsed_tracking_path: Path,
    max_examples: int | None = None,
):
    """Load examples from master JSONL one-by-one; skip those in parsed-tracking.
    Qualifying traces (fail->pass) go to out_path; every processed example is appended to parsed-tracking."""
    problems = load_problems_from_master(master_path)
    if not problems:
        raise FileNotFoundError(
            f"Master file empty or missing: {master_path}. "
            "Run with --build-master first."
        )
    if max_examples is not None:
        problems = problems[:max_examples]

    already_parsed = load_parsed_problems(parsed_tracking_path)
    if already_parsed:
        print(f"[Resume] Found {len(already_parsed)} already-parsed problems; will skip them.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_tracking_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Oracle traces -> {out_path}")
    print(f"[Output] Parsed tracking -> {parsed_tracking_path}")

    already_done: set[str] = set()
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    p = obj.get("problem")
                    if isinstance(p, str) and p:
                        already_done.add(p)
            if already_done:
                print(f"[Resume] Found {len(already_done)} existing oracle records; will skip writing duplicates.")
        except Exception as e:
            print(f"[Resume] Warning: could not read existing oracle output: {e}")

    solved_first_try = 0
    solved_second_try = 0
    solved_third_try = 0
    oracle_traces_written = 0
    skipped_all_failed = 0
    skipped_already_parsed = 0

    with open(out_path, "a", encoding="utf-8") as f_out:
        with open(parsed_tracking_path, "a", encoding="utf-8") as f_parsed:
            for i, row in enumerate(tqdm(problems, desc="Generating oracle traces")):
                problem = row["problem"]
                expected_answer = row["expected_answer"]
                expected_int = _parse_expected_answer_to_int(expected_answer)

                if problem in already_parsed:
                    skipped_already_parsed += 1
                    continue
                if problem in already_done:
                    f_parsed.write(json.dumps({"problem": problem}, ensure_ascii=False) + "\n")
                    f_parsed.flush()
                    try:
                        os.fsync(f_parsed.fileno())
                    except Exception:
                        pass
                    already_parsed.add(problem)
                    continue

                # Log progress before starting expensive operation
                print(f"\n[Processing] idx={i}/{len(problems)} problem={problem[:80]}...")
                
                example = {"problem": problem, "expected_answer_int": expected_int}
                attempt_texts: list[str] = []
                attempt_answers: list[int | None] = []
                success_at = None
                had_failure_before_success = False
                failed_texts_before_success: list[str] = []
                successful_text: str | None = None
                avoid_memo = ""
                per_attempt_reflection: list[str] = []

                for attempt_idx in range(8):
                    print(f"  [Attempt {attempt_idx + 1}/8] Starting...")
                    extra = ""
                    if avoid_memo.strip():
                        extra = (
                            "Previous attempt(s) failed. Use the revised approach guidance below to predict correctly.\n"
                            f"REVISED_APPROACH_GUIDANCE:\n{avoid_memo.strip()}\n"
                        )
                    try:
                        res = score_example_with_tools(example, extra_instruction=extra)
                    except Exception as e:
                        print(f"  [ERROR] Attempt {attempt_idx + 1} failed: {e}")
                        res = {"answer": None, "response_text": f"Error: {str(e)}", "tool_called": False}
                    predicted = res.get("answer") if isinstance(res, dict) else None
                    response_text = res.get("response_text", "") if isinstance(res, dict) else ""
                    attempt_texts.append(response_text)
                    attempt_answers.append(predicted)

                    is_match = False
                    if predicted is not None and expected_int is not None:
                        is_match = str(predicted) == str(expected_int)

                    if is_match:
                        success_at = attempt_idx + 1
                        successful_text = response_text
                        if success_at == 1:
                            solved_first_try += 1
                        elif success_at == 2:
                            solved_second_try += 1
                        else:
                            solved_third_try += 1
                        print(
                            f"\n[OK] idx={i} solved in {success_at} attempt(s): predicted={predicted} expected={expected_int}"
                        )
                        break
                    else:
                        had_failure_before_success = True
                        failed_texts_before_success.append(response_text)
                        print(
                            f"\n[WRONG] idx={i} attempt={attempt_idx+1} predicted={predicted} expected={expected_int}"
                        )
                        if attempt_idx < 2:
                            memo = generate_reflection_avoidance(
                                problem=problem,
                                expected_answer=expected_answer,
                                predicted_answer=predicted,
                                failed_attempt_text=response_text,
                                avoid_so_far=avoid_memo,
                            )
                            per_attempt_reflection.append(memo)
                            if memo:
                                avoid_memo = (avoid_memo + "\n\n" + memo).strip() if avoid_memo else memo.strip()

                if success_at is None:
                    skipped_all_failed += 1
                elif success_at == 1:
                    pass
                elif had_failure_before_success and failed_texts_before_success and successful_text is not None:
                    oracle_trace = generate_oracle_mismatch_trace(
                        problem=problem,
                        expected_answer=expected_answer,
                        failed_attempt_texts=failed_texts_before_success,
                        successful_attempt_text=successful_text,
                    )
                    record = {
                        "problem": problem,
                        "expected_answer": expected_answer,
                        "attempts": [
                            {
                                "attempt": j + 1,
                                "predicted_answer": attempt_answers[j],
                                "response_text": attempt_texts[j],
                                "reflection_avoid_memo": per_attempt_reflection[j] if j < len(per_attempt_reflection) else None,
                            }
                            for j in range(len(attempt_texts))
                        ],
                        "avoid_memo_final": avoid_memo,
                        "solved_attempt": success_at,
                        "oracle_trace": oracle_trace,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    try:
                        os.fsync(f_out.fileno())
                    except Exception:
                        pass
                    oracle_traces_written += 1

                f_parsed.write(json.dumps({"problem": problem}, ensure_ascii=False) + "\n")
                f_parsed.flush()
                try:
                    os.fsync(f_parsed.fileno())
                except Exception:
                    pass
                already_parsed.add(problem)

    print("\n" + "=" * 80)
    print("Oracle traces run complete (from master)")
    print("=" * 80)
    print(f"Total in master (capped): {len(problems)}")
    print(f"Skipped (already parsed): {skipped_already_parsed}")
    print(f"Solved in 1 attempt: {solved_first_try}")
    print(f"Solved in 2 attempts: {solved_second_try}")
    print(f"Solved in 3 attempts: {solved_third_try}")
    print(f"Oracle traces written (fail->pass only): {oracle_traces_written}")
    print(f"Skipped (all 3 failed): {skipped_all_failed}")

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
                # Debug: Check if recipient attribute exists and what its value is
                message_recipient = getattr(last_message, 'recipient', None)
                if message_recipient == 'python':
                    tool_called = True  # Mark that tool was called
                    # Extract tool code for debugging
                    tool_code = last_message.content[0].text if hasattr(last_message.content[0], 'text') else str(last_message.content[0])
                    # Execute tool call
                    tool_responses = local_tool.process_sync_plus(last_message)
                    conversation.messages.extend(tool_responses)
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
        
        # Ensure response_text is the full assistant output (all turns), not just the final boxed line
        parts = []
        for msg in conversation.messages:
            if getattr(msg, "author", None) and getattr(msg.author, "role", None) == Role.ASSISTANT:
                if hasattr(msg, "content") and msg.content:
                    t = msg.content[0].text if hasattr(msg.content[0], "text") else str(msg.content[0])
                    if (t or "").strip():
                        parts.append(t.strip())
        if parts:
            response_text = "\n\n".join(parts)

        # Fallback: try to extract answer from any message if not yet set
        if answer is None:
            for msg in reversed(conversation.messages):
                if hasattr(msg, 'content') and msg.content:
                    text = msg.content[0].text if hasattr(msg.content[0], 'text') else str(msg.content[0])
                    pattern = r'\\boxed\s*\{\s*([0-9,]+)\s*\}'
                    matches = re.findall(pattern, text)
                    if matches:
                        try:
                            answer = int(matches[-1].replace(',', ''))
                            if 0 <= answer <= 99999:
                                break
                        except ValueError:
                            pass
        
        # Return dict with answer, response_text, and tool_called flag
        return {
            "answer": answer if answer is not None else None,
            "response_text": response_text if response_text is not None else "",
            "tool_called": tool_called
        }
        
    except Exception as e:
        print(f"Error scoring example: {e}")
        # Return dict format even on error
        return {
            "answer": None,
            "response_text": f"Error: {str(e)}",
            "tool_called": False
        }

def load_clustering_model():
    """Load k-means model, embedding model, and cluster stats for assigning clusters during prediction."""
    kmeans_model_path = LOCAL_DATA_DIR / "kmeans_model.pkl"
    cluster_stats_path = LOCAL_DATA_DIR / "cluster_stats.json"
    embedding_model_name = "all-MiniLM-L6-v2"
    
    # Check if clustering models exist
    if not kmeans_model_path.exists() or not cluster_stats_path.exists():
        print(f"\n⚠ Warning: Clustering models not found at:")
        print(f"    {kmeans_model_path}")
        print(f"    {cluster_stats_path}")
        print(f"  Running predictions without cluster assignment.")
        print(f"  To enable clustering, run cluster_high_mismatch.py first.\n")
        return None, None, None
    
    try:
        # Import clustering dependencies
        try:
            import pickle
            from sentence_transformers import SentenceTransformer
            from sklearn.cluster import KMeans
        except ImportError:
            print("⚠ Warning: Clustering dependencies not available. Installing...")
            os.system("pip install sentence-transformers scikit-learn -q")
            import pickle
            from sentence_transformers import SentenceTransformer
            from sklearn.cluster import KMeans
        
        # Load k-means model
        print(f"\n[Clustering] Loading clustering models...")
        with open(kmeans_model_path, 'rb') as f:
            kmeans_model = pickle.load(f)
        print(f"  ✓ Loaded k-means model from {kmeans_model_path}")
        
        # Load cluster stats
        with open(cluster_stats_path, 'r', encoding='utf-8') as f:
            cluster_stats = json.load(f)
        print(f"  ✓ Loaded cluster stats from {cluster_stats_path}")
        print(f"    Found {len(cluster_stats)} clusters")
        
        # Load embedding model
        print(f"  Loading embedding model: {embedding_model_name}...")
        embedding_model = SentenceTransformer(embedding_model_name)
        print(f"  ✓ Loaded embedding model")
        
        print(f"  Clustering enabled - will assign clusters during prediction\n")
        return kmeans_model, embedding_model, cluster_stats
        
    except Exception as e:
        print(f"\n⚠ Warning: Error loading clustering models: {e}")
        print(f"  Running predictions without cluster assignment.\n")
        return None, None, None


def assign_cluster_to_problem(problem_text: str, kmeans_model, embedding_model, cluster_stats):
    """Assign a cluster to a problem text using the loaded models.
    
    Returns:
        tuple: (cluster_id, cluster_mismatch_rate) or (None, None) if models not available
    """
    if kmeans_model is None or embedding_model is None or cluster_stats is None:
        return None, None
    
    try:
        # Generate embedding
        embedding = embedding_model.encode([problem_text])
        # Predict cluster
        cluster_id = int(kmeans_model.predict(embedding)[0])
        # Get mismatch rate
        cluster_stat = cluster_stats.get(str(cluster_id), cluster_stats.get(cluster_id, {}))
        mismatch_rate = float(cluster_stat.get("mismatch_rate", 0.0))
        return cluster_id, mismatch_rate
    except Exception as e:
        # Silently fail and return None if clustering fails
        return None, None


def main():
    parser = argparse.ArgumentParser(description="OpenMathReasoning scoring + oracle trace generation")
    parser.add_argument(
        "--mode",
        choices=["score_dataset", "oracle_traces"],
        default="oracle_traces",
        help="score_dataset: score OpenMathReasoning; oracle_traces: generate useful traces for no-match cases",
    )
    parser.add_argument(
        "--predictions-log",
        type=str,
        default=str(PREDICTIONS_LOG_PATH),
        help="Path to predictions_log.jsonl (used for --build-master exclusion and legacy oracle flow)",
    )
    parser.add_argument(
        "--oracle-out",
        type=str,
        default=str(ORACLE_TRACES_OUT_PATH),
        help="Output JSONL path for oracle traces dataset",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of problems to process in oracle_traces mode",
    )
    parser.add_argument(
        "--build-master",
        action="store_true",
        help="Build master JSONL from all_examples_with_clusters (int 0–99999, not in predictions_log), then exit",
    )
    parser.add_argument(
        "--all-examples",
        type=str,
        default=str(ALL_EXAMPLES_PATH),
        help="Path to all_examples_with_clusters.jsonl (for --build-master)",
    )
    parser.add_argument(
        "--master",
        type=str,
        default=str(MASTER_EXAMPLES_PATH),
        help="Path to master examples JSONL (input for oracle_traces)",
    )
    parser.add_argument(
        "--parsed-tracking",
        type=str,
        default=str(PARSED_TRACKING_PATH),
        help="JSONL path to track parsed examples (skip on re-run)",
    )
    args = parser.parse_args()

    if args.build_master:
        print("=" * 80)
        print("Build master examples (int 0–99999, not in predictions_log)")
        print("=" * 80)
        n = build_master_examples_file(
            all_examples_path=Path(args.all_examples),
            predictions_log_path=Path(args.predictions_log),
            master_path=Path(args.master),
        )
        print(f"Wrote {n} examples to {args.master}")
        return

    if OpenAI is None or load_harmony_encoding is None or KernelManager is None:
        raise RuntimeError(
            "Missing optional dependencies for tool-calling execution. "
            "Install: `pip install openai openai-harmony jupyter_client`."
        )

    if args.mode == "oracle_traces":
        print("=" * 80)
        print("Oracle Traces Generator (from master)")
        print("=" * 80)
        run_oracle_traces_from_master(
            master_path=Path(args.master),
            out_path=Path(args.oracle_out),
            parsed_tracking_path=Path(args.parsed_tracking),
            max_examples=args.max_examples,
        )
        return

    if load_dataset is None or Dataset is None:
        raise RuntimeError(
            "Missing optional dependency `datasets`. Install it to run --mode score_dataset "
            "(e.g., `pip install datasets`)."
        )

    print("="*80)
    print("OpenMathReasoning Dataset Preparation (with Tool Calling)")
    print("="*80)
    
    # Load clustering models if available
    kmeans_model, embedding_model, cluster_stats = load_clustering_model()
    
    # Check if high_mismatch_clusters.jsonl exists - if so, use it directly and skip all earlier steps
    high_mismatch_path = LOCAL_DATA_DIR / "high_mismatch_clusters.jsonl"
    
    # Initialize filtered_examples to None - will be set in either branch
    filtered_examples = None
    
    if high_mismatch_path.exists():
        print(f"\n[Direct Mode] High-mismatch clusters file found at {high_mismatch_path}")
        print("  Loading high-mismatch clusters directly (skipping all earlier steps)...")
        try:
            # Load from high_mismatch_clusters.jsonl file
            filtered_examples = []
            with open(high_mismatch_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading high-mismatch clusters"):
                    ex_dict = json.loads(line.strip())
                    # Convert expected_answer_str back to int for processing (if available)
                    if 'expected_answer_str' in ex_dict:
                        try:
                            ex_dict['expected_answer_int'] = int(ex_dict['expected_answer_str'])
                        except (ValueError, OverflowError):
                            ex_dict['expected_answer_int'] = None
                    elif 'expected_answer_int' not in ex_dict:
                        # Try to get from expected_answer field if available
                        if 'expected_answer' in ex_dict:
                            try:
                                ex_dict['expected_answer_int'] = int(ex_dict['expected_answer'])
                            except (ValueError, OverflowError, TypeError):
                                ex_dict['expected_answer_int'] = None
                        else:
                            ex_dict['expected_answer_int'] = None
                    filtered_examples.append(ex_dict)
            print(f"✓ Loaded {len(filtered_examples)} high-mismatch cluster examples")
            print(f"  Proceeding directly to prediction/scoring (skipping Steps 1-3)...")
        except Exception as e:
            print(f"✗ Error loading high-mismatch clusters: {e}")
            print("  Falling back to regular filtered dataset...")
            high_mismatch_path = None  # Mark as not found, will continue to check filtered path
    
    # Check if filtered dataset already exists - if so, skip all earlier steps (only if high_mismatch not found)
    if filtered_examples is None:
        # Use JSONL for memory-efficient streaming I/O
        local_filtered_path = LOCAL_DATA_DIR / "openmath_filtered_integers.jsonl"
        
        if local_filtered_path.exists():
            print(f"\n[Skip Steps 1-3] Filtered dataset found at {local_filtered_path}")
        print("  Loading filtered dataset directly (skipping download and filtering)...")
        try:
            # Load from JSONL file (memory-efficient, streaming)
            filtered_examples = []
            with open(local_filtered_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading filtered examples"):
                    ex_dict = json.loads(line.strip())
                    # Convert expected_answer_str back to int for processing (if available)
                    if 'expected_answer_str' in ex_dict:
                        try:
                            ex_dict['expected_answer_int'] = int(ex_dict['expected_answer_str'])
                        except (ValueError, OverflowError):
                            ex_dict['expected_answer_int'] = None
                    elif 'expected_answer_int' not in ex_dict:
                        ex_dict['expected_answer_int'] = None
                    filtered_examples.append(ex_dict)
            print(f"✓ Loaded {len(filtered_examples)} filtered examples from existing file")
        except Exception as e:
            print(f"✗ Error loading filtered dataset: {e}")
            print("  Will recreate filtered dataset from scratch...")
            # Continue to dataset loading below
            local_filtered_path = None  # Mark as not found
        else:
            local_filtered_path = None  # Mark as not found (already loaded)
    
    # Only do dataset loading and filtering if filtered dataset doesn't exist
    if filtered_examples is None:
        local_filtered_path = LOCAL_DATA_DIR / "openmath_filtered_integers.jsonl"
        if local_filtered_path is None or not local_filtered_path.exists():
            # Step 1: Load dataset from local file if exists, otherwise download
            print("\n[Step 1] Loading dataset (split='tir' only)...")
            local_full_path = LOCAL_DATA_DIR / "openmath_full.parquet"
            
            if local_full_path.exists():
                print(f"  Loading from local file: {local_full_path}")
                try:
                    dataset = Dataset.from_parquet(str(local_full_path))
                    print(f"✓ Loaded {len(dataset)} examples from local file")
                except Exception as e:
                    print(f"✗ Error loading from local file: {e}")
                    print("  Falling back to downloading from HuggingFace...")
                    dataset = load_dataset("nvidia/OpenMathReasoning", split="tir")
                    print(f"✓ Loaded {len(dataset)} examples from HuggingFace")
                    # Save for future use
                    dataset.to_parquet(str(local_full_path))
                    print(f"✓ Saved to {local_full_path}")
            else:
                print(f"  Local file not found. Downloading from HuggingFace...")
                try:
                    dataset = load_dataset("nvidia/OpenMathReasoning", split="tir")
                    print(f"✓ Loaded {len(dataset)} examples from 'tir' split")
                    # Save for future use
                    dataset.to_parquet(str(local_full_path))
                    print(f"✓ Saved to {local_full_path}")
                except Exception as e:
                    print(f"✗ Error loading dataset: {e}")
                    return
            
            # Step 3: Filter for integer answers
            print("\n[Step 3] Filtering for integer answers...")
            print(f"Dataset features: {list(dataset.features.keys())}")
            
            # Try different possible field names for the answer
            answer_field = None
            for field in ["answer", "expected_answer", "target", "solution", "final_answer"]:
                if field in dataset[0]:
                    answer_field = field
                    print(f"✓ Found answer field: {answer_field}")
                    break
            
            if answer_field is None:
                print("⚠ Could not find answer field. Checking first example:")
                print(json.dumps(dataset[0], indent=2))
                answer_field = input("Enter the field name for expected answer: ").strip()
            
            # Step 3.5: Filter and save filtered dataset using streaming JSONL (memory-efficient)
            print("\n[Step 3.5] Filtering and saving dataset (streaming to JSONL)...")
            local_filtered_path = LOCAL_DATA_DIR / "openmath_filtered_integers.jsonl"
            
            filtered_count = 0
            # Stream filter and write directly to JSONL (no memory accumulation)
            with open(local_filtered_path, 'w', encoding='utf-8') as f:
                for example in tqdm(dataset, desc="Filtering and saving integer answers"):
                    answer = example.get(answer_field)
                    if is_integer_answer(answer):
                        int_answer = extract_integer_from_text(answer)
                        if int_answer is not None:
                            example_copy = dict(example)
                            # Convert to string to avoid overflow issues with very large integers
                            example_copy["expected_answer_str"] = str(int_answer)
                            try:
                                # Try to keep as int if it fits in int64 range
                                if abs(int_answer) <= 2**63 - 1:
                                    example_copy["expected_answer_int"] = int_answer
                                else:
                                    example_copy["expected_answer_int"] = None  # Too large for int64
                            except OverflowError:
                                example_copy["expected_answer_int"] = None
                            
                            # Write directly to file (streaming, no memory accumulation)
                            f.write(json.dumps(example_copy, ensure_ascii=False) + '\n')
                            filtered_count += 1
                            
                            # Flush periodically for safety
                            if filtered_count % 10000 == 0:
                                f.flush()
            
            print(f"\n✓ Found {filtered_count} examples with integer answers")
            print(f"  (from {len(dataset)} total examples)")
            print(f"✓ Saved filtered dataset to {local_filtered_path} (JSONL format, memory-efficient)")
            
            # Now load the filtered examples from JSONL for processing
            print("\n[Loading filtered examples for processing...]")
            filtered_examples = []
            with open(local_filtered_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading filtered examples"):
                    ex_dict = json.loads(line.strip())
                    # Convert expected_answer_str back to int for processing (if available)
                    if 'expected_answer_str' in ex_dict:
                        try:
                            ex_dict['expected_answer_int'] = int(ex_dict['expected_answer_str'])
                        except (ValueError, OverflowError):
                            ex_dict['expected_answer_int'] = None
                    elif 'expected_answer_int' not in ex_dict:
                        ex_dict['expected_answer_int'] = None
                    filtered_examples.append(ex_dict)
            print(f"✓ Loaded {len(filtered_examples)} filtered examples into memory")
    
    # Ensure filtered_examples is defined before Step 4
    if filtered_examples is None or len(filtered_examples) == 0:
        raise RuntimeError("filtered_examples was not loaded or is empty. This should not happen.")
    
    # Step 4: Iterative selection process
    print("\n[Step 4] Starting iterative selection process with tool calling...")
    print(f"  Batch size: {BATCH_SIZE} examples per iteration")
    print(f"  GPU batch size: {GPU_BATCH_SIZE} (processing {GPU_BATCH_SIZE} examples concurrently)")
    print(f"  Target: Score all examples and mark as 'match' or 'no match'")
    
    # Sort by solution length descending
    solution_field = None
    for field in ["solution", "generated_solution", "response", "completion"]:
        if field in filtered_examples[0]:
            solution_field = field
            print(f"✓ Found solution field: {solution_field}")
            break
    
    if solution_field is None:
        print("⚠ Could not find solution field. Checking first example:")
        print(json.dumps(filtered_examples[0], indent=2))
        solution_field = input("Enter the field name for generated solution: ").strip()
    
    # Calculate solution lengths
    print("\nCalculating solution lengths...")
    for example in tqdm(filtered_examples, desc="Computing lengths"):
        solution_text = str(example.get(solution_field, ""))
        example["solution_length"] = len(solution_text)
    
    # Sort by solution length descending
    filtered_examples.sort(key=lambda x: x["solution_length"], reverse=True)
    print(f"✓ Sorted by solution length (descending)")
    
    # Track processed indices and all scored examples (matching and non-matching)
    processed_indices = set()
    
    # Check if predictions_log.jsonl exists and load already processed indices
    prediction_log_path = LOCAL_DATA_DIR / "/home/malam/wsl-tunix/imo/openmath_data/predictions_log.jsonl"
    if prediction_log_path.exists():
        print(f"\n✓ Found existing predictions log: {prediction_log_path}")
        try:
            with open(prediction_log_path, 'r', encoding='utf-8') as log_f:
                existing_indices = set()
                for line in log_f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            idx = entry.get("idx")
                            if idx is not None:
                                existing_indices.add(idx)
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            # Skip malformed lines
                            continue
                processed_indices.update(existing_indices)
                print(f"✓ Loaded {len(existing_indices)} already-processed indices from predictions log")
                print(f"  Will skip these examples and continue from where we left off")
        except Exception as e:
            print(f"⚠ Warning: Could not read existing predictions log: {e}")
            print(f"  Starting fresh (will append to existing log)")
    
    all_scored_examples = []
    iteration = 0
    
    while len(processed_indices) < len(filtered_examples):
        iteration += 1
        print(f"\n{'='*80}")
        print(f"Iteration {iteration}")
        print(f"{'='*80}")
        
        # Select next batch of unprocessed examples
        batch_examples = []
        for idx, example in enumerate(filtered_examples):
            if idx not in processed_indices and len(batch_examples) < BATCH_SIZE:
                batch_examples.append((idx, example))
        
        if not batch_examples:
            print("No more examples to process.")
            break
        
        print(f"Processing batch of {len(batch_examples)} examples...")
        print(f"  Using GPU batch size: {GPU_BATCH_SIZE} (processing {GPU_BATCH_SIZE} examples concurrently)")
        
        # Score each example in batch with tool calling (parallel processing)
        batch_matches = 0
        batch_non_matches = 0
        batch_results = []
        
        def score_single_example(idx_example_tuple):
            """Score a single example and return result."""
            idx, example = idx_example_tuple
            expected_answer = example.get("expected_answer_int")
            
            # Handle large integers stored as strings
            if expected_answer is None and "expected_answer_str" in example:
                try:
                    # Try to parse as int, but keep as string if too large
                    expected_answer = int(example["expected_answer_str"])
                except (ValueError, OverflowError):
                    # If too large, we'll compare as strings later
                    expected_answer = example["expected_answer_str"]
            
            # Score with model (using tool calling)
            result = score_example_with_tools(example)
            
            # Handle both old format (int) and new format (dict), or None on error
            tool_called = False  # Default value
            if isinstance(result, dict):
                predicted_answer = result.get("answer")
                response_text = result.get("response_text", "")
                tool_called = result.get("tool_called", False)
            elif result is None:
                # Error case
                predicted_answer = None
                response_text = ""
                tool_called = False
            else:
                # Fallback for old format (int)
                predicted_answer = result
                response_text = ""
                tool_called = False
            
            # Add score_match column: "match" or "no match"
            example_copy = dict(example)  # Create copy to avoid modifying original
            
            # Compare predicted vs expected (handle both int and string)
            is_match = False
            if predicted_answer is not None and expected_answer is not None:
                # Convert both to strings for comparison to handle large integers
                pred_str = str(predicted_answer)
                exp_str = str(expected_answer)
                is_match = (pred_str == exp_str)
            
            score_match = "match" if is_match else "no match"
            if is_match:
                example_copy["score_match"] = score_match
                match_count = 1
            else:
                example_copy["score_match"] = score_match
                match_count = 0
            
            # Store predicted answer and tool_called flag for reference
            example_copy["predicted_answer"] = predicted_answer if predicted_answer is not None else None
            example_copy["response_text"] = response_text
            example_copy["tool_called"] = tool_called
            
            # Assign cluster if clustering models are available
            problem_text = example.get("problem", example.get("question", ""))
            cluster_id, cluster_mismatch_rate = assign_cluster_to_problem(
                problem_text, kmeans_model, embedding_model, cluster_stats
            )
            if cluster_id is not None:
                example_copy["predicted_cluster"] = cluster_id
                example_copy["cluster_mismatch_rate"] = cluster_mismatch_rate
            
            # Save JSON after each prediction (save only specified fields)
            try:
                prediction_log_path = LOCAL_DATA_DIR / "/home/malam/wsl-tunix/imo/openmath_data/predictions_log.jsonl"
                # Extract only specified fields from example
                prediction_entry = {
                    "idx": idx,
                    "problem": problem_text,
                    "expected_answer": expected_answer,
                    "problem_type": example.get("problem_type", None),
                    "problem_source": example.get("problem_source", None),
                    "generation_model": example.get("generation_model", None),
                    "used_in_kaggle": example.get("used_in_kaggle", None),
                    "predicted_answer": predicted_answer,
                    "response_text": response_text,
                    "score_match": score_match,
                    "tool_called": tool_called,
                    "predicted_cluster": cluster_id,
                    "cluster_mismatch_rate": cluster_mismatch_rate if cluster_mismatch_rate is not None else None,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                with open(prediction_log_path, 'a', encoding='utf-8') as log_f:
                    log_f.write(json.dumps(prediction_entry, ensure_ascii=False) + '\n')
                    log_f.flush()  # Ensure it's written immediately
            except Exception as e:
                print(f"\n⚠ Warning: Could not save prediction log for idx {idx}: {e}")
            
            return (idx, example_copy, match_count)
        
        # Process examples in parallel batches
        with ThreadPoolExecutor(max_workers=GPU_BATCH_SIZE) as executor:
            futures = {executor.submit(score_single_example, (idx, ex)): (idx, ex) 
                      for idx, ex in batch_examples}
            
            # Process completed futures as they finish
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Scoring batch {iteration}"):
                try:
                    idx, example_copy, match_count = future.result()
                    batch_results.append((idx, example_copy, match_count))
                    
                    if match_count == 1:
                        batch_matches += 1
                    else:
                        batch_non_matches += 1
                except Exception as e:
                    # Handle errors gracefully
                    idx, ex = futures[future]
                    print(f"\n⚠ Error processing example {idx}: {e}")
                    example_copy = dict(ex)
                    example_copy["score_match"] = "no match"
                    example_copy["predicted_answer"] = None
                    example_copy["response_text"] = f"Error: {str(e)}"
                    example_copy["tool_called"] = False
                    batch_results.append((idx, example_copy, 0))
                    batch_non_matches += 1
                    
                    # Log error case to JSON (save only specified fields)
                    try:
                        prediction_log_path = LOCAL_DATA_DIR / "/home/malam/wsl-tunix/imo/openmath_data/predictions_log.jsonl"
                        # Extract expected_answer for error case
                        error_expected_answer = ex.get("expected_answer_int")
                        if error_expected_answer is None and "expected_answer_str" in ex:
                            try:
                                error_expected_answer = int(ex["expected_answer_str"])
                            except (ValueError, OverflowError):
                                error_expected_answer = ex["expected_answer_str"]
                        # Assign cluster if clustering models are available
                        error_problem_text = ex.get("problem", ex.get("question", ""))
                        error_cluster_id, error_cluster_mismatch_rate = assign_cluster_to_problem(
                            error_problem_text, kmeans_model, embedding_model, cluster_stats
                        )
                        
                        # Extract only specified fields from example
                        prediction_entry = {
                            "idx": idx,
                            "problem": error_problem_text,
                            "expected_answer": error_expected_answer,
                            "problem_type": ex.get("problem_type", None),
                            "problem_source": ex.get("problem_source", None),
                            "generation_model": ex.get("generation_model", None),
                            "used_in_kaggle": ex.get("used_in_kaggle", None),
                            "predicted_answer": None,
                            "response_text": f"Error: {str(e)}",
                            "score_match": "no match",
                            "tool_called": False,
                            "predicted_cluster": error_cluster_id,
                            "cluster_mismatch_rate": error_cluster_mismatch_rate if error_cluster_mismatch_rate is not None else None,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        with open(prediction_log_path, 'a', encoding='utf-8') as log_f:
                            log_f.write(json.dumps(prediction_entry, ensure_ascii=False) + '\n')
                            log_f.flush()
                    except Exception as log_error:
                        print(f"\n⚠ Warning: Could not save error prediction log for idx {idx}: {log_error}")
        
        # Sort results by original index and add to all_scored_examples
        batch_results.sort(key=lambda x: x[0])
        for idx, example_copy, _ in batch_results:
            all_scored_examples.append(example_copy)
            processed_indices.add(idx)
        
        total_matches = sum(1 for ex in all_scored_examples if ex.get("score_match") == "match")
        print(f"\n✓ Batch {iteration} complete:")
        print(f"  Processed: {len(batch_examples)}")
        print(f"  Matches: {batch_matches}")
        print(f"  No matches: {batch_non_matches}")
        if len(batch_examples) > 0:
            print(f"  Match rate: {batch_matches/len(batch_examples)*100:.2f}%")
        print(f"  Total matches so far: {total_matches}/{len(all_scored_examples)}")
        
        # Save progress after each batch (all examples with score_match column)
        if all_scored_examples:
            progress_dataset = Dataset.from_list(all_scored_examples)
            progress_path = LOCAL_DATA_DIR / f"openmath_scored_iter{iteration}.parquet"
            progress_dataset.to_parquet(str(progress_path))
            print(f"  ✓ Saved progress to {progress_path} ({len(all_scored_examples)} examples)")
        
        # Save checkpoint (all examples scored so far)
        checkpoint_dataset = Dataset.from_list(all_scored_examples)
        checkpoint_path = LOCAL_DATA_DIR / "openmath_scored_checkpoint.parquet"
        checkpoint_dataset.to_parquet(str(checkpoint_path))
        print(f"  ✓ Checkpoint saved to {checkpoint_path}")
        
        print(f"\nProcessed {len(processed_indices)}/{len(filtered_examples)} examples")
    
    # Final save - save ALL examples with score_match column
    print(f"\n{'='*80}")
    print("Scoring Complete!")
    print(f"{'='*80}")
    print(f"Total examples processed: {len(processed_indices)}")
    
    matches_count = sum(1 for ex in all_scored_examples if ex.get("score_match") == "match")
    no_matches_count = sum(1 for ex in all_scored_examples if ex.get("score_match") == "no match")
    
    print(f"Total matching examples: {matches_count}")
    print(f"Total non-matching examples: {no_matches_count}")
    
    if all_scored_examples:
        final_dataset = Dataset.from_list(all_scored_examples)
        final_path = LOCAL_DATA_DIR / "openmath_scored_final.parquet"
        final_dataset.to_parquet(str(final_path))
        print(f"\n✓ Final dataset saved to {final_path}")
        print(f"  Total examples: {len(all_scored_examples)}")
        print(f"  Matches: {matches_count} ({matches_count/len(all_scored_examples)*100:.2f}%)")
        print(f"  No matches: {no_matches_count} ({no_matches_count/len(all_scored_examples)*100:.2f}%)")
        
        # Also save as CSV for easy viewing
        final_csv_path = LOCAL_DATA_DIR / "openmath_scored_final.csv"
        #final_dataset.to_csv(str(final_csv_path))
        print(f"  ✓ Also saved as CSV: {final_csv_path}")
    else:
        print("\n⚠ No examples were scored.")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
