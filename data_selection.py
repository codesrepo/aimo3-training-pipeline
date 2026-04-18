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
import queue
import contextlib
import threading
from pathlib import Path
from datasets import load_dataset, Dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add imo directory to path for imports
imo_dir = os.path.dirname(os.path.abspath(__file__))
if imo_dir not in sys.path:
    sys.path.insert(0, imo_dir)

from openai import OpenAI
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    SystemContent,
    ReasoningEffort,
    ToolNamespaceConfig,
    Author,
    Message,
    Role,
    TextContent,
    Conversation
)
from jupyter_client import KernelManager, BlockingKernelClient

# GPT-OSS model configuration
BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "openai/gpt-oss-120b"

# Configuration
BATCH_SIZE = 100000  # Top 100k per iteration
GPU_BATCH_SIZE = 8  # Process 8 examples concurrently on GPU
LOCAL_DATA_DIR = Path("/home/malam/wsl-tunix/imo/openmath_data")
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

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
    
    preference_prompt = (
        'You have access to `math`, `numpy` and `sympy` to solve the problem.'
    )
    
    # API settings
    served_model_name = 'gpt-oss'
    temperature = 0.0
    top_logprobs = 1
    min_p = 0.0
    seed = 42
    context_tokens = 65536
    buffer_tokens = 512
    turns = 128
    jupyter_timeout = 300.0
    sandbox_timeout = 10.0
    session_timeout = 300

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

def score_example_with_tools(example):
    """Score a single example using local GPT-OSS model with tool calling."""
    try:
        # Extract problem text
        question = example.get("question", example.get("problem", ""))
        if not question:
            return None
        
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
        messages = template.apply_chat_template(
            CFG.system_prompt,
            f"{question} {CFG.preference_prompt}",
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
                # Call API
                completion = client.completions.create(
                    model=MODEL_NAME,
                    prompt=prompt_ids,
                    max_tokens=max_tokens,
                    temperature=CFG.temperature,
                    logprobs=CFG.top_logprobs,
                    seed=CFG.seed,
                    stream=False,
                    extra_body={
                        "min_p": CFG.min_p,
                        "stop_token_ids": stop_token_ids,
                        "return_token_ids": True
                    }
                )
                
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
                    response_text = answer_text  # Store the response text
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
                    
            except Exception as e:
                print(f"Error in turn {turn}: {e}")
                break
            
            # Small delay between turns
            if turn < max_turns - 1:
                time.sleep(0.05)
        
        # Cleanup sandbox
        sandbox.reset()
        sandbox.close()
        
        # Fallback: try to extract from any message
        if answer is None:
            for msg in reversed(conversation.messages):
                if hasattr(msg, 'content') and msg.content:
                    text = msg.content[0].text if hasattr(msg.content[0], 'text') else str(msg.content[0])
                    if response_text is None:
                        response_text = text  # Store response text even in fallback
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
    prediction_log_path = LOCAL_DATA_DIR / "predictions_log.jsonl"
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
                prediction_log_path = LOCAL_DATA_DIR / "predictions_log.jsonl"
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
                        prediction_log_path = LOCAL_DATA_DIR / "predictions_log.jsonl"
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
        final_dataset.to_csv(str(final_csv_path))
        print(f"  ✓ Also saved as CSV: {final_csv_path}")
    else:
        print("\n⚠ No examples were scored.")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
