#!/usr/bin/env python3
"""
Convert high_mismatch_clusters.jsonl to Harmony format for SFT/DPGRPO training.

Parses generated_solution text to extract tool calls and final answers,
converting them to Harmony message format with system prompts.
"""

import json
import re
from pathlib import Path
from tqdm import tqdm

# Configuration
LOCAL_DATA_DIR = Path("/home/malam/wsl-tunix/imo/openmath_data")
INPUT_PATH = LOCAL_DATA_DIR / "high_mismatch_clusters.jsonl"
OUTPUT_PATH = LOCAL_DATA_DIR / "high_mismatch_harmony.jsonl"

# Fixed system prompts (as per user specification)
SYSTEM_PROMPTS = [
    {
        "role": "system",
        "content": "You are the top International Mathematical Olympiad (IMO) competitor. The final answer must be a non-negative integer between 0 and 99999. You must place the final integer answer inside \\boxed{}."
    },
    {
        "role": "system",
        "content": "Use this tool to execute Python code. The environment is a stateful Jupyter notebook. You must use print() to output results."
    },
    {
        "role": "system",
        "content": "You have access to `math`, `numpy` and `sympy` to solve the problem."
    }
]


def remove_think_tags(text: str) -> str:
    """Remove <think> tags and their content from text."""
    # Remove <think>...</think> tags (case-insensitive, multiline)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Also remove <think> tags
    text = re.sub(r'<think>', '', text, flags=re.IGNORECASE)
    return text.strip()


def extract_tool_calls_and_final_answer(solution_text: str):
    """Extract tool calls and final answer from solution text.
    
    Returns:
        tool_calls: List of tool call dictionaries
        tool_outputs: List of tool outputs (strings)
        final_answer: Final answer text (contains \\boxed{...})
    """
    tool_calls = []
    tool_outputs = []
    
    # Extract tool calls BEFORE removing tags (they might be inside redacted_reasoning/think tags)
    # Pattern: <tool_call>...code...</tool_call> (case-insensitive, multiline)
    tool_call_pattern = r'<tool_call>\s*\n?(.*?)\s*\n?</tool_call>'
    tool_call_matches = list(re.finditer(tool_call_pattern, solution_text, re.DOTALL | re.IGNORECASE))
    
    # Extract outputs - pattern: ```output\n...\n```
    output_pattern = r'```output\s*\n(.*?)\n```'
    output_matches = list(re.finditer(output_pattern, solution_text, re.DOTALL))
    
    # Process tool calls
    for match in tool_call_matches:
        code_content = match.group(1).strip()
        # Clean up code - remove any markdown formatting that might be inside
        code_content = re.sub(r'```python\s*\n?', '', code_content)
        code_content = re.sub(r'```\s*\n?', '', code_content)
        code_content = code_content.strip()
        
        if code_content:
            tool_calls.append({
                "tool_name": "python",
                "arguments": {
                    "code": code_content
                }
            })
    
    # Process outputs
    for match in output_matches:
        output_text = match.group(1).strip()
        if output_text:
            tool_outputs.append(output_text)
    
    # Remove think tags for final answer extraction (but tool calls already extracted)
    solution_clean = remove_think_tags(solution_text)
    
    # Extract final answer - look for \boxed{...} or boxed{...} (prefer last occurrence)
    final_answer_matches = list(re.finditer(r'\\?boxed\{([^}]+)\}', solution_clean, re.IGNORECASE))
    if final_answer_matches:
        # Use the last boxed answer (most likely the final answer)
        last_match = final_answer_matches[-1]
        answer_value = last_match.group(1)
        final_answer = f"\\boxed{{{answer_value}}}"
    else:
        # Fallback: look for "Final Answer" section
        final_answer = None
    
    return tool_calls, tool_outputs, final_answer


def build_harmony_messages(problem: str, expected_answer: str, generated_solution: str) -> list:
    """Build Harmony format messages from problem and solution.
    
    Args:
        problem: Problem text
        expected_answer: Expected answer (for validation)
        generated_solution: Generated solution text (may contain tool calls)
    
    Returns:
        List of messages in Harmony format
    """
    messages = []
    
    # Add fixed system prompts
    messages.extend(SYSTEM_PROMPTS)
    
    # Add user message with problem
    messages.append({
        "role": "user",
        "content": problem
    })
    
    # Parse solution to extract tool calls and final answer
    tool_calls, tool_outputs, final_answer = extract_tool_calls_and_final_answer(generated_solution)
    
    # If we have tool calls, add them as assistant messages with tool_calls
    if tool_calls:
        # Add assistant message with tool calls in commentary channel
        messages.append({
            "role": "assistant",
            "channel": "commentary",
            "tool_calls": tool_calls
        })
        
        # Add tool responses (pair tool calls with outputs)
        for i, output in enumerate(tool_outputs):
            if i < len(tool_calls):
                messages.append({
                    "role": "tool",
                    "name": tool_calls[i]["tool_name"],
                    "content": output
                })
    
    # Add final assistant message with answer
    if final_answer:
        messages.append({
            "role": "assistant",
            "channel": "final",
            "content": final_answer
        })
    elif expected_answer:
        # Fallback: construct answer from expected_answer
        messages.append({
            "role": "assistant",
            "channel": "final",
            "content": f"\\boxed{{{expected_answer}}}"
        })
    else:
        # No answer found, use a placeholder
        messages.append({
            "role": "assistant",
            "channel": "final",
            "content": "\\boxed{0}"  # Default placeholder
        })
    
    return messages


def convert_example_to_harmony(example: dict) -> dict:
    """Convert a single example to Harmony format.
    
    Args:
        example: Example dictionary from high_mismatch_clusters.jsonl
    
    Returns:
        Dictionary in Harmony format with problem, expected_answer, and generated_solution fields
    """
    problem = example.get("problem", "")
    expected_answer = example.get("expected_answer", example.get("expected_answer_str", ""))
    generated_solution = example.get("generated_solution", "")
    
    # Build messages from solution
    messages = build_harmony_messages(problem, expected_answer, generated_solution)
    
    # Return in format specified by user
    harmony_example = {
        "problem": problem,
        "expected_answer": str(expected_answer),  # Ensure string format
        "generated_solution": messages
    }
    
    return harmony_example


def main():
    """Main conversion function."""
    print("="*80)
    print("Converting high_mismatch_clusters.jsonl to Harmony format")
    print("="*80)
    
    if not INPUT_PATH.exists():
        print(f"✗ Error: Input file not found: {INPUT_PATH}")
        return
    
    print(f"\nInput:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    
    # Read and convert examples
    converted_count = 0
    skipped_count = 0
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        
        # Count total lines first (for progress bar)
        total_lines = sum(1 for _ in f_in)
        f_in.seek(0)  # Reset file pointer
        
        for line in tqdm(f_in, total=total_lines, desc="Converting examples"):
            line = line.strip()
            if not line:
                continue
            
            try:
                example = json.loads(line)
                
                # Convert to Harmony format
                harmony_example = convert_example_to_harmony(example)
                
                # Write to output file
                f_out.write(json.dumps(harmony_example, ensure_ascii=False) + '\n')
                converted_count += 1
                
            except Exception as e:
                print(f"\n⚠ Warning: Error converting example: {e}")
                skipped_count += 1
                continue
    
    print(f"\n{'='*80}")
    print("Conversion Complete!")
    print(f"{'='*80}")
    print(f"✓ Converted: {converted_count} examples")
    if skipped_count > 0:
        print(f"⚠ Skipped:   {skipped_count} examples")
    print(f"✓ Output saved to: {OUTPUT_PATH}")
    
    # Validate output by reading first example
    print(f"\nValidating output...")
    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        if first_line:
            first_example = json.loads(first_line)
            print(f"✓ First example structure:")
            print(f"  - problem: {first_example.get('problem', '')[:80]}...")
            print(f"  - expected_answer: {first_example.get('expected_answer', 'N/A')}")
            print(f"  - generated_solution: {len(first_example.get('generated_solution', []))} messages")
            
            # Show message structure
            messages = first_example.get('generated_solution', [])
            print(f"\n  Message breakdown:")
            for i, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                channel = msg.get('channel', '')
                has_tool_calls = 'tool_calls' in msg
                content_preview = str(msg.get('content', ''))[:50] if 'content' in msg else ''
                print(f"    {i+1}. role={role} channel={channel} tool_calls={has_tool_calls} content={content_preview}...")


if __name__ == "__main__":
    main()
