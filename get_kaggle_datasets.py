"""
Download Crystal-Math-Preview, filter (integer answer, non-null solution),
combine with hard_50_math_problems_set_v6.csv and oracle_traces_no_match.jsonl,
dedupe on problem, filter answer in [0, 99999], save as parquet.
"""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def _answer_is_integer(val) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    try:
        s = str(val).strip().replace(",", "")
        int(s)
        return True
    except (ValueError, TypeError):
        return False


def main():
    # --- 1. Load Crystal-Math-Preview (original) ---
    print("Loading ycchen/Crystal-Math-Preview (original)...")
    ds = load_dataset("ycchen/Crystal-Math-Preview", "original")
    # Use train split if exists, else first split
    split = "train" if "train" in ds else list(ds.keys())[0]
    df_crystal = ds[split].to_pandas()

    print(f"  Crystal-Math raw rows: {len(df_crystal)}")
    print(f"  Crystal-Math columns: {list(df_crystal.columns)}")

    # Normalize column names (dataset may use 'question' vs 'problem', etc.)
    col_map = {}
    for c in df_crystal.columns:
        c_lower = c.lower()
        if c_lower in ("question", "problem"):
            col_map[c] = "problem"
        elif c_lower in ("id", "idx", "index"):
            col_map[c] = "id"
    df_crystal = df_crystal.rename(columns=col_map)

    # Ensure we have problem; use 'question' if problem missing
    if "problem" not in df_crystal.columns and "question" in df_crystal.columns:
        df_crystal["problem"] = df_crystal["question"]

    # Filter: answer is integer and solution is not null
    if "solution" not in df_crystal.columns:
        # Some configs use "solution_detail" or "steps"
        for alt in ("solution_detail", "steps", "solution_detail", "proof"):
            if alt in df_crystal.columns:
                df_crystal["solution"] = df_crystal[alt]
                break
        else:
            df_crystal["solution"] = pd.NA

    mask_int_answer = df_crystal["answer"].apply(_answer_is_integer)
    mask_solution_ok = df_crystal["solution"].notna() & (df_crystal["solution"].astype(str).str.strip() != "")
    df_crystal = df_crystal.loc[mask_int_answer & mask_solution_ok].copy()

    # Keep: id, source, answer, problem, solution (only columns that exist)
    keep_crystal = ["id", "source", "answer", "problem", "solution"]
    existing = [c for c in keep_crystal if c in df_crystal.columns]
    df_crystal = df_crystal[existing].copy()
    for c in keep_crystal:
        if c not in df_crystal.columns:
            df_crystal[c] = None
    df_crystal = df_crystal[["id", "source", "answer", "problem", "solution"]]

    df_crystal["datatype"] = "crystal_math"
    print(f"  Crystal-Math after filter (integer answer, solution not null): {len(df_crystal)}")

    # --- 2. Load hard_50 CSV ---
    csv_path = Path(__file__).resolve().parent / "datasets" / "hard_50_math_problems_set_v6.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df_hard = pd.read_csv(csv_path, quoting=1)  # QUOTE_ALL for multiline fields
    print(f"  hard_50 raw rows: {len(df_hard)}")

    # Keep id, problem, answer; add source and solution as null
    df_hard = df_hard[["id", "problem", "answer"]].copy()
    df_hard["source"] = None
    df_hard["solution"] = None
    df_hard["datatype"] = "hard50"
    # Reorder to match crystal
    df_hard = df_hard[["id", "source", "answer", "problem", "solution", "datatype"]]

    # --- 3. Load oracle_traces_no_match.jsonl ---
    oracle_path = Path("/home/malam/wsl-tunix/imo/openmath_data/oracle_traces_no_match.jsonl")
    oracle_rows = []
    if oracle_path.exists():
        with open(oracle_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                problem = obj.get("problem")
                expected_answer = obj.get("expected_answer")
                oracle_trace = obj.get("oracle_trace")
                if problem is None:
                    continue
                # id = hash of problem (stable string id)
                id_str = hashlib.sha256(problem.encode("utf-8")).hexdigest()[:16]
                oracle_rows.append({
                    "id": id_str,
                    "source": None,
                    "answer": expected_answer,
                    "problem": problem,
                    "solution": oracle_trace if oracle_trace is not None else None,
                    "datatype": "oracle_trace",
                })
        df_oracle = pd.DataFrame(oracle_rows)
        df_oracle = df_oracle[["id", "source", "answer", "problem", "solution", "datatype"]]
        print(f"  oracle_traces raw rows: {len(df_oracle)}")
    else:
        df_oracle = pd.DataFrame(columns=["id", "source", "answer", "problem", "solution", "datatype"])
        print(f"  oracle_traces file not found, skipping: {oracle_path}")

    # --- 4. Combine ---
    combined = pd.concat([df_crystal, df_hard, df_oracle], ignore_index=True)
    print(f"  Combined rows (before answer filter, before dedupe): {len(combined)}")

    # --- 5. Keep only records where answer is in [0, 99999] ---
    def _answer_in_range(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return False
        try:
            s = str(val).strip().replace(",", "")
            n = int(s)
            return 0 <= n <= 99999
        except (ValueError, TypeError):
            return False

    before_range = len(combined)
    combined["_answer_int"] = combined["answer"].apply(
        lambda v: int(str(v).strip().replace(",", "")) if _answer_in_range(v) else pd.NA
    )
    combined = combined[combined["_answer_int"].notna()].copy()
    combined = combined.drop(columns=["_answer_int"])
    print(f"  After filter answer in [0, 99999]: {len(combined)} (removed {before_range - len(combined)})")

    # --- 6. Dedupe on problem (keep first occurrence) ---
    before_dedupe = len(combined)
    combined = combined.drop_duplicates(subset=["problem"], keep="first")
    print(f"  After dedupe on problem: {len(combined)} (removed {before_dedupe - len(combined)})")

    # --- 6b. Remove problems that contain any Chinese character (CJK Unified Ideographs) ---
    _chinese_re = re.compile(r"[\u4e00-\u9fff]")

    def _has_chinese(text):
        if not isinstance(text, str) or pd.isna(text):
            return False
        return _chinese_re.search(text) is not None

    before_chinese = len(combined)
    combined = combined[~combined["problem"].apply(_has_chinese)].copy()
    removed_chinese = before_chinese - len(combined)
    print(f"  After removing problems with Chinese characters: {len(combined)} (removed {removed_chinese})")

    # Normalize answer to string so parquet gets a single type (column was mixed int/str)
    combined["answer"] = combined["answer"].astype(str)

    # --- 7. Save as parquet (keeps types and structure) ---
    out_dir = Path(__file__).resolve().parent / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "combined_math_crystal_hard50.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}")

    # Also save as jsonl for optional use (structure intact per line)
    out_jsonl = out_dir / "combined_math_crystal_hard50.jsonl"
    combined.to_json(out_jsonl, orient="records", lines=True, force_ascii=False)
    print(f"  Saved: {out_jsonl}")


if __name__ == "__main__":
    main()
