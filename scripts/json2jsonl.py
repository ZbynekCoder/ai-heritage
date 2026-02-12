# -*- coding: utf-8 -*-
"""
Convert nested results.json -> flat JSONL where each line is one answer.

Input format (example):
[
  {
    "problem": "...",
    "models": {
      "gpt-4o": [{"answer": "...", "attempt": 1}, ...],
      "o3":     [{"answer": "...", "attempt": 1}, ...]
    }
  },
  ...
]

Output JSONL format (one answer per line):
{"problem_id":"p000001","problem":"...","model":"gpt-4o","attempt":1,"answer":"...","lang":"zh"}
"""

import json
import re
from pathlib import Path


# ======================
# Hard-coded file paths
# ======================
INPUT_PATH = "../results/raw_results.json"
OUTPUT_PATH = "../results/results.jsonl"

# Whether to keep empty/blank answers
KEEP_EMPTY_ANSWERS = False


def detect_lang(text: str) -> str:
    """
    Very simple language detection:
    - zh: contains any CJK characters
    - en: otherwise
    """
    if not text:
        return "unknown"
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def main():
    in_path = Path(INPUT_PATH)
    out_path = Path(OUTPUT_PATH)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path.resolve()}")

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array (list).")

    n_written = 0
    n_skipped = 0

    with out_path.open("w", encoding="utf-8") as out:
        for idx, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                n_skipped += 1
                continue

            problem = item.get("problem", "")
            models = item.get("models", {}) or {}
            problem_id = f"p{idx:06d}"

            if not isinstance(models, dict):
                n_skipped += 1
                continue

            for model_name, attempts in models.items():
                if attempts is None or not isinstance(attempts, list):
                    continue

                for a in attempts:
                    if not isinstance(a, dict):
                        continue

                    answer = a.get("answer", "")
                    attempt_no = a.get("attempt", None)

                    if (answer is None or str(answer).strip() == "") and not KEEP_EMPTY_ANSWERS:
                        continue

                    record = {
                        "problem_id": problem_id,
                        "problem": problem,
                        "model": model_name,
                        "attempt": attempt_no,
                        "answer": answer,
                        "lang": detect_lang(problem),
                    }

                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_written += 1

    print("Conversion finished.")
    print(f"Input : {in_path.resolve()}")
    print(f"Output: {out_path.resolve()}")
    print(f"Written lines: {n_written}")
    print(f"Skipped items: {n_skipped}")


if __name__ == "__main__":
    main()
