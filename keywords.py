import argparse
import json
import multiprocessing as mp
import re
from typing import Any, Dict, List, Optional

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ======= Prompt templates =======

SYSTEM_ZH = """你是信息抽取助手。任务：从给定“回答”中抽取关键词。
硬性要求：
- 只输出 JSON 数组，例如：["关键词1","关键词2"]，不要输出任何解释、前后缀或 Markdown。
- 去重，避免同义重复；不要虚词/套话（如：因此、同时、我们、可以、重要的是）。
- 关键词尽量是名词或名词短语（2-8个汉字），保留必要专有名词/缩写。
- 数量：给出约 K 个（允许上下浮动，但不少于 3）。
如果无法抽取，输出 []。
"""

SYSTEM_EN = """You are an information extraction assistant. Task: extract keywords/keyphrases from the given Statement.
Hard requirements:
- Output ONLY a JSON array, e.g. ["keyword 1","keyword 2"]. No extra text, no Markdown.
- Deduplicate; avoid stopwords and filler phrases.
- Prefer noun phrases (1-4 words). Keep proper nouns/acronyms.
- Return about K items (at least 3).
If unsure, output [].
"""

def build_prompt(tokenizer, answer: str, k: int, lang: str) -> str:
    if lang == "en":
        system = SYSTEM_EN.replace("K", str(k))
        user = f"Statement: \n{answer}\n\nExtract keywords from the Statement."
    else:
        system = SYSTEM_ZH.replace("K", str(k))
        user = f"陈述：\n{answer}\n\n请从这段陈述中抽取关键词。"

    try:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except Exception:
        return f"{system}\n\n{user}\n"


# ======= Output parsing =======

def parse_json_array(text: str) -> List[str]:
    """期望模型只输出 JSON 数组。这里做强鲁棒解析，失败则返回空/兜底。"""
    s = text.strip()

    # 抓取第一个 [...] 段
    m = re.search(r"\[[\s\S]*\]", s)
    if m:
        candidate = m.group(0)
        try:
            arr = json.loads(candidate)
            if isinstance(arr, list):
                out: List[str] = []
                for x in arr:
                    if isinstance(x, str):
                        x = x.strip()
                        if x and x not in out:
                            out.append(x)
                return out
        except Exception:
            pass

    # 兜底：按分隔符拆（尽量别让任务中断）
    s = re.sub(r"^```(?:json)?|```$", "", s).strip()
    s = s.strip().strip("[](){}")
    parts = re.split(r"[，,；;、\n]", s)
    out: List[str] = []
    for p in parts:
        p = p.strip().strip('"').strip("'")
        if p and p not in out:
            out.append(p)
    return out[:30]


# ======= IO helpers =======

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def batched(lst: List[Any], bs: int):
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs]


# ======= Main =======

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results.jsonl", help="input jsonl path")
    ap.add_argument("--output", default="results_with_keywords.jsonl", help="output jsonl path")
    ap.add_argument("--model", default="models/Qwen3-1.7B", help="local model path or HF repo id")
    ap.add_argument("--k", type=int, default=8, help="target keyword count (approx)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem_util", type=float, default=0.75)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--include_question", action="store_true",
                    help="whether to include `problem` as context in prompt (recommended)")
    ap.add_argument("--keep_raw", action="store_true",
                    help="whether to store raw model output in keywords_raw")
    args = ap.parse_args()

    rows = read_jsonl(args.input)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
    )

    # 抽取关键词这类任务建议更“稳”，不要太发散
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # 构建 prompts（按你文件的 lang 字段选择中/英模板）
    prompts: List[str] = []
    for r in rows:
        answer = (r.get("answer") or "").strip()
        lang = (r.get("lang") or "zh").lower()
        prompts.append(build_prompt(tokenizer, answer, args.k, lang))

    # 批量推理 + 回填结果
    idx = 0
    for batch_prompts in batched(prompts, args.batch_size):
        outs = llm.generate(batch_prompts, sampling)
        for o in outs:
            raw = o.outputs[0].text
            kws = parse_json_array(raw)
            rows[idx]["keywords"] = kws
            if args.keep_raw:
                rows[idx]["keywords_raw"] = raw
            idx += 1

    write_jsonl(args.output, rows)
    print(f"✅ Done. Wrote {args.output} (rows={len(rows)})")

if __name__ == "__main__":
    # WSL + vLLM 常见需要 spawn，避免你之前的 multiprocessing 报错
    mp.set_start_method("spawn", force=True)
    main()
