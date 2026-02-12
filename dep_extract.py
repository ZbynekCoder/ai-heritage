#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dependency-based extraction for:
- NOUN (including PROPN)
- ADJ
- Nominalised verbs (verbs used in nominal argument positions)

Language decoupling:
- English: spaCy en_core_web_sm + 3-rule filtering (NO word blacklist)
- Chinese: spaCy zh_core_web_sm (no extra filtering)

Input/Output JSONL structure is unchanged.

Expected per-row fields:
- answer: str
- lang: "zh" or "en" (optional; defaults to --default_lang)

Outputs appended per row:
- nouns: List[str]
- adjectives: List[str]
- nominalized_verbs: List[str]
- keywords: List[str]  # union of the above (dedup, no truncation)
"""

import argparse
import json
from typing import Any, Dict, List, Tuple


# ---------------- IO helpers ----------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        x = (x or "").strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# ---------------- spaCy loader ----------------

def load_spacy(lang: str):
    """
    Load spaCy pipeline with a dependency parser.
    lang: 'en' or 'zh'

    English tweak:
    - Use token_match to keep hyphenated words like "non-physical" as a single token.
      This avoids splitting into "non", "-", "physical".
    """
    try:
        import spacy  # type: ignore
    except Exception as e:
        raise RuntimeError("spaCy is not installed. Install with: pip install spacy") from e

    model = "en_core_web_sm" if lang == "en" else "zh_core_web_sm"
    try:
        nlp = spacy.load(model)
    except Exception as e:
        raise RuntimeError(
            f"spaCy model '{model}' not found.\n"
            f"Install it with:\n  python -m spacy download {model}\n"
        ) from e

    if "parser" not in nlp.pipe_names:
        raise RuntimeError(
            f"spaCy pipeline '{model}' has no dependency parser. "
            f"Please use a model that includes 'parser'."
        )

    # --- English-only: keep hyphenated words as one token via token_match (no Tokenizer(...) rebuild) ---
    if lang == "en":
        import re
        # Only match alphabetic hyphenated words, so "non-physical" stays whole
        # but numeric ranges like "3-5" won't be forced into one token.
        nlp.tokenizer.token_match = re.compile(r"(?i)^[a-z]+(?:-[a-z]+)+$").match

    return nlp


# ---------------- Shared extraction rules ----------------

# UD relations that indicate nominal argument positions
NOMINAL_DEPRELS = {
    "nsubj", "nsubj:pass",
    "obj", "iobj",
    "obl", "obl:arg", "obl:agent", "obl:tmod",
    "pobj",  # legacy in some pipelines
    "nmod", "appos",
}
CONJ_DEPREL = "conj"

# ---------------- English-only filtering (3-rule combo) ----------------

EN_DROP_POS = {"DET", "PRON", "NUM", "AUX", "ADP", "SCONJ", "CCONJ", "PART"}
EN_DROP_DEPS = {"det", "predet"}
EN_KEEP_ADJ_DEPS = {"amod", "acomp"}  # typical adjective functions

EN_STOP_TOKENS = {"few", "little", "many", "much",
                  "other", "such", "that", "it",
                  "thing",
                  "not",
                  "be", "was", "have", "has", "had", "may",
                  "'s"}


def en_is_content_token(tok) -> bool:
    """
    3-rule combo (no word blacklist):
    (1) drop function POS
    (2) drop det/predet deps
    (3) drop numeric-like tokens (like_num)
    """
    if tok.is_space or tok.is_punct:
        return False
    if tok.pos_ in EN_DROP_POS:
        return False
    if tok.dep_ in EN_DROP_DEPS:
        return False
    if getattr(tok, "like_num", False):
        return False
    if tok.lower_ in EN_STOP_TOKENS:
        return False
    return True


def extract_en(doc) -> Tuple[List[str], List[str]]:
    nouns: List[str] = []
    adjs: List[str] = []

    for tok in doc:
        if not en_is_content_token(tok):
            continue

        if tok.pos_ in ("NOUN", "PROPN"):
            nouns.append(tok.text)

        if tok.pos_ == "ADJ":
            # rule-based: only keep typical adjective functions
            if tok.dep_ in EN_KEEP_ADJ_DEPS:
                adjs.append(tok.text)

    return uniq_keep_order(nouns), uniq_keep_order(adjs)


# ---------------- Chinese extraction (no extra filtering) ----------------

def extract_zh(doc) -> List[str]:
    """
    Keep it simple (no additional filtering), same as your original logic.
    """
    keywords: List[str] = []

    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue

        if tok.pos_ in ("NOUN", "PROPN", "ADJ"):
            keywords.append(tok.text)

    return keywords


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results.jsonl", help="input jsonl path")
    ap.add_argument("--output", default="results_with_keywords.jsonl", help="output jsonl path")
    ap.add_argument("--prefer_lang_field", action="store_true",
                    help="If set, use row['lang'] when present; else use --default_lang.")
    ap.add_argument("--default_lang", default="zh", choices=["zh", "en"], help="default language")
    args = ap.parse_args()

    rows = read_jsonl(args.input)

    nlp_en = None
    nlp_zh = None

    def get_lang(row: Dict[str, Any]) -> str:
        if args.prefer_lang_field:
            v = (row.get("lang") or "").lower().strip()
            if v in ("zh", "en"):
                return v
        return args.default_lang

    for r in rows:
        text = (r.get("answer") or "").strip()
        lang = get_lang(r)

        if not text:
            r["nouns"] = []
            r["adjectives"] = []
            r["keywords"] = []
            continue

        if lang == "en":
            if nlp_en is None:
                nlp_en = load_spacy("en")
            doc = nlp_en(text)
            nouns, adjs = extract_en(doc)
            r["nouns"] = nouns
            r["adjectives"] = adjs
            r["keywords"] = uniq_keep_order(nouns + adjs)
        else:
            if nlp_zh is None:
                nlp_zh = load_spacy("zh")
            doc = nlp_zh(text)
            keywords = extract_zh(doc)
            r["keywords"] = keywords

    write_jsonl(args.output, rows)
    print(f"✅ Done. Wrote {args.output} (rows={len(rows)})")


if __name__ == "__main__":
    main()
