#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_3c.py
End-to-end validator for DSM-5 Task 3C (Contrastive Diagnostic Explanations).

Input  : JSONL lines with fields like your smoke run (prompt includes the section).
Output : eval_3C_report.jsonl (one line per contrastive case with metrics + LLM judgment)
Also   : prints a dataset-level summary at the end.
"""

import os, json, re, argparse, random
from typing import Dict, Any, List, Tuple
from collections import Counter
from tqdm import tqdm
from dotenv import load_dotenv

import numpy as np
from sentence_transformers import SentenceTransformer, util

# --- readability + grammar
import textstat
try:
    import language_tool_python
    TOOL = language_tool_python.LanguageTool("en-US")
except Exception:
    TOOL = None

# --- embeddings
EMB = SentenceTransformer("all-MiniLM-L6-v2")

# --- OpenAI
from openai import OpenAI
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()

# -------------------
# CONFIG
# -------------------
CFG = {
    "sample_llm_judge": 0.3,          # fraction to LLM (adjust as needed)
    "difficulty_allowed": {"easy","moderate","hard"},
    "max_items": None,
    "seed": 1337
}

SYSTEM_PROMPT_3C = """You are a DSM-5 diagnostic explanation validator.
You receive:
- DSM reference text (the section used to generate the contrastive vignette)
- One item with: symptoms, two candidate diagnoses (A, B), the labeled correct answer, explanations.

Judge ONLY by the DSM text. Check:
1) Correctness: labeled diagnosis is supported by DSM text
2) Contrastiveness: why_preferred vs why_not_other are consistent and not contradictory
3) Clarity: vignette is clear, concise, unambiguous
4) Difficulty label appropriate (easy/moderate/hard)
5) Supporting_features are short verbatim spans from vignette
6) No hallucinated criteria not present in DSM text

Return STRICT JSON ONLY:
{
  "is_valid": true/false,
  "errors": ["string", ...],
  "quality_score": 1-5,
  "difficulty_ok": true/false,
  "reasoning": "one short paragraph"
}
"""

# -------------------
# helpers
# -------------------

SEC_BEGIN_RE = re.compile(r"\[BEGIN SECTION.*?\]\s*", re.I | re.S)
SEC_END_RE   = re.compile(r"\s*\[END SECTION\]", re.I | re.S)

def extract_section_text(prompt: str) -> Tuple[str, int]:
    m1 = SEC_BEGIN_RE.search(prompt)
    m2 = SEC_END_RE.search(prompt)
    if not m1 or not m2 or m2.start() <= m1.end():
        return prompt, 0
    section_text = prompt[m1.end(): m2.start()]
    return section_text, m1.end()

def grammar_issues(text: str) -> int:
    if TOOL is None:
        return -1
    try:
        return len(TOOL.check(text))
    except Exception:
        return -1

def readability_score(text: str) -> float:
    try:
        return float(textstat.flesch_reading_ease(text))
    except Exception:
        return -1.0

def sim(a: str, b: str) -> float:
    ea = EMB.encode(a, convert_to_tensor=True)
    eb = EMB.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(ea, eb).item())

# -------------------
# validators
# -------------------

def structural_check_3c(item: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errs = []
    req = [
        "symptoms","choice_a","choice_b","answer","correct_diagnosis",
        "difficulty","why_preferred","why_not_other",
        "supporting_features","evidence_spans","hallucination_flag",
        "source_section","disorder_context"
    ]
    for k in req:
        if k not in item:
            errs.append(f"missing_field:{k}")

    if item.get("answer") not in ["A","B"]:
        errs.append("invalid_answer")

    if item.get("difficulty") not in CFG["difficulty_allowed"]:
        errs.append("invalid_difficulty")

    if not isinstance(item.get("supporting_features", []), list) or not item["supporting_features"]:
        errs.append("missing_supporting_features")

    if not isinstance(item.get("evidence_spans", []), list) or not item["evidence_spans"]:
        errs.append("missing_evidence_spans")

    return (len(errs) == 0), errs

def automated_metrics_3c(item: Dict[str, Any], section_text: str) -> Dict[str, Any]:
    symptoms = item.get("symptoms","")
    a, b = item.get("choice_a",""), item.get("choice_b","")
    correct = item.get("correct_diagnosis","")

    sim_sym_a = sim(symptoms, a)
    sim_sym_b = sim(symptoms, b)
    sim_sym_correct = sim(symptoms, correct)
    margin = abs(sim_sym_a - sim_sym_b)

    return {
        "sim_symptoms_a": sim_sym_a,
        "sim_symptoms_b": sim_sym_b,
        "sim_symptoms_correct": sim_sym_correct,
        "margin_a_vs_b": margin,
        "grammar_issues_symptoms": grammar_issues(symptoms),
        "readability_flesch": readability_score(symptoms),
    }

def automated_rules_3c(metrics: Dict[str, Any]) -> List[str]:
    warns = []
    if metrics["margin_a_vs_b"] < 0.05:
        warns.append("very_low_margin")  # choices nearly indistinguishable
    if metrics["grammar_issues_symptoms"] not in (-1,None) and metrics["grammar_issues_symptoms"] > 3:
        warns.append("many_grammar_issues")
    if metrics["readability_flesch"] != -1 and metrics["readability_flesch"] < 40:
        warns.append("very_low_readability")
    return warns

def llm_validate_3c(item: Dict[str, Any], section_text: str) -> Dict[str, Any]:
    payload = {
        "DSM_text": section_text,
        "contrastive_item": item
    }
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"system","content":SYSTEM_PROMPT_3C},
                {"role":"user","content":json.dumps(payload, ensure_ascii=False)}
            ],
            temperature=0
        )
        raw = resp.choices[0].message.content
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.S)
        return json.loads(raw)
    except Exception as e:
        return {
            "is_valid": False,
            "errors": [f"llm_error:{type(e).__name__}"],
            "quality_score": 1,
            "difficulty_ok": False,
            "reasoning": "LLM call failed or invalid JSON."
        }

# -------------------
# main
# -------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to your 3C JSONL")
    ap.add_argument("--output", default="eval_3C_report.jsonl", help="Output JSONL with evaluations")
    ap.add_argument("--no-llm", action="store_true", help="Disable LLM-as-judge")
    ap.add_argument("--sample", type=float, default=None, help="Sample fraction for LLM judge")
    ap.add_argument("--max-items", type=int, default=None, help="Max items to process")
    args = ap.parse_args()

    if args.sample is not None:
        CFG["sample_llm_judge"] = max(0.0, min(1.0, args.sample))
    if args.max_items is not None:
        CFG["max_items"] = args.max_items

    random.seed(CFG["seed"])

    # count total lines
    with open(args.input, "r", encoding="utf-8") as fin:
        total_lines = sum(1 for _ in fin)

    diff_counts = Counter()
    disorder_counts = Counter()
    warnings_count = Counter()

    total_items = 0
    kept_for_llm = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=total_lines, desc="Validating 3C items"):
            data = json.loads(line)
            disorder = data.get("disorder","UNKNOWN")
            section_text, _ = extract_section_text(data["prompt"])

            items = data.get("response",{}).get("contrastive_dx_items", [])
            if not items:
                print(f"[WARN] {disorder} / {data.get('section')} -> no contrastive items found")

            for item in items:
                total_items += 1
                if CFG["max_items"] and total_items > CFG["max_items"]:
                    break

                struct_ok, struct_errs = structural_check_3c(item)
                try:
                    am = automated_metrics_3c(item, section_text)
                except Exception as e:
                    am = {"error": f"autometrics_failed:{type(e).__name__}"}
                rule_warns = []
                if "error" not in am:
                    rule_warns = automated_rules_3c(am)
                    for w in rule_warns:
                        warnings_count[w] += 1

                do_llm = (not args.no_llm) and (random.random() < CFG["sample_llm_judge"])
                llm_j = None
                if do_llm:
                    llm_j = llm_validate_3c(item, section_text)
                    kept_for_llm += 1

                if struct_ok:
                    diff_counts[item["difficulty"]] += 1
                    disorder_counts[disorder] += 1

                out = {
                    "uuid": data.get("uuid"),
                    "disorder": disorder,
                    "task": "3C",
                    "structural_ok": struct_ok,
                    "structural_errors": struct_errs,
                    "auto_metrics": am,
                    "rule_warnings": rule_warns,
                    "llm_judgment": llm_j,
                    "difficulty": item.get("difficulty"),
                    "answer": item.get("answer"),
                    "correct_diagnosis": item.get("correct_diagnosis")
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")

            print(f"[INFO] Processed {len(items)} 3C items for {disorder} / {data.get('section')}")

    # summary
    print("\n=== DATASET SUMMARY (3C) ===")
    print(f"Total items processed   : {total_items}")
    print(f"LLM-judged (sampled)    : {kept_for_llm}")
    print("Difficulty distribution :", dict(diff_counts))
    print("Top disorders:", dict(Counter(dict(disorder_counts)).most_common(10)))
    if warnings_count:
        top_warns = warnings_count.most_common(10)
        print("Top warnings:")
        for w, c in top_warns:
            print(f"  {w:30s} x{c}")
    print("=== END SUMMARY ===")

if __name__ == "__main__":
    main()
