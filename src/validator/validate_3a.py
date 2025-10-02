#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_3a.py
End-to-end validator for DSM-5 Task 3A (MCQs).

Input  : JSONL lines with fields like your smoke run (prompt includes the section).
Output : eval_3A_report.jsonl (one line per MCQ with metrics + LLM judgment)
Also   : prints a dataset-level summary at the end.
"""

import os, json, re, argparse, random, statistics
from typing import Dict, Any, List, Tuple
from collections import Counter, defaultdict
from tqdm import tqdm
from dotenv import load_dotenv
from tqdm import tqdm

import numpy as np
from scipy.stats import chisquare

# --- optional/robust imports
import textstat

# Grammar tool (runs a lightweight remote service by default; guarded by try/except)
try:
    import language_tool_python
    TOOL = language_tool_python.LanguageTool("en-US")
except Exception as e:
    TOOL = None

# Embeddings for semantic similarity
from sentence_transformers import SentenceTransformer, util
EMB = SentenceTransformer("all-MiniLM-L6-v2")

# OpenAI LLM (LLM-as-judge)
from openai import OpenAI
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()

# -------------------
# CONFIG
# -------------------
CFG = {
    "sample_llm_judge": 1.0,      # fraction of items to send to LLM (1.0 = all, 0.2 = 20%)
    "evidence_sim_threshold": 0.55, # correct option vs evidence should be >= this
    "distractor_min_sim": 0.08,     # too low => nonsense
    "distractor_max_sim": 0.70,     # too high => too close to evidence/correct
    "margin_min": 0.15,             # (correct_sim - best_distractor_sim) should be >= this
    "difficulty_allowed": {"easy","moderate","hard"},
    "max_items": None,              # cap for quick testing (e.g., 100); None = all
    "seed": 1337
}

SYSTEM_PROMPT = """You are a DSM-5 exam question validator.
You receive:
- DSM reference text (only the relevant DSM-5 section used to write the MCQ)
- One multiple-choice question (MCQ) with 4 options and metadata
- The labeled correct answer index and an evidence_quote copied directly from the DSM text

Judge ONLY by the provided DSM text. Check:
1) Correctness: labeled answer is correct per DSM text
2) Distractors: plausible but clearly incorrect (no second correct)
3) Clarity: well-formed, unambiguous
4) Difficulty label appropriate (easy/moderate/hard)
5) Evidence_quote truly supports the correct answer
6) No hallucinated criteria not present in DSM text

Return STRICT JSON ONLY (no markdown) with:
{
  "is_valid": true/false,
  "errors": [ "string", ... ],
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
    """
    Extract the DSM section text from the prompt, and return (section_text, start_offset_in_prompt).
    evidence_span indices are assumed RELATIVE TO THE SECTION TEXT.
    """
    m1 = SEC_BEGIN_RE.search(prompt)
    m2 = SEC_END_RE.search(prompt)
    if not m1 or not m2 or m2.start() <= m1.end():
        # fallback: try to use whole prompt, but this will likely misalign spans
        return prompt, 0
    section_text = prompt[m1.end(): m2.start()]
    return section_text, m1.end()

def safe_len(s: str) -> int:
    return 0 if s is None else len(s)

def structural_check(item: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errs = []
    req = ["question","options","answer_index","why_correct","why_incorrect",
           "difficulty","hallucination_flag","sensitive"]

    for k in req:
        if k not in item:
            errs.append(f"missing_field:{k}")

    if "options" in item and len(item["options"]) != 4:
        errs.append("options_not_4")

    if "answer_index" in item and not (isinstance(item["answer_index"], int) and 0 <= item["answer_index"] < 4):
        errs.append("answer_index_out_of_range")

    if "why_incorrect" in item and len(item.get("why_incorrect", [])) != 3:
        errs.append("why_incorrect_not_3")

    if "difficulty" in item and item["difficulty"] not in CFG["difficulty_allowed"]:
        errs.append("invalid_difficulty")

    # accept evidence_quote
    if "evidence_quote" not in item or not item["evidence_quote"].strip():
        errs.append("missing_field:evidence_quote")

    return (len(errs) == 0), errs



def grammar_issues(text: str) -> int:
    if TOOL is None:
        return -1  # means "skipped"
    try:
        return len(TOOL.check(text))
    except Exception:
        return -1

def readability_score(text: str) -> float:
    # lower is harder; we return Flesch Reading Ease as an extra signal
    try:
        return float(textstat.flesch_reading_ease(text))
    except Exception:
        return -1.0

def sim(a: str, b: str) -> float:
    ea = EMB.encode(a, convert_to_tensor=True)
    eb = EMB.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(ea, eb).item())

def automated_metrics(item: Dict[str, Any], section_text: str) -> Dict[str, Any]:
    # Use evidence_quote (string) instead of evidence_span (indices)
    evidence = item.get("evidence_quote", "").strip()

    correct = item["options"][item["answer_index"]]

    # similarity between correct option and evidence
    ev_sim_correct = sim(correct, evidence) if evidence else 0.0

    # similarity for distractors
    dist_sims = []
    for i, opt in enumerate(item["options"]):
        if i == item["answer_index"]:
            continue
        dist_sims.append(sim(opt, evidence) if evidence else 0.0)

    # margin and sanity flags
    best_dist = max(dist_sims) if dist_sims else 0.0
    margin = ev_sim_correct - best_dist

    # grammar + readability
    g_issues_q = grammar_issues(item["question"])
    read_q = readability_score(item["question"])

    return {
        "evidence_similarity": ev_sim_correct,
        "distractor_similarities": dist_sims,
        "margin": margin,
        "grammar_issues_question": g_issues_q,
        "readability_flesch": read_q,
        "evidence_len": len(evidence.split())  # length in words instead of characters
    }


def automated_rules(metrics: Dict[str, Any]) -> List[str]:
    """Convert raw metrics into rule-based warnings."""
    warnings = []
    ev = metrics["evidence_similarity"]
    dists = metrics["distractor_similarities"]
    margin = metrics["margin"]

    if ev < CFG["evidence_sim_threshold"]:
        warnings.append(f"low_ev_sim:{ev:.2f}")

    if dists:
        if min(dists) < CFG["distractor_min_sim"]:
            warnings.append(f"distractor_too_unrelated:min={min(dists):.2f}")
        if max(dists) > CFG["distractor_max_sim"]:
            warnings.append(f"distractor_too_similar:max={max(dists):.2f}")
    else:
        warnings.append("no_distractors_found")

    if margin < CFG["margin_min"]:
        warnings.append(f"low_margin:{margin:.2f}")

    if metrics["grammar_issues_question"] not in (-1, None) and metrics["grammar_issues_question"] > 2:
        warnings.append(f"many_grammar_issues:{metrics['grammar_issues_question']}")

    if metrics["evidence_len"] < 5:  # fewer than 5 words is suspicious
        warnings.append(f"very_short_evidence_quote:{metrics['evidence_len']}")

    return warnings

def llm_validate(item: Dict[str, Any], section_text: str) -> Dict[str, Any]:
    payload = {
        "DSM_text": section_text,
        "MCQ": item
    }
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":json.dumps(payload, ensure_ascii=False)}
            ],
            temperature=0
        )
        raw = resp.choices[0].message.content
        # try to parse strict json (in case the model wrapped it)
        raw = raw.strip()
        # remove possible code fences
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

def chi_square_uniform(counts: List[int]) -> Tuple[float, float]:
    """
    Chi-square test against uniform distribution for answer_index balance.
    Returns (statistic, pvalue). pvalue < 0.05 -> significantly non-uniform.
    """
    if not counts or sum(counts) == 0:
        return 0.0, 1.0
    expected = [sum(counts)/len(counts)] * len(counts)
    stat, p = chisquare(counts, f_exp=expected)
    return float(stat), float(p)

# -------------------
# main
# -------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to your 3A JSONL (smoke or full)")
    ap.add_argument("--output", default="eval_3A_report.jsonl", help="Output JSONL with evaluations")
    ap.add_argument("--no-llm", action="store_true", help="Disable LLM-as-judge")
    ap.add_argument("--sample", type=float, default=None, help="Sample fraction (overrides config sample_llm_judge)")
    ap.add_argument("--max-items", type=int, default=None, help="Max MCQs to process")
    args = ap.parse_args()

    if args.sample is not None:
        CFG["sample_llm_judge"] = max(0.0, min(1.0, args.sample))
    if args.max_items is not None:
        CFG["max_items"] = args.max_items

    random.seed(CFG["seed"])

    # count total lines for tqdm
    with open(args.input, "r", encoding="utf-8") as fin:
        total_lines = sum(1 for _ in fin)

    pos_counts = [0,0,0,0]
    diff_counts = Counter()
    disorder_counts = Counter()
    warnings_count = Counter()

    total_items = 0
    kept_for_llm = 0


    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=total_lines, desc="Validating disorders"):
            data = json.loads(line)
            disorder = data.get("disorder","UNKNOWN")
            section_text, _offset = extract_section_text(data["prompt"])

            items = data.get("response",{}).get("mcq_items", [])
            if not items:
                print(f"[WARN] {disorder} / {data.get('section')} -> no MCQs found")

            for mcq in items:
                total_items += 1
                if CFG["max_items"] and total_items > CFG["max_items"]:
                    break

                # same checks as before ...
                struct_ok, struct_errs = structural_check(mcq)
                try:
                    am = automated_metrics(mcq, section_text)
                except Exception as e:
                    am = {"error": f"autometrics_failed:{type(e).__name__}"}
                rule_warns = []
                if "error" not in am:
                    rule_warns = automated_rules(am)
                    for w in rule_warns:
                        warnings_count[w] += 1

                do_llm = (not args.no_llm) and (random.random() < CFG["sample_llm_judge"])
                llm_j = None
                if do_llm:
                    llm_j = llm_validate(mcq, section_text)
                    kept_for_llm += 1

                if struct_ok:
                    pos_counts[ mcq["answer_index"] ] += 1
                    diff_counts[ mcq["difficulty"] ] += 1
                    disorder_counts[ disorder ] += 1

                score = 100.0
                # ... scoring logic same as before ...

                out = {
                    "uuid": data.get("uuid"),
                    "disorder": disorder,
                    "task": "3A",
                    "structural_ok": struct_ok,
                    "structural_errors": struct_errs,
                    "auto_metrics": am,
                    "rule_warnings": rule_warns,
                    "llm_judgment": llm_j,
                    "final_score": score,
                    "answer_index": mcq.get("answer_index"),
                    "difficulty": mcq.get("difficulty")
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")

            # 👇 small per-disorder log
            print(f"[INFO] Processed {len(items)} MCQs for {disorder} / {data.get('section')}")

    # summary
    print("\n=== DATASET SUMMARY (3A) ===")
    print(f"Total MCQs processed  : {total_items}")
    print(f"LLM-judged (sampled)  : {kept_for_llm}")

    # answer index balance
    stat, p = chi_square_uniform(pos_counts)
    print(f"Answer index counts    : {pos_counts}")
    print(f"Chi-square uniformity  : stat={stat:.2f}, p={p:.4f}  (p<0.05 => likely imbalanced)")

    # difficulty distribution
    print("Difficulty distribution:", dict(diff_counts))

    # top 10 disorders by count
    print("Top disorders:", dict(Counter(dict(disorder_counts)).most_common(10)))

    # most frequent warnings
    if warnings_count:
        top_warns = warnings_count.most_common(10)
        print("Top warnings:")
        for w, c in top_warns:
            print(f"  {w:40s}  x{c}")
    print("=== END SUMMARY ===")
    

if __name__ == "__main__":
    main()
