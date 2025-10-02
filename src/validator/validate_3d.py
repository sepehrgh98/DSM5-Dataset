#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_3d.py
End-to-end validator for DSM-5 Task 3D (Case vignette generation + classification).

Input  : JSONL lines (disorder-level) with nested response.vignette_items
Output : eval_3D_report.jsonl (one line per vignette with metrics + LLM judgment)
Also   : prints a dataset-level summary at the end.
"""

import os, json, re, argparse, random, statistics
from typing import Dict, Any, List, Tuple
from collections import Counter, defaultdict
from tqdm import tqdm
from dotenv import load_dotenv
import textstat

# Embeddings
from sentence_transformers import SentenceTransformer, util
EMB = SentenceTransformer("all-MiniLM-L6-v2")

# OpenAI
from openai import OpenAI
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()

# -------------------
# CONFIG
# -------------------
CFG = {
    "sample_llm_judge": 1.0,          # fraction of items for LLM judge
    "support_sim_threshold": 0.55,    # vignette vs supporting features similarity
    "explanation_min_words": 8,
    "distractor_min_words": 4,
    "difficulty_allowed": {"easy","moderate","hard"},
    "required_distribution": {"easy": 1, "moderate": 1, "hard": 1},
    "max_items": None,
    "seed": 1337
}

SYSTEM_PROMPT = """You are a DSM-5 vignette validator.
You receive:
- DSM reference section text
- One clinical vignette classification case with metadata

Judge ONLY by DSM section text. Check:
1) Correctness: answer matches vignette features
2) Distractors: plausible DSM differential diagnoses, not trivial
3) Explanation: sufficient and DSM-grounded
4) Difficulty label appropriate (easy/moderate/hard)
5) Supporting features truly present in vignette
6) Misleading cues are plausible but not overlapping with supporting features

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

def extract_section_text(prompt: str) -> str:
    """Extract the DSM section text between [BEGIN SECTION] ... [END SECTION]."""
    m1 = SEC_BEGIN_RE.search(prompt)
    m2 = SEC_END_RE.search(prompt)
    if not m1 or not m2 or m2.start() <= m1.end():
        return ""
    return prompt[m1.end(): m2.start()]

def structural_check(item: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errs = []
    req = [
        "uuid","disorder","vignette","options","answer","explanation",
        "option_explanations","supporting_features","misleading_cues",
        "red_flags","difficulty","evidence_span_indices",
        "source_sections","sensitive","flesch_score"
    ]

    for k in req:
        if k not in item:
            errs.append(f"missing_field:{k}")

    if "options" in item and not (3 <= len(item["options"]) <= 5):
        errs.append("options_not_3to5")

    if "answer" in item and "options" in item and item["answer"] not in item["options"]:
        errs.append("answer_not_in_options")

    if "option_explanations" in item and len(item.get("option_explanations", [])) < 2:
        errs.append("option_explanations_too_few")

    if "difficulty" in item:
        d = item["difficulty"]
        if not isinstance(d, dict) or "level" not in d or d["level"] not in CFG["difficulty_allowed"]:
            errs.append("invalid_difficulty")

    return (len(errs) == 0), errs


def sim(a: str, b: str) -> float:
    ea = EMB.encode(a, convert_to_tensor=True)
    eb = EMB.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(ea, eb).item())


def automated_metrics(item: Dict[str, Any]) -> Dict[str, Any]:
    vignette = item.get("vignette","")
    feats = item.get("supporting_features", [])

    sims = []
    for f in feats:
        sims.append(sim(vignette, f))
    best_sim = max(sims) if sims else 0.0
    avg_sim = sum(sims)/len(sims) if sims else 0.0

    expl_len = len(item.get("explanation","").split())
    dist_rats = item.get("option_explanations", [])
    dist_short = sum(1 for d in dist_rats if len(d.get("reason","").split()) < CFG["distractor_min_words"])

    read_score = -1.0
    try:
        read_score = float(textstat.flesch_reading_ease(vignette))
    except Exception:
        pass

    return {
        "support_best_sim": best_sim,
        "support_avg_sim": avg_sim,
        "explanation_len": expl_len,
        "distractor_rationale_too_short": dist_short,
        "readability_flesch": read_score
    }


def automated_rules(metrics: Dict[str, Any]) -> List[str]:
    warnings = []
    if metrics["support_best_sim"] < CFG["support_sim_threshold"]:
        warnings.append(f"low_support_sim:{metrics['support_best_sim']:.2f}")
    if metrics["explanation_len"] < CFG["explanation_min_words"]:
        warnings.append("explanation_too_short")
    if metrics["distractor_rationale_too_short"] > 0:
        warnings.append("short_option_explanations")
    if metrics["readability_flesch"] >= 0 and metrics["readability_flesch"] < 30:
        warnings.append("very_hard_reading")
    return warnings


def llm_validate(item: Dict[str, Any], section_text: str) -> Dict[str, Any]:
    payload = {
        "DSM_text": section_text,
        "Case": item
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
        raw = resp.choices[0].message.content.strip()
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
    ap.add_argument("--input", required=True, help="Path to 3D JSONL")
    ap.add_argument("--output", default="eval_3D_report.jsonl", help="Output JSONL")
    ap.add_argument("--no-llm", action="store_true", help="Disable LLM-as-judge")
    ap.add_argument("--sample", type=float, default=None, help="Sample fraction for LLM judge")
    ap.add_argument("--max-items", type=int, default=None, help="Max items")
    args = ap.parse_args()

    if args.sample is not None:
        CFG["sample_llm_judge"] = max(0.0, min(1.0, args.sample))
    if args.max_items is not None:
        CFG["max_items"] = args.max_items

    random.seed(CFG["seed"])

    # count total
    with open(args.input,"r",encoding="utf-8") as fin:
        total_lines = sum(1 for _ in fin)

    diff_counts = Counter()
    disorder_counts = Counter()
    warnings_count = Counter()
    distribution_violations = []
    total_items = 0
    kept_for_llm = 0

    with open(args.input,"r",encoding="utf-8") as fin, open(args.output,"w",encoding="utf-8") as fout:
        for line in tqdm(fin,total=total_lines,desc="Validating disorders"):
            data = json.loads(line)
            disorder = data.get("disorder","UNKNOWN")
            section_text = extract_section_text(data.get("prompt",""))

            items = data.get("response",{}).get("vignette_items", [])
            if not items:
                print(f"[WARN] {disorder} / {data.get('section')} -> no 3D items found")
                continue

            # --- check per-parent difficulty distribution ---
            parent_diffs = Counter([c.get("difficulty",{}).get("level","unknown") for c in items])
            if any(parent_diffs.get(d,0) != CFG["required_distribution"][d] for d in CFG["required_distribution"]):
                distribution_violations.append({
                    "uuid": data.get("uuid"),
                    "disorder": disorder,
                    "section": data.get("section"),
                    "counts": dict(parent_diffs)
                })

            for case in items:
                total_items += 1
                if CFG["max_items"] and total_items > CFG["max_items"]:
                    break

                struct_ok, struct_errs = structural_check(case)
                try:
                    am = automated_metrics(case)
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
                    llm_j = llm_validate(case, section_text)
                    kept_for_llm += 1

                if struct_ok:
                    diff_counts[case["difficulty"]["level"]] += 1
                    disorder_counts[disorder] += 1

                out = {
                    "uuid": case.get("uuid"),
                    "disorder": disorder,
                    "task": "3D",
                    "structural_ok": struct_ok,
                    "structural_errors": struct_errs,
                    "auto_metrics": am,
                    "rule_warnings": rule_warns,
                    "llm_judgment": llm_j,
                    "difficulty": case.get("difficulty"),
                    "answer": case.get("answer")
                }
                fout.write(json.dumps(out,ensure_ascii=False)+"\n")

            print(f"[INFO] Processed {len(items)} 3D items for {disorder} / {data.get('section')}")

    # summary (console + JSON)
    summary = {
        "total_items": total_items,
        "llm_judged": kept_for_llm,
        "difficulty_distribution": dict(diff_counts),
        "top_disorders": dict(Counter(dict(disorder_counts)).most_common(10)),
        "top_warnings": dict(warnings_count.most_common(10)),
        "distribution_violations": distribution_violations
    }

    print("\n=== DATASET SUMMARY (3D) ===")
    print(f"Total vignette cases   : {summary['total_items']}")
    print(f"LLM-judged (sampled)   : {summary['llm_judged']}")
    print("Difficulty distribution:", summary['difficulty_distribution'])
    print("Top disorders:", summary['top_disorders'])

    if summary["top_warnings"]:
        print("\nTop warnings:")
        for w,c in summary["top_warnings"].items():
            print(f"  {w:35s} x{c}")

    if summary["distribution_violations"]:
        print("\nParent difficulty distribution violations:")
        for v in summary["distribution_violations"]:
            print(f" - {v['disorder']} / {v['section']} -> {v['counts']}")

    print("=== END SUMMARY ===")

    # --- write summary JSON ---
    summary_path = args.output.replace("_report.jsonl", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, indent=2, ensure_ascii=False)
    print(f"\nSummary JSON written to {summary_path}")


if __name__ == "__main__":
    main()
