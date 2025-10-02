#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
summarize_eval.py
Reads eval_3A_report.jsonl (from validate_3a.py) and prints dataset-level stats.
Also saves a CSV for spreadsheet inspection.

Usage:
    python summarize_eval.py --input eval_3A_report.jsonl --csv eval_3A_report.csv
"""

import json
import argparse
import pandas as pd
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="eval_3A_report.jsonl", help="Validator output JSONL file")
    ap.add_argument("--csv", default="eval_3A_report.csv", help="Where to save a flat CSV")
    args = ap.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print("⚠️ Skipping bad line:", e)

    if not rows:
        print("No data found.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False)
    print(f"✅ Saved CSV: {args.csv}")

    # --- Basic Stats
    print("\n=== DATASET SUMMARY ===")
    print(f"Total MCQs: {len(df)}")

    # Final score
    if "final_score" in df.columns:
        print("Mean final_score:", round(df["final_score"].mean(), 2))
        print("Median final_score:", round(df["final_score"].median(), 2))

    # Structural OK rate
    if "structural_ok" in df.columns:
        ok_rate = (df["structural_ok"] == True).mean()
        print("Structural OK rate:", round(ok_rate, 3))

    # Difficulty distribution
    if "difficulty" in df.columns:
        diff_counts = Counter(df["difficulty"].dropna().tolist())
        print("Difficulty distribution:", dict(diff_counts))

    # Evidence similarity & margin
    if "auto_metrics" in df.columns:
        ev_sims = []
        margins = []
        for x in df["auto_metrics"]:
            if isinstance(x, dict):
                ev_sims.append(x.get("evidence_similarity"))
                margins.append(x.get("margin"))
        ev_sims = [v for v in ev_sims if v is not None]
        margins = [v for v in margins if v is not None]
        if ev_sims:
            print("Evidence similarity: mean", round(sum(ev_sims)/len(ev_sims),3),
                  "median", round(sorted(ev_sims)[len(ev_sims)//2],3))
        if margins:
            print("Margin: mean", round(sum(margins)/len(margins),3),
                  "median", round(sorted(margins)[len(margins)//2],3))

    # LLM quality scores
    if "llm_judgment" in df.columns:
        q_scores = []
        valid_flags = []
        for j in df["llm_judgment"]:
            if isinstance(j, dict):
                q_scores.append(j.get("quality_score"))
                valid_flags.append(j.get("is_valid"))
        q_scores = [v for v in q_scores if isinstance(v, (int,float))]
        valid_flags = [v for v in valid_flags if v is not None]
        if q_scores:
            print("LLM quality score: mean", round(sum(q_scores)/len(q_scores),2))
        if valid_flags:
            print("LLM valid rate:", round(sum(valid_flags)/len(valid_flags),3))

    # Disorders with most items
    if "disorder" in df.columns:
        dis_counts = Counter(df["disorder"].dropna().tolist())
        top_disorders = dict(dis_counts.most_common(10))
        print("Top 10 disorders:", top_disorders)

    print("=== END SUMMARY ===")

if __name__ == "__main__":
    main()
