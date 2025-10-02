#!/usr/bin/env python3
import argparse, csv, os
from collections import defaultdict, Counter

def normalize_difficulty(diff):
    """Convert messy difficulty values into clean labels."""
    if not diff:
        return "unknown"
    d = str(diff).lower()
    if "easy" in d:
        return "easy"
    if "moderate" in d:
        return "moderate"
    if "hard" in d:
        return "hard"
    return "unknown"

def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def format_counts(counts: Counter, total: int):
    """Format counts with percentages."""
    parts = []
    for k, v in counts.items():
        pct = (v / total) * 100 if total > 0 else 0
        parts.append(f"{k}: {v} ({pct:.1f}%)")
    return "{ " + ", ".join(parts) + " }"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory with pool CSVs")
    args = ap.parse_args()

    for fname in sorted(os.listdir(args.dir)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(args.dir, fname)
        rows = read_csv(path)

        counts_by_task = defaultdict(Counter)
        for row in rows:
            task = row.get("task", "unknown")
            diff = normalize_difficulty(row.get("difficulty", ""))
            counts_by_task[task][diff] += 1

        print(f"\n=== {fname} === total={len(rows)}")
        for task, counts in counts_by_task.items():
            task_total = sum(counts.values())
            print(f" {task}: {format_counts(counts, task_total)}")

if __name__ == "__main__":
    main()
