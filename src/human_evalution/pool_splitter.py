#!/usr/bin/env python3
import argparse, json, os, csv, random, re

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def normalize_difficulty(diff):
    """Extract level if dict-like string, else pass through."""
    if not diff:
        return ""
    if isinstance(diff, dict):
        return diff.get("level", "")
    d = str(diff).lower()
    if "easy" in d:
        return "easy"
    if "moderate" in d:
        return "moderate"
    if "hard" in d:
        return "hard"
    return str(diff)

def clean_option(opt: str) -> str:
    """Remove leading A./B./C./D. etc. from option text."""
    if not opt:
        return ""
    return re.sub(r'^[A-D]\.\s*', '', opt.strip())

def stringify(x):
    """Safe stringification for lists/dicts."""
    if isinstance(x, (list, dict)):
        return json.dumps(x, ensure_ascii=False)
    return str(x) if x is not None else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="eval_pool.jsonl")
    ap.add_argument("--out_dir", default="pools")
    ap.add_argument("--num_pools", type=int, required=True)
    ap.add_argument("--pool_size", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    data = read_jsonl(args.pool)
    random.shuffle(data)

    needed = args.num_pools * args.pool_size
    if len(data) < needed:
        raise SystemExit(f"Not enough items: have {len(data)}, need {needed}")

    data = data[:needed]

    for i in range(args.num_pools):
        start = i * args.pool_size
        end = start + args.pool_size
        items = data[start:end]
        outp = os.path.join(args.out_dir, f"pool{i+1}.csv")

        with open(outp, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "uuid", "task", "disorder", "section",
                "difficulty", "question", "options", "correct_answer",
                "explanation", "distractor_rationale",
                "why_correct", "why_incorrect",
                "why_preferred", "why_not_other",
                "option_explanations"
            ])

            for it in items:
                diff = normalize_difficulty(it.get("difficulty", ""))

                # Question fallback
                q = it.get("question") or it.get("symptoms") or it.get("vignette") or ""
                q = str(q).replace("\n", " ").strip()

                # Options cleaning
                opts = it.get("options", [])
                if isinstance(opts, list):
                    opts = [clean_option(str(o)) for o in opts]
                    opts_str = " | ".join(opts)
                else:
                    opts_str = str(opts)

                # Correct answer
                correct_ans = clean_option(str(it.get("correct_answer", "")))

                w.writerow([
                    it.get("uuid", ""),
                    it.get("task", ""),
                    it.get("disorder", ""),
                    it.get("section", ""),
                    diff,
                    q,
                    opts_str,
                    correct_ans,
                    stringify(it.get("explanation", "")),
                    stringify(it.get("distractor_rationale", "")),
                    stringify(it.get("why_correct", "")),
                    stringify(it.get("why_incorrect", "")),
                    stringify(it.get("why_preferred", "")),
                    stringify(it.get("why_not_other", "")),
                    stringify(it.get("option_explanations", ""))
                ])

        print(f"[OK] wrote {outp} ({len(items)} rows)")

if __name__ == "__main__":
    main()
