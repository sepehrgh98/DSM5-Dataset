#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, argparse, random
from collections import Counter

def shuffle_mcq(mcq, rng):
    """
    Shuffle options uniformly at random while preserving:
      - the correct option's text
      - the mapping from each distractor to its why_incorrect rationale
    Assumes:
      - options: list[str] of length 4
      - answer_index: int in [0,3]
      - why_incorrect: list[str] of length 3, in the same order as the distractors as emitted
    """
    options = mcq["options"]
    ai = mcq["answer_index"]
    correct_opt = options[ai]

    # Build (distractor_text, rationale) pairs in the order they appear
    distractor_texts = [opt for i, opt in enumerate(options) if i != ai]
    rationales = mcq.get("why_incorrect", [])
    # be defensive: if lengths mismatch, pad/truncate rationales
    if len(rationales) != len(distractor_texts):
        # pad with generic rationale if missing, or trim extras
        base = rationales[:]
        while len(base) < len(distractor_texts):
            base.append("Incorrect per the provided section.")
        rationales = base[:len(distractor_texts)]

    pairs = list(zip(distractor_texts, rationales))
    rng.shuffle(pairs)

    # choose a new target index for the correct option
    new_ai = rng.randint(0, 3)

    new_options = []
    new_why_incorrect = []
    pair_iter = iter(pairs)
    for idx in range(4):
        if idx == new_ai:
            new_options.append(correct_opt)
        else:
            opt, rat = next(pair_iter)
            new_options.append(opt)
            new_why_incorrect.append(rat)

    mcq["options"] = new_options
    mcq["answer_index"] = new_ai
    mcq["why_incorrect"] = new_why_incorrect
    return mcq

def process_file(inp, outp, seed=1337):
    rng = random.Random(seed)
    pos_counts = Counter()
    total = 0
    fixed = 0

    with open(inp, "r", encoding="utf-8") as fin, open(outp, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            if rec.get("task") != "3A":
                # passthrough other tasks untouched
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            resp = rec.get("response", {})
            items = resp.get("mcq_items", [])
            if not isinstance(items, list):
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            for mcq in items:
                total += 1
                before_ai = mcq.get("answer_index", 1)
                # shuffle
                shuffle_mcq(mcq, rng)
                after_ai = mcq.get("answer_index", 1)
                if after_ai != before_ai:
                    fixed += 1
                pos_counts[after_ai] += 1

            # write back
            rec["response"]["mcq_items"] = items
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # quick summary
    print("\n=== REBALANCE SUMMARY (3A) ===")
    print(f"Total MCQs seen     : {total}")
    print(f"Shuffled (changed)  : {fixed}")
    print(f"Answer index counts : [A:{pos_counts[0]}, B:{pos_counts[1]}, C:{pos_counts[2]}, D:{pos_counts[3]}]")
    print("=== END ===")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to 3A.jsonl")
    ap.add_argument("--output", required=True, help="Path to write rebalanced 3A.jsonl")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    process_file(args.input, args.output, seed=args.seed)

if __name__ == "__main__":
    main()
