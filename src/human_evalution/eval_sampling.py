#!/usr/bin/env python3
"""
Sample Q/As from flattened 3A–3D JSONLs (each line = 1 Q/A).
"""

import argparse, json, os, random
from collections import defaultdict

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def normalize_diff(d):
    d = str(d or "unknown").lower()
    if d in ["e","easy"]: return "easy"
    if d in ["m","moderate","medium"]: return "moderate"
    if d in ["h","hard"]: return "hard"
    return "unknown"

def stratified_sample(data, n_total, seed=42):
    random.seed(seed)
    buckets = defaultdict(list)
    for it in data:
        task = str(it.get("task","")).upper()
        diff = normalize_diff(it.get("difficulty"))
        buckets[(task,diff)].append(it)

    total = sum(len(v) for v in buckets.values())
    alloc = {}
    for k,v in buckets.items():
        share = round(n_total * len(v) / total)
        alloc[k] = min(share, len(v))

    # adjust rounding
    diff_sum = sum(alloc.values())
    while diff_sum < n_total:
        k = max(buckets, key=lambda kk: len(buckets[kk]) - alloc[kk])
        if alloc[k] < len(buckets[k]):
            alloc[k]+=1; diff_sum+=1
        else:
            break

    # sample
    out = []
    for k,need in alloc.items():
        pool = buckets[k]
        random.shuffle(pool)
        out.extend(pool[:need])
    random.shuffle(out)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--files", nargs="+", default=["3a.jsonl","3b.jsonl","3c.jsonl","3d.jsonl"])
    ap.add_argument("--total", type=int, default=400)
    ap.add_argument("--out", type=str, default="eval_pool.jsonl")
    args = ap.parse_args()

    data=[]
    for f in args.files:
        path=os.path.join(args.input_dir,f)
        if os.path.exists(path):
            data.extend(read_jsonl(path))
    print(f"[INFO] loaded {len(data)} Q/As total")

    sampled = stratified_sample(data, args.total, seed=42)
    print(f"[INFO] sampled {len(sampled)} Q/As")

    with open(args.out,"w",encoding="utf-8") as f:
        for it in sampled:
            f.write(json.dumps(it, ensure_ascii=False)+"\n")
    print(f"[OK] wrote {args.out}")

if __name__=="__main__":
    main()
