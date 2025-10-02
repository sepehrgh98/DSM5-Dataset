#!/usr/bin/env python3
import os, csv, argparse
from pathlib import Path

def write_with_rater(src, dst, rid):
    with open(src, "r", encoding="utf-8") as fsrc, open(dst, "w", encoding="utf-8", newline="") as fdst:
        r = csv.DictReader(fsrc)
        # keep all original fields + add rater_id
        fieldnames = r.fieldnames + ["rater_id"] if "rater_id" not in r.fieldnames else r.fieldnames
        w = csv.DictWriter(fdst, fieldnames=fieldnames)
        w.writeheader()
        for row in r:
            row["rater_id"] = rid
            w.writerow(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools_dir", default="pools")
    ap.add_argument("--out_dir", default="rater_packs")
    ap.add_argument("--raters_per_pool", type=int, default=3)
    ap.add_argument("--rater_prefix", default="R")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # discover pool files
    pools = [f for f in os.listdir(args.pools_dir) if f.startswith("pool") and f.endswith(".csv")]
    pools.sort()
    total_needed = len(pools) * args.raters_per_pool

    # generate rater IDs R1..Rn
    rater_ids = [f"{args.rater_prefix}{i+1}" for i in range(total_needed)]

    idx = 0
    for i, poolfname in enumerate(pools, start=1):
        src = os.path.join(args.pools_dir, poolfname)
        for j in range(args.raters_per_pool):
            rid = rater_ids[idx]; idx += 1
            dst = os.path.join(args.out_dir, f"pool{i}_{rid}.csv")
            write_with_rater(src, dst, rid)
            print(f"[OK] wrote {dst}")

if __name__ == "__main__":
    main()
