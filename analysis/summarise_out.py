#!/usr/bin/env python3
"""Summarise one run directory: success rate per axis and severity level.

    python analysis/summarise_out.py out/pi05_eval_small [out/pi05_clean_small ...]

Reads the rollout records a run wrote and prints the same shape as the paper's Table 2, so a
reduced run can be compared against it directly. De-duplicates by rollout_id, since a resumed
run rewrites nothing but a re-drawn sample may overlap an earlier one.
"""

import json, glob, os, sys, collections

AXES = ["camera", "lighting", "robot", "sensor", "actuation", "language", "combination"]
LEVELS = ["L1", "L2", "L3"]


def load(dirs):
    rows = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.jsonl")):
            for line in open(f):
                try: r = json.loads(line)
                except Exception: continue
                rows[r["rollout_id"]] = r
    return list(rows.values())


def rate(rows):
    return (100 * sum(1 for r in rows if r["success"]) / len(rows), len(rows)) if rows else (None, 0)


def main(dirs):
    rows = load(dirs)
    if not rows:
        sys.exit(f"no rollout records under {', '.join(dirs)}")
    clean = [r for r in rows if r["axis"] == "clean"]
    if clean:
        sr, n = rate(clean)
        print(f"\nnominal: {sr:.1f}%  (n={n})")
    perturbed = [r for r in rows if r["axis"] != "clean"]
    cell_n = 0
    if perturbed:
        print(f"\n{'axis':<12}" + "".join(f"{L:>14}" for L in LEVELS))
        print("-" * (12 + 14 * len(LEVELS)))
        for ax in AXES:
            cells = []
            for L in LEVELS:
                sr, n = rate([r for r in perturbed if r["axis"] == ax and r["level"] == L])
                cells.append("-" if sr is None else f"{sr:.1f}% ({n})")
                if n: cell_n = min(cell_n, n) if cell_n else n
            if any(c != "-" for c in cells):
                print(f"{ax:<12}" + "".join(f"{c:>14}" for c in cells))
    if cell_n:
        print(f"\n{len(rows)} rollouts. A cell of {cell_n} has a standard error of about "
              f"{50 / cell_n ** 0.5:.0f} points at 50%: this reproduces the shape, not the "
              "third digit.\n")
    else:
        print(f"\n{len(rows)} rollouts.\n")


if __name__ == "__main__":
    main(sys.argv[1:] or sys.exit(__doc__))
