#!/usr/bin/env python3
"""Recompute I from data where both the six single axes and the combination were collected
independently, and compare against the original.

  original    : results/paper/<m>_eval                     (all seven conditions)
  independent : results/paper/<m>_axB (six single axes)
              + results/paper/<m>_repB (combination)
"""

import os as _os
# Results live at results/paper/<run>/rollouts.jsonl, already merged across the machines
# they were collected on (see docs/RESULTS_INDEX.md for the merge rule).
ROOT = _os.environ.get("LIBERO_CTRL_ROOT",
       _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
RESULTS = _os.path.join(ROOT, "results", "paper")
MANIFESTS = _os.path.join(ROOT, "manifests", "v0.1")
OUT_DIR = _os.path.join(ROOT, "analysis", "out")
_os.makedirs(OUT_DIR, exist_ok=True)

import json, glob, collections, sys
import numpy as np
AX = ("camera","lighting","robot","sensor","actuation","language")
LV = ("L1","L2","L3")
def load(dirs):
    o = {}
    for d in dirs:
        for f in glob.glob(f"{RESULTS}/{d}/*.jsonl"):
            for l in open(f):
                r = json.loads(l); o[r["rollout_id"]] = r
    return o
def units(rows):
    u = collections.defaultdict(dict)
    for r in rows.values():
        u[(r["suite"], r["task_id"], r["level"], r["config"])][r["axis"]] = r["success"]
    return u
def terms(u, L, keys):
    N = len(keys)
    a1 = [k for k in keys if all(u[k][a] for a in AX)]
    b = sum(1 for k in a1 if not u[k]["combination"])
    c = sum(1 for k in keys if k not in set(a1) and u[k]["combination"])
    return 100*(c-b)/N, b, c, 100*len(a1)/N
def boot(u0, u1, L, keys, n=4000, seed=0):
    bycell = collections.defaultdict(list)
    for k in keys: bycell[(k[0], k[1])].append(k)
    cells = sorted(bycell)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        cs = [cells[i] for i in rng.integers(0, len(cells), len(cells))]
        ks = [k for c in cs for k in bycell[c]]
        out.append(terms(u1, L, ks)[0] - terms(u0, L, ks)[0])
    return np.percentile(out, [2.5, 97.5])
MODELS = sys.argv[1:] or ["vlajepa", "pi05", "smolvla"]
print(f"{'policy':10s} {'L':3s} {'n':>5s} | {'I(orig)':>8s} {'I(indep)':>11s} {'dI':>7s} {'95%CI':>16s} | {'b->b':>9s} {'c->c':>9s}")
print("-"*95)
for m in MODELS:
    orig = units(load([f"{m}_eval"]))
    ax = load([f"{m}_axB"]); rp = load([f"{m}_repB"])
    if not ax or not rp:
        print(f"{m:10s} (incomplete: axB {len(ax)} / repB {len(rp)})"); continue
    new = units({**ax, **rp})
    for L in LV:
        keys = [k for k in orig if k[2] == L and len(orig[k]) == 7 and len(new.get(k, {})) == 7]
        if not keys: continue
        i0, b0, c0, _ = terms(orig, L, keys)
        i1, b1, c1, _ = terms(new, L, keys)
        lo, hi = boot(orig, new, L, keys)
        print(f"{m:10s} {L:3s} {len(keys):5d} | {i0:+8.1f} {i1:+11.1f} {i1-i0:+7.1f} "
              f"{'['+format(lo,'+.1f')+','+format(hi,'+.1f')+']':>16s} | {b0:4d}→{b1:<4d} {c0:4d}→{c1:<4d}")
