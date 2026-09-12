#!/usr/bin/env python3
"""Recompute I with the combination outcomes replaced by an independent re-collection (repB),
and report |dI|.

The six single axes stay as originally collected -- repB covers the combination axis only -- so
what this measures is how far I moves when only the simultaneous condition is re-collected.
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
TK = {}   # results are already merged; nothing to override with

def load(dirs):
    o = {}
    for d in dirs:
        for f in glob.glob(f"{RESULTS}/{d}/*.jsonl"):
            for l in open(f):
                r = json.loads(l); o[r["rollout_id"]] = r
        if d in TK:
            for l in open(TK[d]):
                r = json.loads(l); o[r["rollout_id"]] = r
    return o

POL = [("VLA-JEPA", ["vlajepa_eval"], "vlajepa_repB"),
       ("SmolVLA",  ["smolvla_eval"], "smolvla_repB"),
       ("pi0.5",    ["pi05_eval"],    "pi05_repB")]

def build(ev, override=None, restrict=None):
    """(suite,task,level,config) -> {axis: success}
    Passing `restrict` keeps only those keys, so that rows repB has not collected are not
    silently counted as agreeing."""
    u = collections.defaultdict(dict)
    for r in ev.values():
        u[(r["suite"], r["task_id"], r["level"], r["config"])][r["axis"]] = r["success"]
    if override:
        for r in override.values():
            k = (r["suite"], r["task_id"], r["level"], r["config"])
            if k in u: u[k]["combination"] = r["success"]
    if restrict is not None:
        u = collections.defaultdict(dict, {k: v for k, v in u.items() if k in restrict})
    return u

def terms(u, L):
    keys = [k for k in u if k[2] == L and len(u[k]) == 7]
    N = len(keys)
    if N == 0: return None
    a1 = [k for k in keys if all(u[k][a] for a in AX)]
    b = sum(1 for k in a1 if not u[k]["combination"])
    c = sum(1 for k in keys if k not in set(a1) and u[k]["combination"])
    return dict(N=N, I=100*(c-b)/N, b=b, c=c,
                Sconj=100*len(a1)/N, Ssim=100*sum(u[k]["combination"] for k in keys)/N)

def boot_ci(u1, u2, L, n=4000, seed=0):
    """Cluster bootstrap over cells, for a CI on dI."""
    keys = [k for k in u1 if k[2] == L and len(u1[k]) == 7 and len(u2.get(k, {})) == 7]
    cells = sorted({(k[0], k[1]) for k in keys})
    bycell = collections.defaultdict(list)
    for k in keys: bycell[(k[0], k[1])].append(k)
    def dI(cs):
        ks = [k for c in cs for k in bycell[c]]
        if not ks: return np.nan
        N = len(ks); out = []
        for u in (u1, u2):
            a1 = [k for k in ks if all(u[k][a] for a in AX)]
            b = sum(1 for k in a1 if not u[k]["combination"])
            c = sum(1 for k in ks if k not in set(a1) and u[k]["combination"])
            out.append(100*(c-b)/N)
        return out[1]-out[0]
    rng = np.random.default_rng(seed)
    bs = [dI([cells[i] for i in rng.integers(0, len(cells), len(cells))]) for _ in range(n)]
    return np.nanpercentile(bs, [2.5, 97.5])

print(f"{'policy':10s} {'L':3s} {'n':>5s} {'mismatch':>9s} {'b/c':>9s} | {'I(orig)':>8s} {'I(repB)':>8s} {'dI':>7s} {'95%CI on dI':>18s}")
print("-"*95)
for name, ed, rd in POL:
    ev = load(ed); rp = load([rd])
    if not rp: print(f"{name:10s} (repB not collected)"); continue
    # Compare only on the keys repB actually has, so uncollected rows are not counted as agreeing.
    got = {(r["suite"], r["task_id"], r["level"], r["config"]) for r in rp.values()}
    u0 = build(ev, restrict=got); u1 = build(ev, rp, restrict=got)
    ntot = len([k for k in u0 if len(u0[k]) == 7])
    print(f"{name:10s} repB has {len(rp)} rows / {ntot} cells comparable")
    for L in LV:
        t0 = terms(u0, L); t1 = terms(u1, L)
        if not t0 or not t1: continue
        ks = [k for k in u0 if k[2] == L and len(u0[k]) == 7]
        dis = sum(1 for k in ks if u0[k]["combination"] != u1[k]["combination"])
        bb = sum(1 for k in ks if u0[k]["combination"] and not u1[k]["combination"])
        cc = dis - bb
        lo, hi = boot_ci(u0, u1, L)
        print(f"{name:10s} {L:3s} {t0['N']:5d} {100*dis/len(ks):6.1f}% {bb:4d}/{cc:<4d} | "
              f"{t0['I']:+8.1f} {t1['I']:+8.1f} {t1['I']-t0['I']:+7.1f} "
              f"{'['+format(lo,'+.1f')+','+format(hi,'+.1f')+']':>18s}")
