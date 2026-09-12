#!/usr/bin/env python3
"""Check reproducibility of the single axes between the original records and the current
pipeline, matched by rollout id."""

import os as _os
# Results live at results/paper/<run>/rollouts.jsonl, already merged across the machines
# they were collected on (see docs/RESULTS_INDEX.md for the merge rule).
ROOT = _os.environ.get("LIBERO_CTRL_ROOT",
       _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
RESULTS = _os.path.join(ROOT, "results", "paper")
MANIFESTS = _os.path.join(ROOT, "manifests", "v0.1")
OUT_DIR = _os.path.join(ROOT, "analysis", "out")
_os.makedirs(OUT_DIR, exist_ok=True)

import json,glob,sys,collections
IDS={json.loads(l)["rollout_id"] for l in open(f"{MANIFESTS}/minerva_single_check.jsonl")}
def ld(pats):
    o={}
    for p in pats:
        for f in (glob.glob(p) if "*" in p else [p]):
            try: fh=open(f)
            except OSError: continue
            for r in map(json.loads,fh):
                if r["rollout_id"] in IDS: o[r["rollout_id"]]=r
    return o
G=lambda d: [f"{RESULTS}/{d}/*.jsonl"]
CASES=[("MINERVA",     G("minerva_eval"),      G("minerva_singlecheck")),
       ("PredVLA",     G("predvla_s13_eval"),  G("predvla_s13_sc")),
       ("OpenVLA-OFT", G("oft_eval"),          G("oft_singlecheck"))]
print(f"{'policy':13s} {'n':>3s} {'outcome':>9s} {'steps':>8s}   by axis (steps)")
for name,osrc,nsrc in CASES:
    o=ld(osrc); n=ld(nsrc); k=sorted(set(o)&set(n))
    if not k: print(f"{name:13s}  --  (incomplete)"); continue
    ds=sum(1 for i in k if o[i]["success"]!=n[i]["success"])
    dt=[i for i in k if o[i]["steps"]!=n[i]["steps"]]
    c=collections.Counter(o[i]["axis"] for i in dt)
    print(f"{name:13s} {len(k):3d} {ds:5d}/{len(k):<3d} {len(dt):5d}/{len(k):<3d}   {dict(sorted(c.items())) if c else 'identical'}")
