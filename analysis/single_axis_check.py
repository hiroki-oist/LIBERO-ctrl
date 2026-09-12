#!/usr/bin/env python3
"""単一5軸の原本と現行パイプラインの再現性を rollout_id で突き合わせる。"""

import os as _os
# ★結果は results/paper/<run名>/rollouts.jsonl に統合済み（fuji と taketomi の両方を、
#   taketomi 優先でマージ）。旧リポジトリの /tmp/tkpull による上書きはもう不要。
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
print(f"{'policy':13s} {'n':>3s} {'結果違い':>8s} {'step違い':>8s}   軸別(step)")
for name,osrc,nsrc in CASES:
    o=ld(osrc); n=ld(nsrc); k=sorted(set(o)&set(n))
    if not k: print(f"{name:13s}  --  (未完了)"); continue
    ds=sum(1 for i in k if o[i]["success"]!=n[i]["success"])
    dt=[i for i in k if o[i]["steps"]!=n[i]["steps"]]
    c=collections.Counter(o[i]["axis"] for i in dt)
    print(f"{name:13s} {len(k):3d} {ds:5d}/{len(k):<3d} {len(dt):5d}/{len(k):<3d}   {dict(sorted(c.items())) if c else '一致'}")
