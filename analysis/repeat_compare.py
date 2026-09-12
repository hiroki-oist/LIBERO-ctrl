#!/usr/bin/env python3
"""同一条件の反復と原本／反復同士を rollout_id で突き合わせ、方策の確率性を測る。"""

import os as _os
# ★結果は results/paper/<run名>/rollouts.jsonl に統合済み（fuji と taketomi の両方を、
#   taketomi 優先でマージ）。旧リポジトリの /tmp/tkpull による上書きはもう不要。
ROOT = _os.environ.get("LIBERO_CTRL_ROOT",
       _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
RESULTS = _os.path.join(ROOT, "results", "paper")
MANIFESTS = _os.path.join(ROOT, "manifests", "v0.1")
OUT_DIR = _os.path.join(ROOT, "analysis", "out")
_os.makedirs(OUT_DIR, exist_ok=True)

import json,glob,math
from math import comb
IDS={json.loads(l)["rollout_id"] for l in open(f"{MANIFESTS}/repeat_pilot.jsonl")}
def load(pats):
    o={}
    for p in pats:
        for f in (glob.glob(p) if "*" in p else [p]):
            try: fh=open(f)
            except OSError: continue
            for r in map(json.loads,fh):
                if r["rollout_id"] in IDS: o[r["rollout_id"]]=r
    return o
L=lambda d:[f"{RESULTS}/{d}/*.jsonl"]
M=[("MINERVA",  L("minerva_eval")+L("minerva_comb"), "minerva"),
   ("VLA-JEPA", L("vlajepa_eval"),                                            "vlajepa"),
   ("SmolVLA",  L("smolvla_eval"),                                            "smolvla"),
   ("pi05",     L("pi05_eval"),                   "pi05"),
   ("OpenVLA-OFT",L("oft_eval"),      "oft"),
   ("UniVLA",   L("univla_eval"),     "univla"),
   ("PredVLA",  L("predvla_s13_eval"),"predvla_s13")]
def wil(k,n,z=1.96):
    if n==0: return (0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (100*(c-h),100*(c+h))
def pair(a,bb):
    k=sorted(set(a)&set(bb))
    b=sum(1 for i in k if a[i]["success"] and not bb[i]["success"])
    c=sum(1 for i in k if not a[i]["success"] and bb[i]["success"])
    return k,b,c
def show(tag,rows):
    print(f"\n### {tag}")
    print(f"{'policy':13s} {'n':>3s} {'不一致':>7s} {'95%CI':>14s} {'b/c':>7s} {'S1':>6s} {'S2':>6s} {'I伝播1SD':>9s}")
    tb=tc=0
    for name,k,b,c,a,bb in rows:
        n=len(k); d=b+c; lo,hi=wil(d,n); tb+=b; tc+=c
        s1=100*sum(a[i]['success'] for i in k)/n; s2=100*sum(bb[i]['success'] for i in k)/n
        print(f"{name:13s} {n:3d} {100*d/n:6.1f}% [{lo:4.1f},{hi:5.1f}] {b:3d}/{c:<3d} {s1:5.1f}% {s2:5.1f}% {100*math.sqrt(2*d/n*400)/400:8.2f}pt")
    n=tb+tc
    pv=min(1.0, 2*sum(comb(n,i) for i in range(0,min(tb,tc)+1))/2**n) if n else 1.0
    print(f"{'合計':13s} {'':3s} {'':7s} {'':14s} {tb:3d}/{tc:<3d}   対称性の二項検定 p={pv:.4f}")
# 原本 vs rep1
r1=[]
for name,src,pfx in M:
    a=load(src); bb=load([f"{RESULTS}/{pfx}_rep1/*.jsonl"])
    if not bb: continue
    k,b,c=pair(a,bb)
    if k: r1.append((name,k,b,c,a,bb))
show("原本 vs rep1（機械・時期が違う。系統差を含む上界）",r1)
# rep1 vs rep2
r2=[]
for name,src,pfx in M:
    a=load([f"{RESULTS}/{pfx}_rep1/*.jsonl"]); bb=load([f"{RESULTS}/{pfx}_rep2/*.jsonl"])
    if not a or not bb: continue
    k,b,c=pair(a,bb)
    if k: r2.append((name,k,b,c,a,bb))
show("rep1 vs rep2（同一機・同一時期。純粋なサンプリング雑音）",r2)
