"""MINERVA 公開3シード（abl_l1 s1000/2000/3000）で逸脱指標の学習シード分散を測る。
   各シード 10,400 本（clean 2,000 + eval 8,400）。本文 §Results と同一の推定式を使う。"""

import os as _os
# ★結果は results/paper/<run名>/rollouts.jsonl に統合済み（fuji と taketomi の両方を、
#   taketomi 優先でマージ）。旧リポジトリの /tmp/tkpull による上書きはもう不要。
ROOT = _os.environ.get("LIBERO_CTRL_ROOT",
       _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
RESULTS = _os.path.join(ROOT, "results", "paper")
MANIFESTS = _os.path.join(ROOT, "manifests", "v0.1")
OUT_DIR = _os.path.join(ROOT, "analysis", "out")
_os.makedirs(OUT_DIR, exist_ok=True)

import json,glob,collections,random
import numpy as np
AX=("camera","lighting","robot","sensor","actuation","language")
def load(seed):
    rows={}
    for f in glob.glob(f"{RESULTS}/mseed_s{seed}/*.jsonl"):
        try:
            for l in open(f): r=json.loads(l); rows[r["rollout_id"]]=r
        except FileNotFoundError: pass
    return list(rows.values())
def binom2(b,c):
    from math import comb
    n=b+c
    if n==0: return 1.0
    k=min(b,c); return min(1.0,2*sum(comb(n,i) for i in range(k+1))/2**n)
OUT={}
for seed in (1000,2000,3000):
    rows=load(seed)
    cl=[r for r in rows if r["axis"]=="clean"]; ev=[r for r in rows if r["axis"]!="clean"]
    cells=collections.defaultdict(dict)
    keys=sorted({(r["suite"],r["task_id"]) for r in cl})
    C={k:{} for k in keys}
    for r in cl: C[(r["suite"],r["task_id"])].setdefault("clean",[]).append(r["success"])
    for r in ev:
        k=(r["suite"],r["task_id"])
        if k in C: C[k].setdefault((r["axis"],r["level"]),[]).append(r["success"])
    tbl=collections.defaultdict(dict)
    for r in ev: tbl[(r["suite"],r["task_id"],r["level"],r["config"])][r["axis"]]=r["success"]
    def m(ks,sel):
        t=[x for k in ks for x in C[k].get(sel,[])]
        return sum(t)/len(t) if t else float("nan")
    def dev(ks,L):
        c=m(ks,"clean"); p=c
        for a in AX: p*=m(ks,(a,L))/c
        return 100*(m(ks,("combination",L))-p),100*p,100*m(ks,("combination",L))
    random.seed(0)
    rec=dict(n=len(rows),clean=100*np.mean([r["success"] for r in cl]),
             eval=100*np.mean([r["success"] for r in ev]),prod={},conj={})
    for L in ("L1","L2","L3"):
        pt,pr,ob=dev(keys,L)
        bs=sorted(dev([random.choice(keys) for _ in keys],L)[0] for _ in range(4000))
        rec["prod"][L]=dict(dev=pt,pred=pr,obs=ob,lo=bs[100],hi=bs[3899],
            p=2*min(sum(1 for b in bs if b>0),sum(1 for b in bs if b<0))/len(bs))
        ks=[k for k in tbl if k[2]==L and all(a in tbl[k] for a in AX) and "combination" in tbl[k]]
        a1=[k for k in ks if all(tbl[k][a] for a in AX)]
        b=sum(1 for k in a1 if not tbl[k]["combination"])
        c=sum(1 for k in ks if k not in set(a1) and tbl[k]["combination"])
        rec["conj"][L]=dict(n=len(ks),AND=100*len(a1)/len(ks),
            obs=100*sum(tbl[k]["combination"] for k in ks)/len(ks),b=b,c=c,p=binom2(b,c))
    OUT[seed]=rec
    print(f"seed {seed}: {rec['n']}本 clean {rec['clean']:.2f}% eval {rec['eval']:.1f}%")
    for L in ("L1","L2","L3"):
        d=rec["prod"][L]; j=rec["conj"][L]
        print(f"   {L} 積: pred {d['pred']:5.1f} obs {d['obs']:5.1f} dev {d['dev']:+6.1f} [{d['lo']:+.1f},{d['hi']:+.1f}] p={d['p']:.3f}"
              f" | AND {j['AND']:5.1f} b={j['b']} c={j['c']} pM={j['p']:.3f}")
print("\n=== シード間のばらつき ===")
print(f"{'量':<26}"+"".join(f"{f's{s}':>9}" for s in (1000,2000,3000))+f"{'mean':>9}{'SD':>7}{'range':>8}")
def row(lab,f):
    v=[f(OUT[s]) for s in (1000,2000,3000)]
    print(f"{lab:<26}"+"".join(f"{x:>9.2f}" for x in v)+f"{np.mean(v):>9.2f}{np.std(v,ddof=1):>7.2f}{max(v)-min(v):>8.2f}")
row("clean SR (%)",lambda r:r["clean"])
for L in ("L1","L2","L3"):
    row(f"逸脱 {L} (pp)",lambda r,L=L:r["prod"][L]["dev"])
for L in ("L1","L2","L3"):
    row(f"観測-AND {L} (pp)",lambda r,L=L:r["conj"][L]["obs"]-r["conj"][L]["AND"])
json.dump(OUT,open(f"{OUT_DIR}/seed_variance.json","w"),indent=1,default=float)
