"""Compute every number the paper reports, in one place, so that none of them is transcribed
by hand."""

import os as _os
# Results live at results/paper/<run>/rollouts.jsonl, already merged across the machines
# they were collected on (see docs/RESULTS_INDEX.md for the merge rule).
ROOT = _os.environ.get("LIBERO_CTRL_ROOT",
       _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
RESULTS = _os.path.join(ROOT, "results", "paper")
MANIFESTS = _os.path.join(ROOT, "manifests", "v0.1")
OUT_DIR = _os.path.join(ROOT, "analysis", "out")
_os.makedirs(OUT_DIR, exist_ok=True)

import json,glob,collections,random,itertools
import numpy as np
AX=("camera","lighting","robot","sensor","actuation","language")
tk=collections.defaultdict(list)   # results are already merged; nothing else to read
def load(dirs):
    rows={}
    for d in dirs:
        for f in glob.glob(f"{RESULTS}/{d}/*.jsonl"):
            for l in open(f): rows[json.loads(l)["rollout_id"]]=json.loads(l)
        for r in tk.get(d,[]): rows[r["rollout_id"]]=r
    return list(rows.values())
MODELS=[("MINERVA","0.54M",["minerva_clean"],["minerva_eval","minerva_comb"],95.75),
        ("SmolVLA","450M",["smolvla_clean"],["smolvla_eval"],87.3),
        ("VLA-JEPA","2.77B",["vlajepa_clean"],["vlajepa_eval"],97.2),
        ("pi05","4.14B",["pi05_clean"],["pi05_eval"],96.9),
        ("OpenVLA-OFT","7.54B",["oft_clean"],["oft_eval"],97.1),
        ("UniVLA","7.54B",["univla_clean"],["univla_eval"],95.2),
        # PredVLA reports its primary seed (s13) as the representative row, as the other
        # multi-seed policy does; the remaining three seeds feed the seed-variance section.
        ("PredVLA","0.68M",["predvla_s13_clean"],["predvla_s13_eval"],77.80)]
SUITES=("libero_spatial","libero_object","libero_goal","libero_10")
def binom2(b,c):
    from math import comb
    n=b+c
    if n==0: return 1.0
    k=min(b,c)
    return min(1.0, 2*sum(comb(n,i) for i in range(k+1))/2**n)
OUT={}
for name,sz,cd,ed,pub in MODELS:
    cl=load(cd); ev=load(ed)
    cn=collections.Counter((r["suite"],r["task_id"]) for r in ev)
    cc=collections.Counter((r["suite"],r["task_id"]) for r in cl)
    full=sorted({k for k in cn if cn[k]>=210 and cc[k]>=50})
    rec=dict(size=sz,pub=pub,cells=len(full),
             clean_per={s:(sum(r["success"] for r in cl if r["suite"]==s),
                           len([r for r in cl if r["suite"]==s])) for s in SUITES},
             clean_all=(sum(r["success"] for r in cl),len(cl)),
             eval_all=(sum(r["success"] for r in ev),len(ev)))
    cells={k:{} for k in full}
    for r in cl:
        k=(r["suite"],r["task_id"])
        if k in cells: cells[k].setdefault("clean",[]).append(r["success"])
    for r in ev:
        k=(r["suite"],r["task_id"])
        if k in cells: cells[k].setdefault((r["axis"],r["level"]),[]).append(r["success"])
    def m(ks,sel):
        t=[x for k in ks for x in cells[k].get(sel,[])]
        return sum(t)/len(t) if t else float("nan")
    # per-axis success rate, over completed cells only
    rec["axis_sr"]={a:{L:100*m(full,(a,L)) for L in ("L1","L2","L3")} for a in AX+("combination",)}
    # product model
    def dev(ks,L):
        c=m(ks,"clean"); p=c
        for a in AX: p*=m(ks,(a,L))/c
        return 100*(m(ks,("combination",L))-p), 100*p, 100*m(ks,("combination",L))
    random.seed(0); rec["prod"]={}
    for L in ("L1","L2","L3"):
        pt,pr,ob=dev(full,L)
        bs=sorted(dev([random.choice(full) for _ in full],L)[0] for _ in range(4000))
        p=2*min(sum(1 for b in bs if b>0),sum(1 for b in bs if b<0))/len(bs)
        rec["prod"][L]=dict(dev=pt,pred=pr,obs=ob,lo=bs[100],hi=bs[3899],p=p)
    # trajectory level (AND), the decomposition, and McNemar
    tbl=collections.defaultdict(dict)
    for r in ev:
        if (r["suite"],r["task_id"]) in cells:
            tbl[(r["suite"],r["task_id"],r["level"],r["config"])][r["axis"]]=r["success"]
    rec["conj"]={}
    for L in ("L1","L2","L3"):
        ks=[k for k in tbl if k[2]==L and all(a in tbl[k] for a in AX) and "combination" in tbl[k]]
        if len(ks)<100: continue
        a1=[k for k in ks if all(tbl[k][a] for a in AX)]
        b=sum(1 for k in a1 if not tbl[k]["combination"])
        c=sum(1 for k in ks if k not in set(a1) and tbl[k]["combination"])
        n=len(ks)
        rec["conj"][L]=dict(n=n,AND=100*len(a1)/n,obs=100*sum(tbl[k]["combination"] for k in ks)/n,
                            b=b,c=c,emg=100*b/len(a1) if a1 else float("nan"),
                            cmp=100*c/(n-len(a1)) if n-len(a1) else float("nan"),
                            p=binom2(b,c))
    OUT[name]=rec
# aggregate measures
pts=[(OUT[n]["prod"][L]["pred"],OUT[n]["prod"][L]["obs"]) for n in OUT for L in OUT[n]["prod"]]
pr=np.array([a for a,_ in pts]); ob=np.array([b for _,b in pts])
AND=[(OUT[n]["conj"][L]["AND"],OUT[n]["conj"][L]["obs"]) for n in OUT for L in OUT[n]["conj"]]
ar=np.array([a for a,_ in AND]); ao=np.array([b for _,b in AND])
sl=np.polyfit(pr,ob-pr,1)[0]
emg=np.array([OUT[n]["conj"][L]["emg"] for n in OUT for L in OUT[n]["conj"]])
cmp_=np.array([OUT[n]["conj"][L]["cmp"] for n in OUT for L in OUT[n]["conj"]])
# emg is nan in cells with no row that survives all six axes individually; drop those
ok=~(np.isnan(emg)|np.isnan(cmp_))
# cells whose prediction is on the floor (<4%) cannot resolve a signed residual, so the
# aggregates are reported both with and without them
res=np.array([p>=4 for p in pr])
def agg(m):
    return dict(n=int(m.sum()), r=float(np.corrcoef(pr[m],ob[m])[0,1]),
                prod_mae=float(np.mean(abs(ob[m]-pr[m]))), prod_bias=float(np.mean(ob[m]-pr[m])),
                conj_mae=float(np.mean(abs(ao[m]-ar[m]))), conj_bias=float(np.mean(ao[m]-ar[m])),
                resid_slope=float(np.polyfit(pr[m],ob[m]-pr[m],1)[0]))
OUT["_agg"]=dict(all=agg(np.ones(len(pr),bool)), resolvable=agg(res),
                 emg_cmp_r=float(np.corrcoef(emg[ok],cmp_[ok])[0,1]), n_emg_cmp=int(ok.sum()))
json.dump(OUT,open(f"{OUT_DIR}/manuscript_numbers.json","w"),indent=1,default=float)
for n in OUT:
    if n=="_agg": continue
    r=OUT[n]
    print(f"{n:<12} cells={r['cells']} clean={100*r['clean_all'][0]/r['clean_all'][1]:.2f}(pub {r['pub']}) "
          f"eval={100*r['eval_all'][0]/r['eval_all'][1]:.1f}")
    for L in ("L1","L2","L3"):
        d=r["prod"].get(L); j=r["conj"].get(L)
        if d: print(f"   {L} product: pred {d['pred']:5.1f} obs {d['obs']:5.1f} dev {d['dev']:+6.1f} "
                    f"[{d['lo']:+.1f},{d['hi']:+.1f}] p={d['p']:.3f}"
                    + (f" | AND {j['AND']:5.1f} b={j['b']} c={j['c']} emg={j['emg']:.1f} cmp={j['cmp']:.1f} pM={j['p']:.2e}" if j else ""))
print("\naggregates:", json.dumps(OUT["_agg"],indent=1))
