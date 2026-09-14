#!/usr/bin/env python3
"""Generate tab:decomp exactly.

I is computed as the integer ratio 100*(c-b)/N. Writing it as 100*(S_sim - S_conj) introduces
float error -- 0.2875 - 0.15 = 0.13749999999999998 -- and the printed value comes out 0.1 off,
which it actually did.
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

import json, glob, collections
import numpy as np
AX=("camera","lighting","robot","sensor","actuation","language")
POL=[("MINERVA$^{\\ast}$",["minerva_eval"],["minerva_clean"]),
     ("PredVLA",["predvla_s13_eval"],["predvla_s13_clean"]),
     ("SmolVLA",["smolvla_eval"],["smolvla_clean"]),
     ("VLA-JEPA",["vlajepa_eval"],["vlajepa_clean"]),
     ("$\\pi_{0.5}$",["pi05_eval"],["pi05_clean"]),
     ("OpenVLA-OFT",["oft_eval"],["oft_clean"]),
     ("UniVLA",["univla_eval"],["univla_clean"])]
def load(ds):
    o={}
    for d in ds:
        for f in glob.glob(f"{RESULTS}/{d}/*.jsonl"):
            for l in open(f): r=json.loads(l); o[r["rollout_id"]]=r
    return o
def s0_matched(ok, init, ks):
    """The nominal success rate on the configurations actually being analysed.

    The per-axis rates Sa are computed over these same configurations, so S0 has to be too;
    taking it over the whole nominal split would divide one population by another. It also
    makes the restriction to nominally successful configurations give S0 = 1 exactly.
    """
    v = [ok[(k[0], k[1], init[k])] for k in ks]
    return sum(v) / len(v)


def counts(u, ks):
    N=len(ks); a1=[k for k in ks if all(u[k][a] for a in AX)]
    s=set(a1)
    b=sum(1 for k in a1 if not u[k]["combination"])
    c=sum(1 for k in ks if k not in s and u[k]["combination"])
    nsim=sum(u[k]["combination"] for k in ks)
    Sa={a: sum(u[k][a] for k in ks)/N for a in AX}
    return N,b,c,len(a1),nsim,Sa
rows=[]
for name,ed,cd in POL:
    ev=load(ed); cl=load(cd)
    ok={(r["suite"],r["task_id"],r["init_id"]):r["success"] for r in cl.values() if r["axis"]=="clean"}
    u=collections.defaultdict(dict); init={}
    for r in ev.values():
        k=(r["suite"],r["task_id"],r["level"],r["config"])
        u[k][r["axis"]]=r["success"]; init[k]=r["init_id"]
    for L in ("L1","L2","L3"):
        ks=[k for k in u if k[2]==L and len(u[k])==7]
        S0=s0_matched(ok,init,ks)
        N,b,c,na1,nsim,Sa=counts(u,ks)
        Sp=S0*np.prod([Sa[a]/S0 for a in AX]); Si=float(np.prod([Sa[a] for a in AX]))
        Sc=na1/N; Ss=nsim/N
        I=100*(c-b)/N                      # exact, as an integer ratio
        B=100*(Si-Sp); D=100*(Sc-Si); res=100*(Ss-Sp)
        bycell=collections.defaultdict(list)
        for k in ks: bycell[(k[0],k[1])].append(k)
        cells=sorted(bycell); rng=np.random.default_rng(0)
        bs=[]
        for _ in range(4000):
            kk=[k for cc in [cells[i] for i in rng.integers(0,len(cells),len(cells))] for k in bycell[cc]]
            n2,b2,c2,_,_,_=counts(u,kk); bs.append(100*(c2-b2)/n2)
        lo,hi=np.percentile(bs,[2.5,97.5])
        p=2*min((np.array(bs)>0).mean(),(np.array(bs)<0).mean())
        if lo==hi==0.0: p=1.0          # degenerate: every resample is exactly 0
        rows.append(dict(pol=name,L=L,res=res,B=B,D=D,I=I,lo=lo,hi=hi,p=p,Sp=100*Sp,N=N,b=b,c=c))
DAG={("SmolVLA","L2"),("$\\pi_{0.5}$","L2")}   # cells whose significance did not survive
                                              # independent re-collection
def f(x,d=1): return f"${x:+.{d}f}$"
print("\\begin{table*}[t]")
print("\\centering")
print("\\caption{Decomposition of the rate-level residual, $S_{\\mathrm{sim}}-S_{\\mathrm{prod}}=B+D+I$,")
print("in percentage points. Bold marks an interval on $I$ excluding zero. Cells whose")
print("$S_{\\mathrm{prod}}$ falls below $4\\%$ are shaded: the decomposition is still exact there, but")
print("every term is compressed against the floor and the estimate carries little information")
print("(Section~\\ref{sec:method:floor}). $^{\\ddagger}$Original significance did not carry over to the")
print("independent replication (Section~\\ref{sec:res:noise}).}")
print("\\label{tab:decomp}")
print("\\small")
print("\\begin{tabular}{llrrrrlr}")
print("\\toprule")
print("Policy & L & Residual & $B$ & $D$ & $I$ & $95\\%$ CI on $I$ & $S_{\\mathrm{prod}}$ \\\\")
print("\\midrule")
prev=None
for r in rows:
    if prev is not None and r["pol"]!=prev: print("\\midrule")
    head=f"\\multirow{{3}}{{*}}{{{r['pol']}}}\n" if r["pol"]!=prev else ""
    prev=r["pol"]
    Itxt=f"{r['I']:+.1f}"
    if r["lo"]>0 or r["hi"]<0: Itxt=f"\\mathbf{{{Itxt}}}"
    Itxt=f"${Itxt}$"
    if (r["pol"],r["L"]) in DAG: Itxt+="$^{\\ddagger}$"
    ci=f"$[{r['lo']:+.1f},{r['hi']:+.1f}]$"
    shade="\\rowcolor{black!7} " if r["Sp"]<4 else ""
    print(f"{head}{shade} & {r['L']} & {f(r['res'])} & {f(r['B'])} & {f(r['D'])} & {Itxt} & {ci} & ${r['Sp']:.1f}$ \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table*}")
