"""The three-term decomposition B + D + I, with cluster-consistent inference and robustness
checks.

  S_sim - S_prod = (S_indep - S_prod) + (S_conj - S_indep) + (S_sim - S_conj)
                 =        B           +        D           +        I
  B: the normalisation artefact -- how far conditioning on the nominal rate lifts
     S_prod = S0 * prod(Sa/S0) above the independent baseline
  D: cross-axis survival dependence
  I: the superposition effect, the only term that uses simultaneous data

All inference is a cluster bootstrap over the 40 (suite, task) cells.
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

import json, glob, collections, random, sys
import numpy as np
AX = ("camera","lighting","robot","sensor","actuation","language")
LV = ("L1","L2","L3")
tk = collections.defaultdict(list)   # results are already merged; nothing else to read
def load(ds):
    o = {}
    for d in ds:
        for f in glob.glob(f"{RESULTS}/{d}/*.jsonl"):
            for l in open(f): r = json.loads(l); o[r["rollout_id"]] = r
        for r in tk.get(d, []): o[r["rollout_id"]] = r
    return list(o.values())
PV = lambda s: ([f"predvla_s{s}_eval"]+[f"predvla_s{s}_sh{i}_eval" for i in range(4)],
                [f"predvla_s{s}_clean"]+[f"predvla_s{s}_sh{i}_clean" for i in range(2)])
POL = [("MINERVA",["minerva_eval","minerva_comb"],["minerva_clean"]),("PredVLA",*PV(13)),
       ("SmolVLA",["smolvla_eval"],["smolvla_clean"]),("VLA-JEPA",["vlajepa_eval"],["vlajepa_clean"]),
       ("pi0.5",["pi05_eval"],["pi05_clean"]),("OpenVLA-OFT",["oft_eval"],["oft_clean"]),
       ("UniVLA",["univla_eval"],["univla_clean"])]

def build(name, ed, cd):
    ev, cl = load(ed), load(cd)
    S0 = sum(r["success"] for r in cl if r["axis"]=="clean")/max(1,sum(1 for r in cl if r["axis"]=="clean"))
    ok = {(r["suite"],r["task_id"],r["init_id"]): r["success"] for r in cl if r["axis"]=="clean"}
    unit = collections.defaultdict(dict); init = {}
    for r in ev:
        k = (r["suite"], r["task_id"], r["level"], r["config"])
        unit[k][r["axis"]] = r["success"]; init[k] = (r["suite"], r["task_id"], r["init_id"])
    return dict(S0=S0, unit=dict(unit), init=init, ok=ok, ev=ev)

M = {n: build(n, e, c) for n, e, c in POL}

def terms(d, L, keys):
    """keys: a set of (suite, task, level, config) -> (B, D, I, S_prod, S_indep, S_conj,
    S_sim, N, b, c)"""
    u = d["unit"]; N = len(keys)
    if N == 0: return None
    Sa = {a: sum(u[k][a] for k in keys)/N for a in AX}
    S0 = d["S0"]
    Sp = S0 * np.prod([Sa[a]/S0 for a in AX]) if S0 > 0 else float("nan")
    Si = float(np.prod([Sa[a] for a in AX]))
    a1 = [k for k in keys if all(u[k][a] for a in AX)]
    Sc = len(a1)/N
    Ss = sum(u[k]["combination"] for k in keys)/N
    b = sum(1 for k in a1 if not u[k]["combination"])
    c = sum(1 for k in keys if k not in set(a1) and u[k]["combination"])
    return (100*(Si-Sp), 100*(Sc-Si), 100*(Ss-Sc), 100*Sp, 100*Si, 100*Sc, 100*Ss, N, b, c)

def cellkeys(d, L, clean_only=False):
    u, init, ok = d["unit"], d["init"], d["ok"]
    ks = [k for k in u if k[2]==L and all(a in u[k] for a in AX) and "combination" in u[k]]
    if clean_only: ks = [k for k in ks if ok.get(init[k], False)]
    return ks

def clusterboot(d, L, keys, idx, n=4000, seed=0):
    """Resample the 40 (suite, task) cells; idx 0..2 are B, D, I."""
    rnd = random.Random(seed)
    by = collections.defaultdict(list)
    for k in keys: by[(k[0],k[1])].append(k)
    gk = sorted(by)
    out = []
    for _ in range(n):
        s = [k for g in (rnd.choice(gk) for _ in gk) for k in by[g]]
        t = terms(d, L, s)
        if t: out.append(t[idx])
    out.sort()
    return out[int(.025*len(out))], out[int(.975*len(out))], \
           2*min(sum(1 for x in out if x>0), sum(1 for x in out if x<0))/len(out)

print("## 1. three-term decomposition  S_sim - S_prod = B + D + I")
print(f"{'policy':12s}{'L':3s}{'S_prod':>7s}{'S_ind':>7s}{'S_conj':>7s}{'S_sim':>7s} |"
      f"{'B':>7s}{'D':>7s}{'I':>7s} | {'95%CI on I':>17s}{'p':>7s} | {'residual':>9s}")
print("-"*104)
RES = {}
for name, *_ in POL:
    d = M[name]
    for L in LV:
        ks = cellkeys(d, L)
        t = terms(d, L, ks)
        if not t: continue
        B, D, I, Sp, Si, Sc, Ss, N, b, c = t
        lo, hi, p = clusterboot(d, L, ks, 2)
        RES[(name,L)] = dict(B=B, D=D, I=I, Sp=Sp, Si=Si, Sc=Sc, Ss=Ss, N=N, b=b, c=c,
                             Ilo=lo, Ihi=hi, Ip=p)
        pf = "<0.001" if p < .001 else f"{p:.3f}"
        print(f"{name if L=='L1' else '':12s}{L:3s}{Sp:7.1f}{Si:7.1f}{Sc:7.1f}{Ss:7.1f} |"
              f"{B:+7.1f}{D:+7.1f}{I:+7.1f} | [{lo:+6.1f},{hi:+6.1f}]{pf:>7s} | {Ss-Sp:+8.1f}")
json.dump({f"{k[0]}|{k[1]}": v for k, v in RES.items()}, open(f"{OUT_DIR}/bdi.json","w"))

SU = ("libero_spatial","libero_object","libero_goal","libero_10")
SH = {"libero_spatial":"Spat","libero_object":"Obj","libero_goal":"Goal","libero_10":"Long"}
KEY = [("OpenVLA-OFT","L1"),("SmolVLA","L1"),("VLA-JEPA","L2"),("VLA-JEPA","L3"),
       ("OpenVLA-OFT","L3"),("pi0.5","L2"),("UniVLA","L2")]
print("\n## 2. I per suite and leave-one-suite-out (does the sign hold?)")
print(f"{'policy/L':17s}{'all':>7s} | " + "".join(f"{SH[s]:>7s}" for s in SU) +
      " | " + "".join(f"{'-'+SH[s]:>8s}" for s in SU) + "  sign held")
print("-"*104)
for name, L in KEY:
    d = M[name]; ks = cellkeys(d, L)
    full = terms(d, L, ks)[2]
    per = [terms(d, L, [k for k in ks if k[0]==s]) for s in SU]
    lo1 = [terms(d, L, [k for k in ks if k[0]!=s]) for s in SU]
    sg = all((x[2] > 0) == (full > 0) for x in lo1 if x)
    print(f"{name+'/'+L:17s}{full:+7.1f} | " +
          "".join(f"{(x[2] if x else float('nan')):+7.1f}" for x in per) + " | " +
          "".join(f"{(x[2] if x else float('nan')):+8.1f}" for x in lo1) +
          ("   ✓ 4/4" if sg else "   ✗"))
print("\n## 3. leave-one-task-out (does the sign hold with any one of the 40 removed?)")
for name, L in KEY:
    d = M[name]; ks = cellkeys(d, L); full = terms(d, L, ks)[2]
    tks = sorted({(k[0],k[1]) for k in ks})
    vals = [terms(d, L, [k for k in ks if (k[0],k[1]) != t])[2] for t in tks]
    same = sum(1 for v in vals if (v > 0) == (full > 0))
    print(f"  {name+'/'+L:17s} I={full:+6.1f}  sign held {same}/{len(tks)}  "
          f"range [{min(vals):+.1f}, {max(vals):+.1f}]")

print("\n## 4. negative control for stochastic fluctuation, via the language axis of a")
print("##    policy that cannot accept paraphrases")
print("  That policy looks the instruction up as a task index, so its L1/L2/L3 language rows")
print("  are the same input and the same initial state; only the rollout seed differs. They")
print("  are therefore repeats of an identical condition.")
d = M["MINERVA"]; u = d["unit"]; init = d["init"]; ok = d["ok"]
lang = collections.defaultdict(dict); steps = collections.defaultdict(dict)
for r in d["ev"]:
    if r["axis"] == "language":
        lang[(r["suite"],r["task_id"],r["config"])][r["level"]] = r["success"]
        steps[(r["suite"],r["task_id"],r["config"])][r["level"]] = r["steps"]
ks = [k for k, v in lang.items() if len(v) == 3]
for a, b_ in (("L1","L2"),("L1","L3"),("L2","L3")):
    dis = sum(1 for k in ks if lang[k][a] != lang[k][b_])
    sd = np.median([abs(steps[k][a]-steps[k][b_]) for k in ks])
    print(f"    {a} vs {b_}: outcome mismatch {100*dis/len(ks):.1f}%  ({dis}/{len(ks)})   median step difference {sd:.0f}")
cl_ = {(r["suite"],r["task_id"],r["init_id"]): r["success"] for r in load(["minerva_clean"]) if r["axis"]=="clean"}
kk = [k for k in ks if (k[0],k[1],init[(k[0],k[1],"L1",k[2])][2]) in cl_]
dis = sum(1 for k in kk if cl_[(k[0],k[1],init[(k[0],k[1],'L1',k[2])][2])] != lang[k]["L1"])
print(f"    nominal vs L1: outcome mismatch {100*dis/len(kk):.1f}%  ({dis}/{len(kk)})")
print(f"  -> that is the level of outcome flipping under an identical condition. The |I| values"
      f" that matter are 4-14 pp.")

print("\n## 5. spread of I across training seeds (three released seeds of one policy)")
seeds = {}
for s in (1000, 2000, 3000):
    dd = build(f"ms{s}", [f"mseed_s{s}"], [f"mseed_s{s}"])
    seeds[s] = {L: (terms(dd, L, cellkeys(dd, L)) or [np.nan]*3)[2] for L in LV}
print(f"  {'L':4s}" + "".join(f"{'s'+str(s):>9s}" for s in seeds) + f"{'SD':>8s}{'across pol.':>12s}")
for L in LV:
    v = [seeds[s][L] for s in seeds]
    sp = max(RES[(n,L)]["I"] for n,*_ in POL if (n,L) in RES) - min(RES[(n,L)]["I"] for n,*_ in POL if (n,L) in RES)
    print(f"  {L:4s}" + "".join(f"{x:+9.1f}" for x in v) + f"{np.std(v, ddof=1):8.2f}{sp:10.1f}")

print("\n## 6. decomposition over five axes, excluding language, for the policy that cannot\n##    accept paraphrases")
AX5 = tuple(a for a in AX if a != "language")
def terms5(d, L, keys):
    u = d["unit"]; N = len(keys)
    Sa = {a: sum(u[k][a] for k in keys)/N for a in AX5}
    Si = float(np.prod([Sa[a] for a in AX5]))
    a1 = [k for k in keys if all(u[k][a] for a in AX5)]
    Sc = len(a1)/N; Ss = sum(u[k]["combination"] for k in keys)/N
    return 100*(Sc-Si), 100*(Ss-Sc)
for L in LV:
    ks = cellkeys(M["MINERVA"], L)
    D6, I6 = RES[("MINERVA",L)]["D"], RES[("MINERVA",L)]["I"]
    D5, I5 = terms5(M["MINERVA"], L, ks)
    print(f"  {L}: six axes D={D6:+.1f} I={I6:+.1f}  ->  five axes (no language) D={D5:+.1f} I={I5:+.1f}")
