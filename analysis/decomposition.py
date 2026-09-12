"""Decompose the simultaneous-condition gap exactly, into a dependence term and a
superposition term.

  S_indep = prod_a P(survive_a)   P(survive all) if the six survival indicators were independent
  S_conj  = observed P(survive all), determined by the single-axis data alone
  S_sim   = the simultaneous success rate -- the first quantity that uses simultaneous data

  S_sim - S_indep = (S_conj - S_indep) + (S_sim - S_conj)
                     ~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~
                     dependence           superposition
                     (not an interaction) (the only interaction)

The product model used in the paper, S_prod = clean * prod(SR_a / clean), conditions on the
nominal rate and so sits systematically above S_indep; calling that difference "dependence"
flips its sign. S_indep is the reference used here.

Restricted to trajectories that succeed nominally, clean = 100%, so S_indep and S_prod coincide
and the choice of reference stops mattering.
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
from math import comb

AX = ("camera","lighting","robot","sensor","actuation","language")
LV = ("L1","L2","L3")
tk = collections.defaultdict(list)
# results are already merged; nothing else to read
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

def analyse(name, ed, cd, clean_only):
    ev, cl = load(ed), load(cd)
    ok = {(r["suite"], r["task_id"], r["init_id"]): r["success"] for r in cl if r["axis"]=="clean"}
    unit = collections.defaultdict(dict); init = {}
    for r in ev:
        k = (r["suite"], r["task_id"], r["level"], r["config"])
        unit[k][r["axis"]] = r["success"]; init[k] = (r["suite"], r["task_id"], r["init_id"])
    out = {}
    for L in LV:
        ks = [k for k in unit if k[2]==L and all(a in unit[k] for a in AX) and "combination" in unit[k]]
        if clean_only:
            ks = [k for k in ks if ok.get(init[k], False)]
        if len(ks) < 60: continue
        N = len(ks)
        ind = 1.0
        for a in AX: ind *= sum(unit[k][a] for k in ks)/N
        a1 = [k for k in ks if all(unit[k][a] for a in AX)]
        b = sum(1 for k in a1 if not unit[k]["combination"])
        c = sum(1 for k in ks if k not in set(a1) and unit[k]["combination"])
        Sc = 100*len(a1)/N; Ss = 100*sum(unit[k]["combination"] for k in ks)/N
        nf = N - sum(unit[k]["combination"] for k in ks)
        # K = number of axes survived individually
        KH = collections.defaultdict(lambda: [0,0])
        for k in ks:
            K = sum(unit[k][a] for a in AX)
            KH[K][0] += unit[k]["combination"]; KH[K][1] += 1
        out[L] = dict(N=N, indep=100*ind, conj=Sc, sim=Ss, dep=Sc-100*ind, I=Ss-Sc, b=b, c=c,
                      Fh=100*b/nf if nf else float("nan"),
                      pmc=min(1.0, 2*sum(comb(b+c,i) for i in range(min(b,c)+1))/2**(b+c)) if b+c else 1.0,
                      K={k_: (100*v[0]/v[1], v[1]) for k_, v in KH.items()})
    return out

for tag, co in (("all trajectories", False), ("nominally successful trajectories only", True)):
    print(f"\n{'='*94}\n=== {tag} ===")
    print(f"{'policy':12s}{'L':3s}{'N':>5s}{'S_indep':>8s}{'S_conj':>7s}{'S_sim':>7s} | "
          f"{'depend':>7s}{'superp':>7s}{'McN':>7s} | {'F_hidden':>9s}")
    print("-"*94)
    for name, ed, cd in POL:
        r = analyse(name, ed, cd, co)
        for L in LV:
            if L not in r: continue
            d = r[L]
            print(f"{name if L=='L1' else '':12s}{L:3s}{d['N']:5d}{d['indep']:8.1f}{d['conj']:7.1f}"
                  f"{d['sim']:7.1f} | {d['dep']:+7.1f}{d['I']:+7.1f}{d['pmc']:7.3f} | {d['Fh']:8.1f}%")
