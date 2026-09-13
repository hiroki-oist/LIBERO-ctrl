#!/usr/bin/env python3
"""Decompose the superposition effect into its two directions.

I is a net effect. It is the difference of two trajectory-level discrepancies:

  R_emergent    = b / N   every single-axis condition succeeded, the simultaneous one failed
  R_compensated = c / N   at least one single-axis condition failed, the simultaneous one
                          succeeded
  I             = S_sim - S_conj = R_compensated - R_emergent

Reporting I alone hides the case where both directions are large and cancel. This script
reports all three, with cluster-bootstrap intervals on each, and in four settings:

  1. main analysis, six text-conditioned policies
  2. independent replication, for the three policies that sample at inference
  3. nominally successful trajectories only (the solvability control)
  4. the policy that cannot accept paraphrases, as an identity-language control
"""
import os as _os
ROOT = _os.environ.get("LIBERO_CTRL_ROOT",
       _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
RESULTS = _os.path.join(ROOT, "results", "paper")
OUT_DIR = _os.path.join(ROOT, "analysis", "out")
_os.makedirs(OUT_DIR, exist_ok=True)

import collections, glob, json
import numpy as np

AX = ("camera", "lighting", "robot", "sensor", "actuation", "language")
LV = ("L1", "L2", "L3")
NBOOT, SEED = 4000, 0

# Main analysis. Ordered as in docs/POLICIES.md.
MAIN = [("pi_0.5",      ["pi05_eval"],        ["pi05_clean"]),
        ("OpenVLA-OFT", ["oft_eval"],         ["oft_clean"]),
        ("UniVLA",      ["univla_eval"],      ["univla_clean"]),
        ("SmolVLA",     ["smolvla_eval"],     ["smolvla_clean"]),
        ("VLA-JEPA",    ["vlajepa_eval"],     ["vlajepa_clean"]),
        ("PredVLA",     ["predvla_s13_eval"], ["predvla_s13_clean"])]
# Independently re-collected: six single axes in *_axB, the simultaneous condition in *_repB.
REPL = [("pi_0.5",   "pi05"), ("SmolVLA", "smolvla"), ("VLA-JEPA", "vlajepa")]
AUX = [("MINERVA", ["minerva_eval"], ["minerva_clean"])]


def load(dirs):
    o = {}
    for d in dirs:
        for f in glob.glob(f"{RESULTS}/{d}/*.jsonl"):
            for l in open(f):
                r = json.loads(l)
                o[r["rollout_id"]] = r
    return o


def units(rows):
    """(suite, task_id, level, config) -> {axis: success}, plus the shared init_id."""
    u = collections.defaultdict(dict)
    init = {}
    for r in rows.values():
        k = (r["suite"], r["task_id"], r["level"], r["config"])
        u[k][r["axis"]] = r["success"]
        init[k] = r["init_id"]
    return u, init


def counts(u, keys):
    """b, c and N over a list of unit keys.

    `keys` is a multiset under bootstrap resampling, so `survived` has to stay a list: making
    it a set would collapse repeated draws and count b on a different denominator from c.
    """
    survived = [k for k in keys if all(u[k][a] for a in AX)]
    seen = set(survived)
    b = sum(1 for k in survived if not u[k]["combination"])
    c = sum(1 for k in keys if k not in seen and u[k]["combination"])
    nsim = sum(u[k]["combination"] for k in keys)
    return len(keys), b, c, len(survived), nsim


def estimate(u, keys):
    N, b, c, nconj, nsim = counts(u, keys)
    bycell = collections.defaultdict(list)
    for k in keys:
        bycell[(k[0], k[1])].append(k)
    cells = sorted(bycell)
    rng = np.random.default_rng(SEED)
    be, bc, bi = [], [], []
    for _ in range(NBOOT):
        draw = [cells[i] for i in rng.integers(0, len(cells), len(cells))]
        kk = [k for cc in draw for k in bycell[cc]]
        n2, b2, c2, _, _ = counts(u, kk)
        be.append(100 * b2 / n2)
        bc.append(100 * c2 / n2)
        bi.append(100 * (c2 - b2) / n2)

    def ci(v):
        return tuple(np.percentile(v, [2.5, 97.5]))

    return dict(N=N, b=b, c=c,
                Re=100 * b / N, Rc=100 * c / N, I=100 * (c - b) / N,
                Rd=100 * (b + c) / N,
                S_conj=100 * nconj / N, S_sim=100 * nsim / N,
                ci_Re=ci(be), ci_Rc=ci(bc), ci_I=ci(bi),
                # auxiliary: emergent failures as a share of all simultaneous failures
                F_hidden=(100 * b / (N - nsim)) if N > nsim else float("nan"))


def show(title, table, note=None):
    print(f"\n### {title}")
    if note:
        print(f"    {note}")
    print(f"{'policy':12s}{'L':4s}{'N':>5s} | {'R_emerg':>8s}{'95% CI':>16s} | "
          f"{'R_comp':>8s}{'95% CI':>16s} | {'I':>7s}{'95% CI':>16s} | {'R_disc':>7s}")
    for name, L, e in table:
        star = lambda lo, hi: "*" if (lo > 0 or hi < 0) else " "
        print(f"{name:12s}{L:4s}{e['N']:5d} | "
              f"{e['Re']:7.2f}{star(*e['ci_Re'])}[{e['ci_Re'][0]:5.2f},{e['ci_Re'][1]:5.2f}] | "
              f"{e['Rc']:7.2f}{star(*e['ci_Rc'])}[{e['ci_Rc'][0]:5.2f},{e['ci_Rc'][1]:5.2f}] | "
              f"{e['I']:+6.2f}{star(*e['ci_I'])}[{e['ci_I'][0]:+5.2f},{e['ci_I'][1]:+5.2f}] | "
              f"{e['Rd']:6.2f}")


def run(spec, restrict_nominal=False):
    out = []
    for name, ev_dirs, cl_dirs in spec:
        ev = load(ev_dirs)
        u, init = units(ev)
        nominal = None
        if restrict_nominal:
            cl = load(cl_dirs)
            nominal = {(r["suite"], r["task_id"], r["init_id"]): r["success"]
                       for r in cl.values() if r["axis"] == "clean"}
        for L in LV:
            keys = [k for k in u if k[2] == L and len(u[k]) == 7]
            if restrict_nominal:
                keys = [k for k in keys if nominal.get((k[0], k[1], init[k]), False)]
            if not keys:
                continue
            out.append((name, L, estimate(u, keys)))
    return out


def main():
    print("# Emergent and compensated components of the superposition effect")
    print(f"# {NBOOT} cluster-bootstrap resamples over 40 (suite, task) cells. "
          f"* marks an interval excluding zero.")

    main_tbl = run(MAIN)
    show("1. Main analysis - six text-conditioned policies, all trajectories", main_tbl)

    repl_tbl = []
    for name, pfx in REPL:
        ev = load([f"{pfx}_axB", f"{pfx}_repB"])
        u, _ = units(ev)
        for L in LV:
            keys = [k for k in u if k[2] == L and len(u[k]) == 7]
            if keys:
                repl_tbl.append((name, L, estimate(u, keys)))
    show("2. Independent replication - the three policies that sample at inference", repl_tbl,
         "six single axes from *_axB, the simultaneous condition from *_repB")

    nom_tbl = run(MAIN, restrict_nominal=True)
    show("3. Nominally successful trajectories only (solvability control)", nom_tbl)

    aux_tbl = run(AUX)
    show("4. Auxiliary - identity-language control", aux_tbl,
         "this policy resolves the instruction to a task index, so its simultaneous condition "
         "is five perturbations plus an identity language intervention")

    print("\n### Auxiliary: emergent failures as a share of all simultaneous failures")
    print(f"{'policy':12s}" + "".join(f"{L:>10s}" for L in LV))
    for name, *_ in MAIN:
        vals = {L: e["F_hidden"] for n, L, e in main_tbl if n == name}
        print(f"{name:12s}" + "".join(f"{vals.get(L, float('nan')):9.1f}%" for L in LV))

    dump = {"main": [(n, L, e) for n, L, e in main_tbl],
            "replication": [(n, L, e) for n, L, e in repl_tbl],
            "nominal_subset": [(n, L, e) for n, L, e in nom_tbl],
            "identity_language_control": [(n, L, e) for n, L, e in aux_tbl]}
    with open(f"{OUT_DIR}/emergent_compensated.json", "w") as f:
        json.dump(dump, f, indent=1, default=float)
    print(f"\nwritten -> {OUT_DIR}/emergent_compensated.json")


if __name__ == "__main__":
    main()
