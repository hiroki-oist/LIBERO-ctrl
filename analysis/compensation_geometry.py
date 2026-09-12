"""Test whether compensation and emergent failure are explained by the *direction* of the
perturbation vector.

By construction, configs at the same severity level have equal norm on every axis and differ
only in direction, so "that one just happened to be a weaker perturbation" is not available as
a confound. The question of which combinations of directions rescue or kill a trajectory can
therefore be asked directly.

  c (compensated): an initial state killed by at least one factor alone, but successful when
                   all six are applied together
  b (emergent):    an initial state that survives each of the six factors alone, but fails when
                   they are applied together

Each is labelled *within its conditioning set*, and correlated point-biserially against the 25
perturbation parameters and their pairwise products. Features are standardised per level, and
robustness is judged by sign agreement across policies.
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

import json, glob, collections, itertools, math, sys
import numpy as np

AX = ("camera", "lighting", "robot", "sensor", "actuation", "language")
ROOT = ROOT

# ---- the perturbation vector (only combination rows carry all five axes)
FEAT, VEC = [], {}
for l in open(f"{ROOT}/manifests/v0.1/langcomb_all.jsonl"):
    r = json.loads(l)
    if r["axis"] != "combination":
        continue
    p = r["perturb"]
    if not FEAT:
        FEAT = [f"{a}.{k}" for a in ("camera", "lighting", "robot", "sensor", "actuation")
                for k in p[a]]
    VEC[(r["suite"], r["task_id"], r["level"], r["config"])] = np.array(
        [p[f.split(".")[0]][f.split(".")[1]] for f in FEAT], float)

# ---- load results
tk = collections.defaultdict(list)   # results are already merged; nothing else to read

def load(ds):
    o = {}
    for d in ds:
        for f in glob.glob(f"{RESULTS}/{d}/*.jsonl"):
            for l in open(f):
                r = json.loads(l); o[r["rollout_id"]] = r
        for r in tk.get(d, []):
            o[r["rollout_id"]] = r
    return list(o.values())

SPEC = {"MINERVA": ["minerva_eval", "minerva_comb"], "SmolVLA": ["smolvla_eval"],
        "VLA-JEPA": ["vlajepa_eval"], "pi05": ["pi05_eval"],
        "OpenVLA-OFT": ["oft_eval"], "UniVLA": ["univla_eval"],
        "PredVLA": ["predvla_s13_eval"]}
FLOOR = {("MINERVA", "L3"), ("SmolVLA", "L3"), ("UniVLA", "L3")}   # floor cells, prediction <4%

def labels(name):
    """(level, key) -> 'c' | 'b' | None. None means outside the conditioning set."""
    tbl = collections.defaultdict(dict)
    for r in load(SPEC[name]):
        tbl[(r["suite"], r["task_id"], r["level"], r["config"])][r["axis"]] = r["success"]
    out = {}
    for k, v in tbl.items():
        if not all(a in v for a in AX) or "combination" not in v:
            continue
        all_alone = all(v[a] for a in AX)
        out[k] = ("b" if not v["combination"] else "ok") if all_alone else \
                 ("c" if v["combination"] else "ng")
    return out

# ---- feature matrix, standardised per level
def design(keys):
    X = np.array([VEC[k] for k in keys])
    return (X - X.mean(0)) / (X.std(0) + 1e-12)

def pbis(x, y):
    """Point-biserial correlation and a two-sided p, via the Fisher z approximation."""
    if y.sum() < 5 or (1 - y).sum() < 5:
        return float("nan"), float("nan")
    r = np.corrcoef(x, y)[0, 1]
    n = len(y)
    if abs(r) >= 1 or n < 6:
        return r, float("nan")
    z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(n - 3)
    return r, math.erfc(abs(z) / math.sqrt(2))

def scan(term_names, term_fn, tag):
    """Correlate per (policy, level), then aggregate across policies."""
    acc = collections.defaultdict(list)
    for name in SPEC:
        lab = labels(name)
        for L in ("L1", "L2", "L3"):
            if (name, L) in FLOOR:
                continue
            for tgt, cond in (("c", ("c", "ng")), ("b", ("b", "ok"))):
                keys = [k for k, v in lab.items() if k[2] == L and v in cond]
                if len(keys) < 40:
                    continue
                y = np.array([1.0 if lab[k] == tgt else 0.0 for k in keys])
                if y.sum() < 5 or (1 - y).sum() < 5:
                    continue
                Z = design(keys); T = term_fn(Z)
                for j, tn in enumerate(term_names):
                    r, p = pbis(T[:, j], y)
                    if not math.isnan(r):
                        acc[(tgt, tn)].append(r)
    rows = []
    for (tgt, tn), rs in acc.items():
        rs = np.array(rs)
        if len(rs) < 6:
            continue
        # mean correlation across (policy, level), and how many share a sign
        same = max((rs > 0).sum(), (rs < 0).sum())
        # two-sided binomial test on sign agreement
        from math import comb
        n = len(rs)
        pb = min(1.0, 2 * sum(comb(n, i) for i in range(same, n + 1)) / 2 ** n)
        rows.append((tgt, tn, rs.mean(), n, same, pb))
    rows.sort(key=lambda t: t[5])
    print(f"\n=== {tag} ({len(rows)} terms, sorted by the sign-agreement test)")
    print(f"{'':4s} {'term':38s} {'mean r':>7s} {'n':>3s} {'same':>5s} {'p':>8s}")
    for tgt, tn, m, n, same, pb in rows[:12]:
        print(f"  {tgt:2s} {tn:38s} {m:+7.3f} {n:3d} {same:4d}/{n} {pb:8.4f}")
    return rows

single = scan(FEAT, lambda Z: Z, "single parameters")
pairs = list(itertools.combinations(range(len(FEAT)), 2))
pnames = [f"{FEAT[i]} x {FEAT[j]}" for i, j in pairs]
_ = scan(pnames, lambda Z: np.column_stack([Z[:, i] * Z[:, j] for i, j in pairs]), "products of parameter pairs")
