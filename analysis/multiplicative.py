"""Test whether the simultaneous perturbation is explained by the product of the single ones.

Null hypothesis (the independent model): if failures on each axis occurred independently, then
    SR_comb = SR_clean * Π_a (SR_a / SR_clean)
that is the product of relative survival rates; the gap to the observation is the interaction.

Significance of the gap comes from a paired bootstrap. The combination observation is 400
rollouts (4 suites x 10 tasks x 10 configs) and the prediction is built from 6 axes x 400, so
both are resampled together.
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

import json, glob, os, collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AXES = ["camera", "lighting", "robot", "sensor", "actuation", "language"]
LEVELS = ["L1", "L2", "L3"]


def load(model):
    # de-duplicate by rollout_id: work split across machines and merged back can repeat an id
    seen = {}
    for suf in ("clean", "eval", "comb"):
        for f in sorted(glob.glob(os.path.join(ROOT, "results", f"{model}_{suf}", "*.jsonl"))):
            for l in open(f):
                try: r = json.loads(l)
                except Exception: continue
                seen[r.get("rollout_id", len(seen))] = r
    rows = list(seen.values())
    return rows


def boot(rows, level, n_boot=4000, seed=0):
    """Interval estimate, by bootstrap, of the gap between the independent model and the
    observation."""
    rng = np.random.default_rng(seed)
    clean = np.array([r["success"] for r in rows if r["axis"] == "clean"], float)
    ax = {a: np.array([r["success"] for r in rows
                       if r["axis"] == a and r["level"] == level], float) for a in AXES}
    comb = np.array([r["success"] for r in rows
                     if r["axis"] == "combination" and r["level"] == level], float)
    if len(comb) == 0 or len(clean) == 0 or any(len(v) == 0 for v in ax.values()):
        return None
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        c = clean[rng.integers(0, len(clean), len(clean))].mean()
        if c <= 0: diffs[b] = np.nan; continue
        pred = c
        for a in AXES:
            v = ax[a]
            pred *= v[rng.integers(0, len(v), len(v))].mean() / c
        obs = comb[rng.integers(0, len(comb), len(comb))].mean()
        diffs[b] = obs - pred
    d = diffs[np.isfinite(diffs)]
    # point estimate from the original data
    c0 = clean.mean(); p0 = c0
    for a in AXES: p0 *= ax[a].mean() / c0
    o0 = comb.mean()
    return dict(n_comb=len(comb), n_clean=len(clean),
                pred=p0, obs=o0, diff=o0 - p0,
                ci=(float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
                p_two_sided=float(2 * min((d <= 0).mean(), (d >= 0).mean())))


def main():
    # The combination rows live either in `<model>_comb` or inside `<model>_eval`, depending
    # on when that policy was run. Look for whichever actually contains combination rows.
    models = []
    for d in sorted(glob.glob(os.path.join(ROOT, "results", "*_clean"))):
        m = os.path.basename(d)[: -len("_clean")]
        if any(r["axis"] == "combination" for r in load(m)): models.append(m)
    if not models:
        print("no combination results yet"); return
    print("simultaneous perturbation vs the independent model (paired bootstrap, 4000 draws)\n")
    print(f"{'model':12s}{'level':>6s}{'pred':>9s}{'obs':>9s}{'gap':>8s}{'95% CI':>18s}{'p':>8s}{'n':>6s}")
    for m in models:
        rows = load(m)
        for lv in LEVELS:
            r = boot(rows, lv)
            if r is None:
                print(f"{m:12s}{lv:>6s}{'insufficient data':>28s}"); continue
            print(f"{m:12s}{lv:>6s}{r['pred']*100:8.1f}%{r['obs']*100:8.1f}%"
                  f"{r['diff']*100:+7.1f}"
                  f"  [{r['ci'][0]*100:+5.1f},{r['ci'][1]*100:+5.1f}]"
                  f"{r['p_two_sided']:8.3f}{r['n_comb']:6d}")
    print("\n  A 95% CI on the gap that contains 0 means no detectable departure from the")
    print("  independent model -- the basis for claiming the simultaneous condition is")
    print("  predictable from single-axis measurements.")


if __name__ == "__main__":
    main()
