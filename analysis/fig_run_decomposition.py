#!/usr/bin/env python3
"""The compound-robustness decomposition of one run's own records.

    python analysis/fig_run_decomposition.py out/pi05_clean_small out/pi05_eval_small \
        --name "pi_0.5" --out out/pi05_decomposition.png

This is Figure 4 of the paper computed from whatever rollouts you have, so a reduced run
(`setup/small.sh`) produces the same picture at its own precision rather than a table of numbers
that can only be compared to the paper's.

Per severity level, over the paired units -- one `(suite, task, level, config)` holds the six
single-axis rollouts and the simultaneous one, all from the same initial state:

  S_indep  the product of the six single-axis rates: what independence predicts
  S_conj   the fraction of units that survived all six axes *separately*
  S_sim    the fraction that survived all six applied *at once*
  D        S_conj - S_indep, cross-axis survival dependence
  I        S_sim - S_conj, the superposition effect

and the disagreement between the conjunction and the simultaneous condition, split into the two
directions that cancel inside I:

  R_e      emergent failures: survived every axis alone, failed when they were combined
  R_c      compensated successes: failed at least one axis alone, survived the combination

A unit is only usable if all seven of its rollouts are present, which is why the reduced
benchmark draws whole units rather than individual rollouts.
"""

import json, glob, os, sys, argparse, collections

AX = ("camera", "lighting", "robot", "sensor", "actuation", "language")
LEVELS = ("L1", "L2", "L3")
GREY, POS, NEG = "#BFC3C7", "#4C72B0", "#C44E52"
R_E, R_C = "#E08A1E", "#0E8F76"
TEXT, MUTED = "#3C4043", "#80868B"


def load(dirs):
    rows = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.jsonl")):
            for line in open(f):
                try: r = json.loads(line)
                except Exception: continue
                rows[r["rollout_id"]] = r
    return list(rows.values())


def decompose(rows):
    """level -> the terms, over the units that are complete."""
    unit = collections.defaultdict(dict)
    for r in rows:
        if r["axis"] == "clean": continue
        unit[(r["suite"], r["task_id"], r["level"], r["config"])][r["axis"]] = r["success"]
    out = {}
    for L in LEVELS:
        keys = [k for k, v in unit.items()
                if k[2] == L and all(a in v for a in AX) and "combination" in v]
        n = len(keys)
        if not n: continue
        u = unit
        rate = {a: sum(u[k][a] for k in keys) / n for a in AX}
        s_indep = 1.0
        for a in AX: s_indep *= rate[a]
        survived = [k for k in keys if all(u[k][a] for a in AX)]
        s_conj = len(survived) / n
        s_sim = sum(u[k]["combination"] for k in keys) / n
        emergent = sum(1 for k in survived if not u[k]["combination"])
        compensated = sum(1 for k in keys if k not in set(survived) and u[k]["combination"])
        out[L] = dict(n=n, s_indep=100 * s_indep, s_conj=100 * s_conj, s_sim=100 * s_sim,
                      d=100 * (s_conj - s_indep), i=100 * (s_sim - s_conj),
                      r_e=100 * emergent / n, r_c=100 * compensated / n)
    return out


def figure(terms, name, out_path, note):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levels = [L for L in LEVELS if L in terms]
    fig, grid = plt.subplots(len(levels), 2, figsize=(9.2, 2.5 * len(levels)),
                             gridspec_kw=dict(width_ratios=[2.2, 1.0]))
    grid = grid.reshape(len(levels), 2)

    for row, L in enumerate(levels):
        t = terms[L]
        a = grid[row][0]
        heights = [t["s_indep"], t["s_conj"], t["s_sim"]]
        top = max(heights + [1.0]) * 1.32
        for x, h in zip((0, 2, 4), heights):
            a.bar(x, h, width=0.62, color=GREY, edgecolor="none", zorder=2)
            a.text(x, h + top * 0.035, f"{h:.1f}", ha="center", va="bottom",
                   fontsize=8.5, color=TEXT)
        for x, lo, hi, val in ((1, t["s_indep"], t["s_conj"], t["d"]),
                               (3, t["s_conj"], t["s_sim"], t["i"])):
            a.bar(x, hi - lo, bottom=lo, width=0.34,
                  color=POS if val >= 0 else NEG, edgecolor="none", zorder=3)
            a.plot([x - 0.5, x + 0.5], [lo, lo], color=MUTED, lw=0.6, ls=":", zorder=1)
            a.text(x, max(lo, hi) + top * 0.035, f"{val:+.1f}", ha="center", va="bottom",
                   fontsize=8.5, color=POS if val >= 0 else NEG)
        a.set_xticks(range(5))
        a.set_xticklabels(["$S_\\mathrm{indep}$", "$D$", "$S_\\mathrm{conj}$",
                           "$I$", "$S_\\mathrm{sim}$"], fontsize=9)
        a.set_ylim(0, top); a.set_ylabel(f"{L}\nsuccess rate (%)", fontsize=9, color=TEXT)
        a.tick_params(labelsize=8, length=2, colors=MUTED)
        a.grid(axis="y", alpha=0.25, lw=0.5); a.set_axisbelow(True)
        for side in ("top", "right"): a.spines[side].set_visible(False)

        b = grid[row][1]
        b.barh(0, t["r_e"], color=R_E, edgecolor="none", height=0.5,
               label="$R_\\mathrm{e}$  emergent failures")
        b.barh(0, t["r_c"], left=t["r_e"], color=R_C, edgecolor="none", height=0.5,
               label="$R_\\mathrm{c}$  compensated successes")
        for val, left in ((t["r_e"], 0), (t["r_c"], t["r_e"])):
            if val > 1.5:
                b.text(left + val / 2, 0, f"{val:.1f}", ha="center", va="center",
                       fontsize=8.5, color="white")
        total = t["r_e"] + t["r_c"]
        b.text(total + 0.6, 0, f"{total:.1f}", ha="left", va="center", fontsize=8.5, color=TEXT)
        b.set_xlim(0, max(total * 1.35, 5)); b.set_ylim(-0.9, 0.9); b.set_yticks([])
        b.set_xlabel("disagreement rate (%)", fontsize=8.5, color=MUTED)
        b.tick_params(labelsize=8, length=2, colors=MUTED)
        for side in ("top", "right", "left"): b.spines[side].set_visible(False)
        if row == 0:
            b.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.55))

    fig.suptitle(f"{name} — {note}", fontsize=10, color=TEXT, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"written: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="run directories (the clean and eval outputs)")
    ap.add_argument("--name", default="this run")
    ap.add_argument("--out", default="analysis/out/run_decomposition.png")
    a = ap.parse_args()

    rows = load(a.dirs)
    if not rows: sys.exit(f"no rollout records under {', '.join(a.dirs)}")
    terms = decompose(rows)
    if not terms:
        sys.exit("no complete paired unit in these records: the decomposition needs the six "
                 "single axes and the simultaneous condition on the same (suite, task, level, "
                 "config). A run made with --sample draws whole units, so this should not "
                 "happen unless the run was interrupted.")

    print(f"\n{'':4s}{'N':>5s}{'S_indep':>9s}{'D':>8s}{'S_conj':>9s}{'I':>8s}{'S_sim':>8s}"
          f"{'R_e':>8s}{'R_c':>8s}")
    print("-" * 71)
    for L, t in terms.items():
        print(f"{L:4s}{t['n']:5d}{t['s_indep']:9.1f}{t['d']:+8.1f}{t['s_conj']:9.1f}"
              f"{t['i']:+8.1f}{t['s_sim']:8.1f}{t['r_e']:8.1f}{t['r_c']:8.1f}")
    n = min(t["n"] for t in terms.values())
    print(f"\nN is the number of paired units behind each level. At N={n} a rate has a standard "
          f"error of about {50/ n**0.5:.0f} points at 50%,\nso read the signs and the ordering "
          "rather than the decimals.\n")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    figure(terms, a.name, a.out, f"{sum(t['n'] for t in terms.values())} paired units")


if __name__ == "__main__":
    main()
