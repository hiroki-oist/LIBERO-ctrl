#!/usr/bin/env python3
"""Show what the actuation axis does, by measuring it.

The actuation axis transforms the action on its way to the controller, so it leaves the
observation untouched and cannot be shown in a rendered frame. Instead, one fixed open-loop
command sequence is issued under the nominal condition and under all ten configurations of each
severity level, and the end-effector path that results is recorded from the simulator. The
commands are identical in every run, so all the difference between the paths is the
perturbation. A single configuration is one direction on the severity sphere and its open-loop
displacement is not monotone in the radius -- contacts and joint limits intervene -- so the
deviation is summarised over the ten directions.

  python analysis/fig_actuation.py --paper
"""
import os as _os
ROOT = _os.environ.get("LIBERO_CTRL_ROOT",
       _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
OUT_DIR = _os.path.join(ROOT, "docs", "figs")
_os.makedirs(OUT_DIR, exist_ok=True)

import argparse, json, sys
import numpy as np
sys.path.insert(0, ROOT)

LEVELS = ("L1", "L2", "L3")
COLOUR = {"L1": "#4C72B0", "L2": "#DD8452", "L3": "#C44E52"}


def command_sequence(n=150):
    """A fixed open-loop sequence: descend, translate sideways while yawing, then lift.

    Magnitudes are in the range of the demonstration actions the axis is calibrated against,
    so the perturbation acts on a representative command.
    """
    a = np.zeros((n, 7))
    a[:, 6] = -1.0
    third = n // 3
    a[:third, 1] = -0.55; a[:third, 2] = -0.45
    a[third:2 * third, 0] = 0.55; a[third:2 * third, 5] = 0.18
    a[2 * third:, 1] = 0.35; a[2 * third:, 2] = 0.60; a[2 * third:, 4] = 0.12
    a[2 * third:, 6] = 1.0
    return a


def trace(task, row, cmds, res, init_id):
    """Issue `cmds` under a row's perturbation and return the end-effector path."""
    from libero_ctrl.perturb import PerturbSpec, build, Perturbation
    from libero_ctrl.env import env_reset, warmup
    from libero_ctrl.manifest import rollout_seed
    if row is None:
        p = Perturbation()
        seed = 0
    else:
        p = build(PerturbSpec.from_row(row), suite=row["suite"], shape=(res, res))
        seed = rollout_seed(row["rollout_id"])
        p.reset(seed)
    env_reset(task)
    task.reset_model()
    st = np.array(task.S[init_id])
    warmup(task, st, 10)
    path = []
    for a in cmds:
        task.env.env.done = False
        task.env.step(p.transform_action(a))
        path.append(task.env.env._get_observations()["robot0_eef_pos"].copy())
    return np.asarray(path)


def deviations(task, rows, cmds, res, init_id):
    """Deviation of each configuration's path from the nominal one, in mm, per level."""
    ref = trace(task, None, cmds, res, init_id)
    paths = {L: {c: trace(task, rows[L][c], cmds, res, init_id) for c in sorted(rows[L])}
             for L in LEVELS}
    dev = {L: np.array([np.linalg.norm(paths[L][c] - ref, axis=1) * 1000 for c in paths[L]])
           for L in LEVELS}
    return ref, paths, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=2)
    ap.add_argument("--config", type=int, default=0, help="the configuration drawn in panel (b)")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--fontsize", type=float, default=11.0)
    ap.add_argument("--width", type=float, default=7.16)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if a.paper:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
            "font.size": a.fontsize, "pdf.fonttype": 42, "ps.fonttype": 42,
            "axes.linewidth": 0.6, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

    from libero_ctrl.manifest import iter_rows, env_seed
    from libero_ctrl.env import make_task, close_task

    rows = {}
    for r in iter_rows("eval", suite=a.suite, task_id=a.task, axis="actuation"):
        rows.setdefault(r["level"], {})[r["config"]] = r
    init_id = rows["L1"][a.config]["init_id"]
    cmds = command_sequence(a.steps)

    task = make_task(a.suite, a.task, res=a.res, seed=env_seed())
    try:
        ref, paths, dev = deviations(task, rows, cmds, a.res, init_id)
    finally:
        close_task(task)

    t = np.arange(len(cmds))
    fig, ax = plt.subplots(1, 3, figsize=(a.width, a.width / 3.35))

    # (a) the commanded sequence itself
    for k, lab, c in ((0, "$x$", "#444444"), (1, "$y$", "#8c8c8c"), (2, "$z$", "#c0c0c0")):
        ax[0].plot(t, cmds[:, k], color=c, lw=1.0, label=lab)
    ax[0].set_xlabel("control step"); ax[0].set_ylabel("commanded action")
    ax[0].set_title("(a) command, identical in every run")
    ax[0].legend(frameon=False, ncol=3, handlelength=1.2, columnspacing=0.9,
                 loc="lower center", borderpad=0.1)
    ax[0].set_ylim(-1.15, 1.4)

    # (b) the paths that result: ten directions per level, seen from above
    for L in LEVELS:
        for j, c in enumerate(sorted(paths[L])):
            ax[1].plot(paths[L][c][:, 0] * 100, paths[L][c][:, 1] * 100, color=COLOUR[L],
                       lw=0.7, alpha=0.55, label=L if j == 0 else None, zorder=2)
    ax[1].plot(ref[:, 0] * 100, ref[:, 1] * 100, color="k", lw=1.7, label="nominal", zorder=3)
    ax[1].set_xlabel("$x$ (cm)"); ax[1].set_ylabel("$y$ (cm)")
    ax[1].set_title("(b) end-effector path, top view")
    ax[1].legend(frameon=False, handlelength=1.3, borderpad=0.1, labelspacing=0.25)
    ax[1].set_aspect("equal", adjustable="datalim")

    # (c) deviation from the nominal path, median and range over the ten directions
    for L in LEVELS:
        med = np.median(dev[L], axis=0)
        ax[2].fill_between(t, dev[L].min(0), dev[L].max(0), color=COLOUR[L], alpha=0.16, lw=0)
        ax[2].plot(t, med, color=COLOUR[L], lw=1.4,
                   label=f"{L}  ({med[-1]:.0f} mm)")
    ax[2].set_xlabel("control step"); ax[2].set_ylabel("displacement (mm)")
    ax[2].set_title("(c) deviation from the nominal path")
    ax[2].legend(frameon=False, handlelength=1.3, borderpad=0.1, labelspacing=0.25)
    for h in ax:
        h.spines["top"].set_visible(False); h.spines["right"].set_visible(False)
        h.tick_params(length=2.5, width=0.6)
    fig.tight_layout(pad=0.4, w_pad=1.4)

    name = "actuation_paper" if a.paper else "actuation"
    fig.savefig(f"{a.out}/{name}.pdf")
    fig.savefig(f"{a.out}/{name}.png", dpi=200)
    print(f"written -> {a.out}/{name}.pdf (+ .png)")
    summary = {L: dict(median_final_mm=float(np.median(dev[L][:, -1])),
                       median_max_mm=float(np.median(dev[L].max(1))),
                       range_final_mm=[float(dev[L][:, -1].min()), float(dev[L][:, -1].max())])
               for L in LEVELS}
    with open(f"{a.out}/{name}.json", "w") as f:
        json.dump(summary, f, indent=1)
    for L in LEVELS:
        s_ = summary[L]
        print(f"  {L}: median final {s_['median_final_mm']:6.1f} mm   "
              f"median max {s_['median_max_mm']:6.1f} mm   "
              f"range over 10 directions [{s_['range_final_mm'][0]:.0f}, {s_['range_final_mm'][1]:.0f}]")


if __name__ == "__main__":
    main()
