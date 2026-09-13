#!/usr/bin/env python3
"""Render what each axis at each severity level actually looks like.

One task, one initial state, one configuration index: the agentview camera is rendered under
the nominal condition and under L1, L2 and L3 of every axis that changes the image. Because
severity is a radius in a normalised parameter space, a camera L2 and a lighting L2 are the
same distance from nominal, and the grid can be read across as well as down.

  python analysis/fig_perturbation_grid.py --suite libero_object --task 2 --config 0
"""
import os as _os
ROOT = _os.environ.get("LIBERO_CTRL_ROOT",
       _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
OUT_DIR = _os.path.join(ROOT, "docs", "figs")
_os.makedirs(OUT_DIR, exist_ok=True)

import argparse, json, sys
import numpy as np
sys.path.insert(0, ROOT)

VISUAL = ("camera", "lighting", "robot", "sensor", "combination")
LEVELS = ("L1", "L2", "L3")


def render(task, row, res):
    """Apply a manifest row's perturbation and return the agentview image, upright."""
    from libero_ctrl.perturb import PerturbSpec, build
    from libero_ctrl.env import env_reset, warmup
    from libero_ctrl.manifest import rollout_seed
    p = build(PerturbSpec.from_row(row), suite=row["suite"], shape=(res, res))
    p.reset(rollout_seed(row["rollout_id"]))
    env_reset(task)
    task.reset_model()
    p.apply_model(task)
    st = p.transform_init_state(task, np.array(task.S[row["init_id"]]))
    warmup(task, st, row["num_steps_wait"])
    img = task.env.env._get_observations()["agentview_image"]
    return p.transform_obs(img)[::-1]          # flip once, for a human-viewable figure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=2)
    ap.add_argument("--config", type=int, default=0)
    ap.add_argument("--res", type=int, default=256)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from libero_ctrl.manifest import iter_rows, env_seed
    from libero_ctrl.env import make_task, close_task

    per = {}
    for r in iter_rows("eval", suite=a.suite, task_id=a.task):
        if r["config"] == a.config:
            per[(r["axis"], r["level"])] = r
    init_id = per[("camera", "L1")]["init_id"]
    clean = next(r for r in iter_rows("clean", suite=a.suite, task_id=a.task)
                 if r["init_id"] == init_id)

    task = make_task(a.suite, a.task, res=a.res, seed=env_seed())
    try:
        base = render(task, clean, a.res)
        grid = {("clean", ax): base for ax in VISUAL}
        for ax in VISUAL:
            for L in LEVELS:
                grid[(L, ax)] = render(task, per[(ax, L)], a.res)
    finally:
        close_task(task)

    cols = ("clean",) + LEVELS
    fig, axes = plt.subplots(len(VISUAL), len(cols),
                             figsize=(2.05 * len(cols), 2.05 * len(VISUAL)))
    for i, ax_name in enumerate(VISUAL):
        for j, L in enumerate(cols):
            h = axes[i, j]
            h.imshow(grid[(L, ax_name)]); h.set_xticks([]); h.set_yticks([])
            for sp in h.spines.values(): sp.set_linewidth(0.4)
            if i == 0:
                h.set_title("nominal" if L == "clean" else f"{L}  (r={ {'L1':2,'L2':4,'L3':8}[L] })",
                            fontsize=10)
            if j == 0:
                h.set_ylabel(ax_name, fontsize=10)
    fig.suptitle(f"{a.suite} task {a.task}, initial state {init_id}, configuration {a.config}: "
                 f"“{clean['language']}”", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_DIR}/perturbation_grid.{ext}", dpi=150, bbox_inches="tight")
    print(f"written -> {OUT_DIR}/perturbation_grid.png")

    meta = {"suite": a.suite, "task_id": a.task, "config": a.config, "init_id": init_id,
            "language": {"nominal": clean["language"],
                         **{L: per[("language", L)]["language"] for L in LEVELS}},
            "actuation": {L: per[("actuation", L)]["perturb"] for L in LEVELS},
            "perturb": {ax: {L: per[(ax, L)]["perturb"] for L in LEVELS} for ax in VISUAL}}
    with open(f"{OUT_DIR}/perturbation_grid.json", "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f"written -> {OUT_DIR}/perturbation_grid.json")


if __name__ == "__main__":
    main()
