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
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

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


ROW_LABEL = {"camera": "Camera", "lighting": "Light", "robot": "Robot",
             "sensor": "Sensor", "combination": "Simultaneous"}
COL_LABEL = {"clean": "Nominal", "L1": "L1  (r = 2)", "L2": "L2  (r = 4)", "L3": "L3  (r = 8)"}


def style(paper: bool, fontsize: float = 11.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if paper:
        plt.rcParams.update({
            "font.family": "serif",
            # Nimbus Roman is metrically compatible with the Times face IEEE templates use.
            "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
            "font.size": fontsize,
            "pdf.fonttype": 42,     # embed TrueType rather than Type 3
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
        })
    return plt


def draw(plt, grid, *, levels_as_rows, width, paper):
    """Full grid: every axis at every level, with the nominal condition repeated."""
    cols = ("clean",) + LEVELS
    if levels_as_rows:
        rows, cells = cols, VISUAL
        label_row, label_col = COL_LABEL, ROW_LABEL
    else:
        rows, cells = VISUAL, cols
        label_row, label_col = ROW_LABEL, COL_LABEL
    panel = width / len(cells)
    fig, ax = plt.subplots(len(rows), len(cells), figsize=(width, panel * len(rows)))
    for i, r in enumerate(rows):
        for j, c in enumerate(cells):
            key = (r, c) if levels_as_rows else (c, r)
            _panel(ax[i, j], grid[key],
                   title=label_col[c] if i == 0 else None,
                   ylabel=label_row[r] if j == 0 else None)
    fig.subplots_adjust(wspace=0.035, hspace=0.035,
                        left=0.0, right=1.0, top=1.0, bottom=0.0)
    if not paper:
        fig.tight_layout()
    return fig


def draw_reference(plt, grid, *, width, act=None):
    """Paper layout: the nominal reference at the top left, severity down, axis across.

    The nominal frame is identical for every axis, so it is shown once rather than repeated
    along a whole row. When `act` is given, the space this frees underneath carries the one
    axis that cannot be drawn as an image: the same command sequence issued under each severity
    level, and the tracking error it opens up.
    """
    ratios = [2.15, 0.24] + [1.0] * len(VISUAL)
    panel = width / sum(ratios)
    fig = plt.figure(figsize=(width, panel * len(LEVELS) * 1.04))
    gs = fig.add_gridspec(len(LEVELS), len(ratios), width_ratios=ratios,
                          wspace=0.06, hspace=0.05,
                          left=0.0, right=1.0, top=1.0, bottom=0.0)
    nom = fig.add_subplot(gs[0, 0])
    _panel(nom, grid[("clean", VISUAL[0])], title=COL_LABEL["clean"])
    nom.set_anchor("SW")        # sit above the actuation block rather than centred in the cell
    for i, L in enumerate(LEVELS):
        for j, axis in enumerate(VISUAL):
            _panel(fig.add_subplot(gs[i, j + 2]), grid[(L, axis)],
                   title=ROW_LABEL[axis] if i == 0 else None,
                   ylabel=COL_LABEL[L] if j == 0 else None)
    if act is not None:
        _actuation_block(fig, gs[1:, 0], act)
    return fig


COLOUR_L = {"L1": "#4C72B0", "L2": "#DD8452", "L3": "#C44E52"}


def _actuation_block(fig, cell, act, config=0):
    """The actuation axis, which leaves the image untouched, measured in the simulator.

    Left: the end-effector path produced by one fixed command sequence, nominally and at each
    level. Right: the tracking error of all ten configurations per level.
    """
    import matplotlib as mpl
    ref, paths, dev = act
    fs = mpl.rcParams["font.size"]
    sub = cell.subgridspec(1, 2, wspace=0.70)
    a1, a2 = fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])
    for h in (a1, a2):                      # keep a sane aspect inside a two-row-tall cell
        b = h.get_position()
        h.set_position([b.x0, b.y0 + b.height * 0.20, b.width, b.height * 0.60])
        h.spines["top"].set_visible(False); h.spines["right"].set_visible(False)
        for sp in h.spines.values(): sp.set_linewidth(0.6)
        h.tick_params(labelsize=fs * 0.7, length=2.0, width=0.55, pad=1.2)

    a1.plot(ref[:, 0] * 100, ref[:, 1] * 100, color="k", lw=1.5, label="nominal", zorder=4)
    for L in LEVELS:
        a1.plot(paths[L][config][:, 0] * 100, paths[L][config][:, 1] * 100,
                color=COLOUR_L[L], lw=1.1, label=L, zorder=3)
    a1.set_xlabel("$x$ (cm)", fontsize=fs * 0.8, labelpad=1)
    a1.set_ylabel("$y$ (cm)", fontsize=fs * 0.8, labelpad=1)
    a1.set_title("Actuation: path", fontsize=fs * 0.95, pad=3)
    a1.legend(frameon=False, fontsize=fs * 0.72, handlelength=0.9, labelspacing=0.12,
              borderpad=0.05, loc="best")
    a1.set_aspect("equal", adjustable="datalim")

    rng = np.random.default_rng(0)
    for i, L in enumerate(LEVELS):
        e = dev[L].mean(axis=1)                       # time-averaged tracking error, per config
        a2.scatter(i + rng.uniform(-0.13, 0.13, len(e)), e, s=5, color=COLOUR_L[L],
                   alpha=0.55, lw=0, zorder=2)
        q1, med, q3 = np.percentile(e, [25, 50, 75])
        a2.plot([i - 0.28, i + 0.28], [med, med], color=COLOUR_L[L], lw=1.6, zorder=3)
        a2.plot([i, i], [q1, q3], color=COLOUR_L[L], lw=0.9, zorder=3)
    a2.set_xticks(range(len(LEVELS))); a2.set_xticklabels(LEVELS)
    a2.set_xlim(-0.55, len(LEVELS) - 0.45)
    a2.set_ylabel("tracking error (mm)", fontsize=fs * 0.8, labelpad=1)
    a2.set_title("Actuation: error", fontsize=fs * 0.95, pad=3)
    a2.set_ylim(bottom=0)


def _panel(h, img, *, title=None, ylabel=None):
    import matplotlib as mpl
    fs = mpl.rcParams["font.size"]
    h.imshow(img, interpolation="lanczos")
    h.set_xticks([]); h.set_yticks([])
    for sp in h.spines.values():
        sp.set_linewidth(0.4); sp.set_color("0.55")
    if title:  h.set_title(title, fontsize=fs, pad=4)
    if ylabel: h.set_ylabel(ylabel, fontsize=fs, labelpad=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=2)
    ap.add_argument("--config", type=int, default=0)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--paper", action="store_true",
                    help="typeset for the paper: serif faces, embedded fonts, no title")
    ap.add_argument("--levels-as-rows", action="store_true",
                    help="severity down, axes across; fits a full-width figure better")
    ap.add_argument("--full-grid", action="store_true",
                    help="repeat the nominal frame for every axis instead of showing it once")
    ap.add_argument("--no-actuation", action="store_true",
                    help="leave out the measured actuation panel")
    ap.add_argument("--width", type=float, default=None,
                    help="figure width in inches (default 7.16, the IEEE two-column text width)")
    ap.add_argument("--fontsize", type=float, default=11.0,
                    help="label size in points (paper mode)")
    ap.add_argument("--name", default=None, help="output basename")
    ap.add_argument("--out", default=OUT_DIR, help="output directory")
    a = ap.parse_args()

    plt = style(a.paper, a.fontsize)
    from libero_ctrl.manifest import iter_rows, env_seed
    from libero_ctrl.env import make_task, close_task

    per = {}
    for r in iter_rows("eval", suite=a.suite, task_id=a.task):
        if r["config"] == a.config:
            per[(r["axis"], r["level"])] = r
    init_id = per[("camera", "L1")]["init_id"]
    clean = next(r for r in iter_rows("clean", suite=a.suite, task_id=a.task)
                 if r["init_id"] == init_id)

    reference = a.paper and not a.full_grid
    want_dev = reference and not a.no_actuation
    task = make_task(a.suite, a.task, res=a.res, seed=env_seed())
    try:
        base = render(task, clean, a.res)
        grid = {("clean", ax): base for ax in VISUAL}
        for ax in VISUAL:
            for L in LEVELS:
                grid[(L, ax)] = render(task, per[(ax, L)], a.res)
        act = None
        if want_dev:
            from fig_actuation import command_sequence, deviations
            rows_act = {}
            for r in iter_rows("eval", suite=a.suite, task_id=a.task, axis="actuation"):
                rows_act.setdefault(r["level"], {})[r["config"]] = r
            act = deviations(task, rows_act, command_sequence(), a.res, init_id)
    finally:
        close_task(task)

    width = a.width if a.width else (7.16 if (a.levels_as_rows or reference) else 5.4)
    fig = (draw_reference(plt, grid, width=width, act=act) if reference else
           draw(plt, grid, levels_as_rows=a.levels_as_rows, width=width, paper=a.paper))
    if not a.paper:
        fig.suptitle(f"{a.suite} task {a.task}, initial state {init_id}, "
                     f"configuration {a.config}: \u201c{clean['language']}\u201d", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    name = a.name or ("perturbation_grid_paper" if a.paper else "perturbation_grid")
    _os.makedirs(a.out, exist_ok=True)
    fig.savefig(f"{a.out}/{name}.pdf")          # vector, for the paper
    fig.savefig(f"{a.out}/{name}.png", dpi=200)  # raster, for the repository
    print(f"written -> {a.out}/{name}.pdf (+ .png)")

    meta = {"suite": a.suite, "task_id": a.task, "config": a.config, "init_id": init_id,
            "language": {"nominal": clean["language"],
                         **{L: per[("language", L)]["language"] for L in LEVELS}},
            "actuation": {L: per[("actuation", L)]["perturb"] for L in LEVELS},
            "perturb": {ax: {L: per[(ax, L)]["perturb"] for L in LEVELS} for ax in VISUAL}}
    with open(f"{a.out}/{name}.json", "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
