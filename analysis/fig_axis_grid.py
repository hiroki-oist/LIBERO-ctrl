#!/usr/bin/env python3
"""The 147 numbers of tab:axes as one grid: seven policies down, seven axes across.

Each cell is one (policy, axis) pair and shows how the success rate falls with severity,
clean -> L1 -> L2 -> L3. Every cell shares its axes, so the cells are directly comparable:
the dashed line is that policy's own nominal score and the shaded area is what the axis
takes away from it.

A single hue is used throughout. Identity is carried by the row and column labels, which a
facet grid already gives for free, so colour is left to do one job only -- separating the
curve from the nominal reference.

Writes docs/figs/axis_grid.{png,pdf}, which the README embeds.
"""

import os as _os
ROOT = _os.environ.get("LIBERO_CTRL_ROOT",
       _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
RESULTS = _os.path.join(ROOT, "results", "paper")
OUT_DIR = _os.path.join(ROOT, "docs", "figs")
_os.makedirs(OUT_DIR, exist_ok=True)

import json, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

AXES = ["camera", "lighting", "robot", "sensor", "actuation", "language", "combination"]
HEAD = ["Camera", "Lighting", "Initial pose", "Sensor", "Actuation", "Language", "All six\nat once"]
POL = [(r"$\pi_{0.5}$",  "4.14 B", "pi05"),
       ("OpenVLA-OFT",   "7.54 B", "oft"),
       ("UniVLA",        "7.54 B", "univla"),
       ("SmolVLA",       "450 M",  "smolvla"),
       ("VLA-JEPA",      "2.77 B", "vlajepa"),
       ("PredVLA",       "0.68 M", "predvla_s13"),
       ("MINERVA$^{*}$", "0.54 M", "minerva")]
# MINERVA resolves the instruction to an index in a fixed table, so a paraphrase is not an
# admissible input. Its language axis is the identity, which makes both that cell and the
# simultaneous one -- five effective perturbations plus an identity control -- uninterpretable
# as a result for those axes. They are left empty rather than plotted.
NA = {("minerva", "language"), ("minerva", "combination")}

INK   = "#3D6FA8"      # the one hue; identity comes from the labels
GREY  = "#9AA0A6"
RULE  = "#B9BEC4"
TEXT  = "#3C4043"
MUTED = "#80868B"


def load(run):
    rows = {}
    for f in glob.glob(f"{RESULTS}/{run}/*.jsonl"):
        for line in open(f):
            try: r = json.loads(line)
            except Exception: continue
            rows[r["rollout_id"]] = r
    return list(rows.values())


def rate(rows, axis, level):
    s = [r["success"] for r in rows if r.get("axis") == axis and r.get("level") == level]
    return 100 * sum(s) / len(s) if s else None


X, XT = [0, 1, 2, 3], ["clean", "L1", "L2", "L3"]
fig, grid = plt.subplots(len(POL), len(AXES), figsize=(11.6, 8.6), sharex=True, sharey=True)

for i, (name, size, run) in enumerate(POL):
    ev, cl = load(f"{run}_eval"), load(f"{run}_clean")
    s0 = 100 * sum(r["success"] for r in cl) / len(cl)
    for j, axis in enumerate(AXES):
        a = grid[i][j]
        if (run, axis) in NA:
            a.text(1.5, 50, "n/a", ha="center", va="center", fontsize=9, color=GREY)
        else:
            ys = [s0] + [rate(ev, axis, lv) for lv in ("L1", "L2", "L3")]
            a.axhline(s0, color=RULE, lw=0.8, ls=(0, (3, 2)), zorder=1)
            a.fill_between(X, ys, s0, color=INK, alpha=0.13, lw=0, zorder=2)
            a.plot(X, ys, "-o", color=INK, lw=1.7, ms=3.6, mew=0, zorder=3,
                   solid_capstyle="round")
            a.annotate(f"{ys[-1]:.0f}", (3, ys[-1]), textcoords="offset points",
                       xytext=(4, -1), ha="left", va="center", fontsize=7.5, color=TEXT)
        a.set_ylim(-6, 112); a.set_xlim(-0.3, 4.25)
        a.set_xticks(X); a.set_xticklabels(XT, fontsize=7.5)
        a.set_yticks([0, 50, 100])
        a.tick_params(labelsize=7.5, length=2, colors=MUTED)
        a.grid(axis="y", color=RULE, alpha=0.35, lw=0.5)
        a.set_axisbelow(True)
        for side in ("top", "right"): a.spines[side].set_visible(False)
        for side in ("left", "bottom"): a.spines[side].set_color(RULE)
        if i == 0:
            a.set_title(HEAD[j], fontsize=9.5, color=TEXT, pad=7)
        if j == 0:
            a.set_ylabel(f"{name}\n{size}", rotation=0, ha="right", va="center",
                         labelpad=12, fontsize=9.5, color=TEXT)

fig.supylabel("success rate (%)", x=0.012, fontsize=9.5, color=MUTED)
fig.tight_layout(pad=0.5, w_pad=0.7, h_pad=0.8)
out = f"{OUT_DIR}/axis_grid.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
print("written:", out, "and .png")
