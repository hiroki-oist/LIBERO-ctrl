#!/usr/bin/env python3
"""方策ごとに1パネル、軸ごとに1本の線。tab:axes の 147 個の数字を 7 つの指紋にする。

  x = clean, L1, L2, L3        y = 成功率 (%)
  色付きの線 = 単一6軸、太い黒線 = 6軸同時
各パネルは同じ y 軸なので、パネル間で「最悪の軸が違う」ことがそのまま見える。
"""

import os as _os
# ★結果は results/paper/<run名>/rollouts.jsonl に統合済み（fuji と taketomi の両方を、
#   taketomi 優先でマージ）。旧リポジトリの /tmp/tkpull による上書きはもう不要。
ROOT = _os.environ.get("LIBERO_CTRL_ROOT",
       _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
RESULTS = _os.path.join(ROOT, "results", "paper")
MANIFESTS = _os.path.join(ROOT, "manifests", "v0.1")
OUT_DIR = _os.path.join(ROOT, "analysis", "out")
_os.makedirs(OUT_DIR, exist_ok=True)

import json, glob, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = ROOT
AX   = ["camera","lighting","robot","sensor","actuation","language"]
LAB  = {"camera":"Camera","lighting":"Lighting","robot":"Initial pose",
        "sensor":"Sensor","actuation":"Actuation","language":"Language"}
COL  = {"camera":"#4C72B0","lighting":"#DD8452","robot":"#55A868",
        "sensor":"#C44E52","actuation":"#8172B3","language":"#937860"}
POL = [("MINERVA$^{*}$","0.54 M",["minerva_eval"],["minerva_clean"]),
       ("PredVLA","0.68 M",["predvla_s13_eval"],["predvla_s13_clean"]),
       ("SmolVLA","450 M",["smolvla_eval"],["smolvla_clean"]),
       ("VLA-JEPA","2.77 B",["vlajepa_eval"],["vlajepa_clean"]),
       (r"$\pi_{0.5}$","4.14 B",["pi05_eval"],["pi05_clean"]),
       ("OpenVLA-OFT","7.54 B",["oft_eval"],["oft_clean"]),
       ("UniVLA","7.54 B",["univla_eval"],["univla_clean"])]
def load(ds):
    o={}
    for d in ds:
        for f in glob.glob(f"{RESULTS}/{d}/*.jsonl"):
            for l in open(f): r=json.loads(l); o[r["rollout_id"]]=r
    return list(o.values())
def rate(rows, ax, lv):
    s=[r["success"] for r in rows if r.get("axis")==ax and r.get("level")==lv]
    return 100*sum(s)/len(s) if s else None

fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.0), sharex=True, sharey=True)
X = [0,1,2,3]; XT = ["clean","L1","L2","L3"]
for i,(name,sz,ed,cd) in enumerate(POL):
    a = axes[i//4][i%4]
    ev = load(ed); cl = load(cd)
    s0 = 100*sum(r["success"] for r in cl)/len(cl)
    for ax_ in AX:
        ys = [s0]+[rate(ev,ax_,l) for l in ("L1","L2","L3")]
        a.plot(X, ys, "-o", ms=3.2, lw=1.4, color=COL[ax_], label=LAB[ax_], zorder=2)
    ys = [s0]+[rate(ev,"combination",l) for l in ("L1","L2","L3")]
    a.plot(X, ys, "-s", ms=4.0, lw=2.4, color="black", label="All six", zorder=3)
    a.set_title(f"{name}  ({sz})", fontsize=9.5, pad=3)
    a.set_ylim(-3,103); a.set_xticks(X); a.set_xticklabels(XT, fontsize=8)
    a.grid(alpha=.25, lw=.5); a.tick_params(labelsize=8)
    if i%4==0: a.set_ylabel("success rate (%)", fontsize=9)
h,l = axes[0][0].get_legend_handles_labels()
axes[1][3].axis("off")
axes[1][3].legend(h, l, loc="center", fontsize=9, frameon=False, handlelength=1.8)
fig.tight_layout(pad=0.6)
out=f"{OUT_DIR}/policy_profiles.pdf"
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf",".png"), dpi=160, bbox_inches="tight")
print("書き出し:", out)
