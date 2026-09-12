"""軸ごとに、全方策の L1-L3 の劣化を「保持率」と「絶対値」の両方で描く。

  左軸 = 保持率（clean を 100 とした割合）、丸 + 実線
  右軸 = 絶対成功率（%）、三角 + 破線
両軸とも 0-105 に固定してあるので、2 本の曲線の乖離がそのまま
「clean が低い方策ほど絶対値では沈む」ことを表す。
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = ROOT
AXES = ["camera", "lighting", "robot", "sensor", "actuation", "language", "combination"]
TITLE = {"camera": "Camera", "lighting": "Lighting", "robot": "Initial pose", "sensor": "Sensor",
         "actuation": "Actuation", "language": "Language", "combination": "All six"}
SPEC = [("MINERVA",     ["minerva_eval", "minerva_comb"], ["minerva_clean"], "#4C72B0"),
        ("SmolVLA",     ["smolvla_eval"],                 ["smolvla_clean"], "#DD8452"),
        ("VLA-JEPA",    ["vlajepa_eval"],                 ["vlajepa_clean"], "#55A868"),
        (r"$\pi_{0.5}$",["pi05_eval"],                    ["pi05_clean"],    "#C44E52"),
        ("OpenVLA-OFT", ["oft_eval"],                     ["oft_clean"],     "#8172B3"),
        ("UniVLA",      ["univla_eval"],                  ["univla_clean"],  "#937860"),
        # PredVLA は公開 4 シードの平均。eval はシャード出力ディレクトリも束ねる
        ("PredVLA (4 seeds)",
         [d for s_ in (13, 5, 8, 10) for d in
          (f"predvla_s{s_}_eval", f"predvla_s{s_}_sh0_eval", f"predvla_s{s_}_sh1_eval",
           f"predvla_s{s_}_sh2_eval", f"predvla_s{s_}_sh3_eval")],
         ["predvla_s13_clean", "predvla_s5_clean",
          "predvla_s8_sh0_clean", "predvla_s8_sh1_clean",
          "predvla_s10_sh0_clean", "predvla_s10_sh1_clean"], "#DA8BC3")]

tk = collections.defaultdict(list)
# 統合済みなので追加の読み込み元は無い

def load(ds):
    o = {}
    for d in ds:
        for f in glob.glob(f"{RESULTS}/{d}/*.jsonl"):
            for l in open(f):
                r = json.loads(l); o[r["rollout_id"]] = r
        for r in tk.get(d, []):
            o[r["rollout_id"]] = r
    return list(o.values())

DATA = {}
for name, ed, cd, col in SPEC:
    ev, cl = load(ed), load(cd)
    clean = 100 * sum(r["success"] for r in cl) / len(cl)
    a = collections.defaultdict(lambda: [0, 0])
    for r in ev:
        k = (r["axis"], r["level"])
        a[k][0] += r["success"]; a[k][1] += 1
    DATA[name] = (clean, {k: 100 * v[0] / v[1] for k, v in a.items() if v[1]}, col)

fig, axarr = plt.subplots(2, 4, figsize=(19, 8.6))
L = ["L1", "L2", "L3"]; x = [1, 2, 3]
for i, ax_name in enumerate(AXES):
    ax = axarr[i // 4][i % 4]
    ax2 = ax.twinx()
    for name, _, _, _ in SPEC:
        clean, sr, col = DATA[name]
        y_abs = [sr.get((ax_name, l)) for l in L]
        if any(v is None for v in y_abs):
            continue
        y_ret = [100 * v / clean for v in y_abs]
        ax.plot(x, y_ret, "-o", color=col, ms=7, lw=2.0, label=name, zorder=3)
        ax2.plot(x, y_abs, "--^", color=col, ms=7, lw=1.4, alpha=0.75, zorder=2)
    for a_ in (ax, ax2):
        a_.set_ylim(-3, 108); a_.set_xlim(0.7, 3.3)
    ax.set_xticks(x); ax.set_xticklabels(L)
    ax.grid(alpha=.25, zorder=0)
    ax.set_title(TITLE[ax_name], fontsize=13, fontweight="bold")
    if i % 4 == 0: ax.set_ylabel("Retention (% of clean)   $\\bullet$")
    else:          ax.set_yticklabels([])
    if i % 4 == 3: ax2.set_ylabel("Absolute success rate (%)   $\\blacktriangle$")
    else:          ax2.set_yticklabels([])

# 凡例パネル
lg = axarr[1][3]; lg.axis("off"); axarr[1][3].twinx().axis("off")
h, lb = axarr[0][0].get_legend_handles_labels()
lg.legend(h, lb, loc="center", fontsize=13, frameon=False, title="Policy (nominal %)",
          title_fontsize=13)
lg.text(0.5, 0.12, "$\\bullet$ solid: retention (left axis)\n"
                   "$\\blacktriangle$ dashed: absolute (right axis)",
        ha="center", va="center", fontsize=11, transform=lg.transAxes)
for name, _, _, _ in SPEC:
    pass
fig.suptitle("Per-axis degradation of six VLA policies, LIBERO-CTRL", fontsize=15, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = f"{OUT_DIR}/axis_profiles.png"
fig.savefig(out, dpi=160)
print("saved ->", out)
for name, _, _, _ in SPEC:
    clean, sr, _ = DATA[name]
    print(f"  {name:14s} clean={clean:5.1f}  " +
          "  ".join(f"{a[:4]}:{sr.get((a,'L3'),float('nan')):5.1f}" for a in AXES))
