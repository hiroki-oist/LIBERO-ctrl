"""複合摂動が単一摂動の積で説明できるかを検定する。

帰無仮説（独立モデル）: 各軸の失敗が独立に起きるなら
    SR_comb = SR_clean * Π_a (SR_a / SR_clean)
これは「相対生存率の積」。観測との差が交互作用である。

差の有意性は **対応のあるブートストラップ**で見る。
combination の観測は 400 本（4 suite × 10 タスク × 10 config）、
予測側は 6 軸 × 400 本から作られるので、両方を同時にリサンプルする。
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

import json, glob, os, collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AXES = ["camera", "lighting", "robot", "sensor", "actuation", "language"]
LEVELS = ["L1", "L2", "L3"]


def load(model):
    # ★rollout_id で重複を落とす（2台分担 -> 片方へ集約するので同じ id が複数回入りうる）
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
    """独立モデルの予測と観測の差を、ブートストラップで区間推定する。"""
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
    # 点推定は元データで
    c0 = clean.mean(); p0 = c0
    for a in AXES: p0 *= ax[a].mean() / c0
    o0 = comb.mean()
    return dict(n_comb=len(comb), n_clean=len(clean),
                pred=p0, obs=o0, diff=o0 - p0,
                ci=(float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
                p_two_sided=float(2 * min((d <= 0).mean(), (d >= 0).mean())))


def main():
    # ★combination は `<model>_comb` に置く場合と `<model>_eval` に含める場合がある。
    #   MINERVA だけ別ディレクトリに流した経緯があり、VLA-JEPA 以降は eval に一括。
    #   どちらでも拾えるように、combination 行が実在するモデルを探す。
    models = []
    for d in sorted(glob.glob(os.path.join(ROOT, "results", "*_clean"))):
        m = os.path.basename(d)[: -len("_clean")]
        if any(r["axis"] == "combination" for r in load(m)): models.append(m)
    if not models:
        print("combination の結果がまだありません"); return
    print("複合摂動 vs 独立モデルの予測（対応のあるブートストラップ, 4000 回）\n")
    print(f"{'model':12s}{'level':>6s}{'予測':>9s}{'観測':>9s}{'差':>8s}{'95% CI':>18s}{'p':>8s}{'n':>6s}")
    for m in models:
        rows = load(m)
        for lv in LEVELS:
            r = boot(rows, lv)
            if r is None:
                print(f"{m:12s}{lv:>6s}{'データ不足':>28s}"); continue
            print(f"{m:12s}{lv:>6s}{r['pred']*100:8.1f}%{r['obs']*100:8.1f}%"
                  f"{r['diff']*100:+7.1f}"
                  f"  [{r['ci'][0]*100:+5.1f},{r['ci'][1]*100:+5.1f}]"
                  f"{r['p_two_sided']:8.3f}{r['n_comb']:6d}")
    print("\n  差の 95% CI が 0 を含めば「独立モデルからの逸脱は検出できない」")
    print("  = 単一軸の測定から複合摂動を予測できる、という主張の根拠になる")


if __name__ == "__main__":
    main()
