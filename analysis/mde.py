"""検出可能な最小効果量 (MDE) の計算。監査論文 §18-② への回答。

LIBERO-CTRL は factorial 化した分、セルあたりの n が標準 LIBERO（500/suite）より小さい。
**「カバレッジと引き換えにセルあたりの精度を落としている」というトレードオフを
自分から明示する**ために、実際に何ポイントの差を検出できるのかを出す。

設計:
  clean      : 40 タスク × 50 init = 2,000（suite あたり 500）
  摂動        : 40 タスク × 6軸 × 3レベル × 10 config = 7,200
               -> (suite, axis, level) セルあたり 10 タスク × 10 config = **n=100**
               -> (axis, level) を4 suite でプールすると **n=400**

主たる検定は **paired McNemar**（同一 init 上の clean と摂動を対にする）。
McNemar は不一致ペアのみを使うので、検出力は不一致率 p_d に強く依存する。
ここでは正規近似ではなく**二項分布で厳密に**検出力を計算する。
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

import numpy as np
from scipy.stats import binom, norm

ALPHA, POWER = 0.05, 0.80


def mcnemar_power(n, p10, p01, alpha=ALPHA):
    """厳密検出力。不一致数 D~Bin(n,p_d)、そのうち片側 ~Bin(D, p10/p_d) の二項検定。"""
    pd = p10 + p01
    if pd <= 0: return 0.0
    q = p10 / pd
    tot = 0.0
    for D in range(1, n + 1):
        pD = binom.pmf(D, n, pd)
        if pD < 1e-12: continue
        k = np.arange(D + 1)
        # 両側二項検定 (p=0.5) の p 値
        pv = np.minimum(1.0, 2 * np.minimum(binom.cdf(k, D, .5), binom.sf(k - 1, D, .5)))
        tot += pD * binom.pmf(k, D, q)[pv <= alpha].sum()
    return float(tot)


def mcnemar_mde(n, pd, alpha=ALPHA, power=POWER):
    """不一致率 pd を固定したときに検出できる最小の |p10-p01| [pp]。"""
    lo, hi = 0.0, pd
    for _ in range(40):
        d = (lo + hi) / 2
        p10, p01 = (pd + d) / 2, (pd - d) / 2
        if mcnemar_power(n, p10, p01, alpha) >= power: hi = d
        else: lo = d
    return hi * 100


def prop_mde(n1, n2, p=0.5, alpha=ALPHA, power=POWER):
    """対でない2群の比率差の MDE [pp]（正規近似、最悪ケース p=0.5）。"""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return z * np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) * 100


def ci(n, p=0.5, alpha=ALPHA):
    return norm.ppf(1 - alpha / 2) * np.sqrt(p * (1 - p) / n) * 100


print("=" * 78)
print("① 単一セルの成功率の 95% CI 半幅 [pp]（p=0.5 の最悪ケース）")
print(f"  {'n':>6s}  {'CI半幅':>8s}   用途")
for n, use in ((100, "(suite, axis, level) 1セル"), (400, "(axis, level) 4 suite プール"),
               (500, "clean 1 suite"), (1000, "(axis) 1 suite で3レベル分"),
               (2000, "clean 全体"), (7200, "摂動 全体")):
    print(f"  {n:6d}  {ci(n):7.2f}    {use}")

print("\n" + "=" * 78)
print("② paired McNemar の MDE [pp]  — clean と摂動を同一 init で対にする")
print("   不一致率 p_d = clean と摂動で結果が異なるペアの割合")
print(f"  {'p_d':>6s}" + "".join(f"{('n='+str(n)):>12s}" for n in (100, 200, 400, 1000)))
for pd in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
    print(f"  {pd*100:5.0f}%" + "".join(f"{mcnemar_mde(n, pd):11.2f} " for n in (100, 200, 400, 1000)))

print("\n" + "=" * 78)
print("③ 参考: 対でない比較の MDE [pp]")
print(f"  clean(500) vs 摂動セル(100)  : {prop_mde(500, 100):.2f}")
print(f"  モデルA vs モデルB, 同一セル  : {prop_mde(100, 100):.2f}  (対にすれば McNemar で大幅に縮む)")
print(f"  モデルA vs モデルB, 4 suite   : {prop_mde(400, 400):.2f}")

print("\n" + "=" * 78)
print("④ 設計の代替案（摂動の総本数と、(suite,axis,level) セルの n）")
print(f"  {'設計':38s}{'n/セル':>8s}{'総本数':>10s}{'MDE(p_d=.2)':>13s}")
for lab, ncfg, ninit in (("10 config × 1 init（現行）", 10, 1), ("10 config × 2 init", 10, 2),
                         ("20 config × 1 init", 20, 1), ("10 config × 5 init", 10, 5)):
    n = 10 * ncfg * ninit                      # 10 タスク × config × init
    tot = 4 * n * 6 * 3
    print(f"  {lab:38s}{n:8d}{tot:10,d}{mcnemar_mde(n, 0.20):12.2f}")

print("\n" + "=" * 78)
print("⑤ severity トレンド検定 (Cochran-Armitage) の検出力")
print("   主結果は『severity を上げると成功率が単調に下がる』という**傾き**なので、")
print("   L1/L2/L3 を対比較するより傾向検定のほうが検出力が高い。")
print("   モデル: SR = p0 - slope * s  (s=1,2,3)、各レベル n 本。")


def ca_trend_power(n, p0, slope, alpha=ALPHA, nsim=20000, seed=0):
    rng = np.random.default_rng(seed)
    s = np.array([1., 2., 3.]); sbar = s.mean()
    p = np.clip(p0 - slope * s, 0.001, 0.999)
    x = rng.binomial(n, p[None, :], size=(nsim, 3))
    N = 3 * n; R = x.sum(1)
    T = ((s - sbar) * x).sum(1)
    pbar = R / N
    var = pbar * (1 - pbar) * n * ((s - sbar) ** 2).sum()
    z = np.divide(T, np.sqrt(var), out=np.zeros_like(T, float), where=var > 0)
    return float((np.abs(z) > norm.ppf(1 - alpha / 2)).mean())


print(f"  {'1レベルあたりの傾き':>20s}" + "".join(f"{('n='+str(n)):>10s}" for n in (100, 200, 400)))
for slope in (0.02, 0.03, 0.05, 0.08, 0.10):
    print(f"  {slope*100:17.0f}pp " + "".join(f"{ca_trend_power(n, 0.80, slope)*100:9.1f}%" for n in (100, 200, 400)))
print("  ※ n は 1 レベルあたりの本数。(suite,axis) なら n=100、4 suite プールなら n=400")


def ca_trend_mde(n, p0=0.80):
    lo, hi = 0.0, 0.30
    for _ in range(25):
        m = (lo + hi) / 2
        if ca_trend_power(n, p0, m, nsim=8000) >= POWER: hi = m
        else: lo = m
    return hi * 100


print(f"\n  検出できる最小の傾き [pp/レベル]: "
      + "  ".join(f"n={n}: {ca_trend_mde(n):.1f}" for n in (100, 200, 400)))
