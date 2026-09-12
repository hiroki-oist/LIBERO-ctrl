"""Minimum detectable effect.

Making the design factorial buys coverage at the cost of n per cell: standard LIBERO puts 500
rollouts in a suite, and LIBERO-CTRL spreads its budget across axes and levels. This script
states that trade-off explicitly, by computing how many points of difference the design can
actually detect.

Design:
  nominal   : 40 tasks x 50 initial states = 2,000 (500 per suite)
  perturbed : 40 tasks x 6 axes x 3 levels x 10 configs = 7,200
              -> per (suite, axis, level) cell: 10 tasks x 10 configs = n=100
              -> pooling (axis, level) over four suites: n=400

The primary test is a paired McNemar, pairing the nominal and perturbed rollout on the same
initial state. McNemar uses only the discordant pairs, so power depends strongly on the
discordance rate p_d. Power is computed exactly from the binomial, not by normal approximation.
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

import numpy as np
from scipy.stats import binom, norm

ALPHA, POWER = 0.05, 0.80


def mcnemar_power(n, p10, p01, alpha=ALPHA):
    """Exact power. The discordant count is D ~ Bin(n, p_d), and within it one side is
    ~Bin(D, p10/p_d), tested binomially."""
    pd = p10 + p01
    if pd <= 0: return 0.0
    q = p10 / pd
    tot = 0.0
    for D in range(1, n + 1):
        pD = binom.pmf(D, n, pd)
        if pD < 1e-12: continue
        k = np.arange(D + 1)
        # p-value of the two-sided binomial test (p=0.5)
        pv = np.minimum(1.0, 2 * np.minimum(binom.cdf(k, D, .5), binom.sf(k - 1, D, .5)))
        tot += pD * binom.pmf(k, D, q)[pv <= alpha].sum()
    return float(tot)


def mcnemar_mde(n, pd, alpha=ALPHA, power=POWER):
    """Smallest detectable |p10 - p01| in points, at a fixed discordance rate pd."""
    lo, hi = 0.0, pd
    for _ in range(40):
        d = (lo + hi) / 2
        p10, p01 = (pd + d) / 2, (pd - d) / 2
        if mcnemar_power(n, p10, p01, alpha) >= power: hi = d
        else: lo = d
    return hi * 100


def prop_mde(n1, n2, p=0.5, alpha=ALPHA, power=POWER):
    """MDE in points for an unpaired difference of proportions (normal approximation, worst
    case p=0.5)."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return z * np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) * 100


def ci(n, p=0.5, alpha=ALPHA):
    return norm.ppf(1 - alpha / 2) * np.sqrt(p * (1 - p) / n) * 100


print("=" * 78)
print("1. half-width of the 95% CI on a single cell's success rate, in points (worst case p=0.5)")
print(f"  {'n':>6s}  {'half-width':>11s}   what it is")
for n, use in ((100, "one (suite, axis, level) cell"), (400, "(axis, level) pooled over 4 suites"),
               (500, "nominal, one suite"), (1000, "one axis, one suite, three levels"),
               (2000, "nominal, all"), (7200, "perturbed, all")):
    print(f"  {n:6d}  {ci(n):7.2f}    {use}")

print("\n" + "=" * 78)
print("2. MDE of the paired McNemar test, in points -- nominal and perturbed on the same init")
print("   p_d is the fraction of pairs whose outcomes differ")
print(f"  {'p_d':>6s}" + "".join(f"{('n='+str(n)):>12s}" for n in (100, 200, 400, 1000)))
for pd in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
    print(f"  {pd*100:5.0f}%" + "".join(f"{mcnemar_mde(n, pd):11.2f} " for n in (100, 200, 400, 1000)))

print("\n" + "=" * 78)
print("3. for reference: MDE of unpaired comparisons, in points")
print(f"  nominal(500) vs perturbed cell(100) : {prop_mde(500, 100):.2f}")
print(f"  policy A vs policy B, same cell     : {prop_mde(100, 100):.2f}  (pairing and using McNemar shrinks this a lot)")
print(f"  policy A vs policy B, 4 suites      : {prop_mde(400, 400):.2f}")

print("\n" + "=" * 78)
print("4. alternative designs: total perturbed rollouts, and n per (suite, axis, level) cell")
print(f"  {'design':38s}{'n/cell':>8s}{'total':>10s}{'MDE(p_d=.2)':>13s}")
for lab, ncfg, ninit in (("10 configs x 1 init (as used)", 10, 1), ("10 configs x 2 inits", 10, 2),
                         ("20 configs x 1 init", 20, 1), ("10 configs x 5 inits", 10, 5)):
    n = 10 * ncfg * ninit                      # 10 tasks x configs x inits
    tot = 4 * n * 6 * 3
    print(f"  {lab:38s}{n:8d}{tot:10,d}{mcnemar_mde(n, 0.20):12.2f}")

print("\n" + "=" * 78)
print("5. power of the Cochran-Armitage trend test over severity")
print("   The main claim is a slope -- success falls monotonically as severity rises -- so a")
print("   trend test has more power than pairwise comparisons of L1/L2/L3.")
print("   Model: SR = p0 - slope * s for s = 1,2,3, with n rollouts per level.")


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


print(f"  {'slope per level':>20s}" + "".join(f"{('n='+str(n)):>10s}" for n in (100, 200, 400)))
for slope in (0.02, 0.03, 0.05, 0.08, 0.10):
    print(f"  {slope*100:17.0f}pp " + "".join(f"{ca_trend_power(n, 0.80, slope)*100:9.1f}%" for n in (100, 200, 400)))
print("  n is per level: 100 for one (suite, axis), 400 pooled over four suites")


def ca_trend_mde(n, p0=0.80):
    lo, hi = 0.0, 0.30
    for _ in range(25):
        m = (lo + hi) / 2
        if ca_trend_power(n, p0, m, nsim=8000) >= POWER: hi = m
        else: lo = m
    return hi * 100


print(f"\n  smallest detectable slope, points per level: "
      + "  ".join(f"n={n}: {ca_trend_mde(n):.1f}" for n in (100, 200, 400)))
