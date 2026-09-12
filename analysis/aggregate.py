"""結果 jsonl を集計して、論文用の表と図の元データを作る。

出力:
  analysis/summary.json   モデル × 軸 × レベルの成功率と件数
  analysis/summary.md     人が読む表
GPU を使わないので評価と並行して回せる。
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

import json, glob, os, sys, collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = RESULTS
AXES = ["camera", "lighting", "robot", "sensor", "actuation", "language", "combination"]
LEVELS = ["L1", "L2", "L3"]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
# 結果ディレクトリ名 -> (モデル名, split)
def discover():
    out = collections.defaultdict(dict)
    for d in sorted(glob.glob(os.path.join(RES, "*"))):
        if not os.path.isdir(d): continue
        b = os.path.basename(d)
        if b.startswith("triage_") or b in ("test",): continue
        for suf in ("_clean", "_eval", "_comb"):
            if b.endswith(suf):
                out[b[: -len(suf)]][suf[1:]] = d
    return out


def load(d):
    """★rollout_id で重複を落とす。

    2台のマシンで分担して走らせ、あとで片方へ集約するので、同じ rollout_id が
    複数ファイルに入りうる（実測: taketomi の minerva_eval 4,680 本は
    Fujiwara の 7,200 本に完全に含まれていた）。素朴に足すと成功率が歪む。
    resume でも同じ rollout_id が再度書かれることがある。**後勝ちで1本に潰す。**
    """
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            rows[r.get("rollout_id", len(rows))] = r
    return list(rows.values())


def rate(rows):
    n = len(rows)
    return (sum(1 for r in rows if r["success"]) / n if n else float("nan")), n


def main():
    models = discover()
    S = {}
    for m, dirs in models.items():
        rows = []
        for k, d in dirs.items(): rows += load(d)
        if not rows: continue
        e = {}
        cl = [r for r in rows if r["axis"] == "clean"]
        e["clean"] = dict(zip(("sr", "n"), rate(cl)))
        e["clean_by_suite"] = {s: dict(zip(("sr", "n"), rate([r for r in cl if r["suite"] == s])))
                               for s in SUITES}
        for a in AXES:
            e[a] = {}
            for lv in LEVELS:
                sub = [r for r in rows if r["axis"] == a and r["level"] == lv]
                e[a][lv] = dict(zip(("sr", "n"), rate(sub)))
                e[a][lv]["by_suite"] = {s: dict(zip(("sr", "n"),
                    rate([r for r in sub if r["suite"] == s]))) for s in SUITES}
        # 乗法性: 独立仮定からの予測 vs 観測
        c = e["clean"]["sr"]
        e["multiplicative"] = {}
        for lv in LEVELS:
            pred = c
            ok = True
            for a in AXES[:-1]:                       # combination を除く6軸
                r = e[a][lv]["sr"]
                if not np.isfinite(r) or not np.isfinite(c) or c == 0: ok = False; break
                pred *= r / c
            obs = e["combination"][lv]["sr"]
            e["multiplicative"][lv] = dict(pred=pred if ok else None,
                                           obs=obs if np.isfinite(obs) else None,
                                           diff=(obs - pred) if (ok and np.isfinite(obs)) else None)
        S[m] = e
    json.dump(S, open(os.path.join(OUT_DIR, "summary.json"), "w"),
              ensure_ascii=False, indent=1, default=float)

    # ---- 人が読む表
    L = []
    L.append("# 結果まとめ（自動生成）\n")
    L.append(f"生成: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}\n")
    for m, e in sorted(S.items()):
        L.append(f"\n## {m}\n")
        L.append(f"clean: **{e['clean']['sr']*100:.2f}%** (n={e['clean']['n']})\n")
        L.append(f"| 軸 | L1 | L2 | L3 | 傾き/level |")
        L.append("|---|---:|---:|---:|---:|")
        for a in AXES:
            row = f"| {a} "
            v = []
            for lv in LEVELS:
                x = e[a][lv]
                row += f"| {x['sr']*100:.1f}% ({x['n']}) " if np.isfinite(x["sr"]) else "| - "
                v.append(x["sr"])
            sl = (e["clean"]["sr"] - v[2]) / 3 * 100 if np.isfinite(v[2]) else float("nan")
            row += f"| {sl:.1f}pp |" if np.isfinite(sl) else "| - |"
            L.append(row)
        mm = e["multiplicative"]
        L.append(f"\n乗法性（独立仮定の予測 vs 観測）\n")
        L.append("| level | 予測 | 観測 | 差 |")
        L.append("|---|---:|---:|---:|")
        for lv in LEVELS:
            d = mm[lv]
            if d["pred"] is None or d["obs"] is None:
                L.append(f"| {lv} | - | - | - |")
            else:
                L.append(f"| {lv} | {d['pred']*100:.1f}% | {d['obs']*100:.1f}% | {d['diff']*100:+.1f} |")
    open(os.path.join(ROOT, "analysis", "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:60]))
    print(f"\nsaved analysis/summary.json, analysis/summary.md  （モデル {len(S)} 本）")


if __name__ == "__main__":
    main()
