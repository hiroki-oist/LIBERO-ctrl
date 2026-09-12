"""補償（compensation）と創発失敗（emergent failure）が、摂動ベクトルの向きで説明できるかを検定する。

設計上、同一 severity レベルの config は**軸ごとのノルムが等しく向きだけが違う**ので、
「たまたま弱い摂動だった」という交絡は構成上ない。したがって
「どの向きの組合せが軌道を救う/殺すか」を直接問える。

  c（補償）: 単独では少なくとも1因子に殺されるが、6因子同時では成功する初期状態
  b（創発）: 6因子すべてに単独では耐えるが、同時では失敗する初期状態

それぞれ**条件付け集合の中で**ラベルを付け、25 個の摂動パラメータおよびその対積との
点双列相関を取る。レベルごとに z 化し、方策をまたいだ符号の一致で頑健性を見る。
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

import json, glob, collections, itertools, math, sys
import numpy as np

AX = ("camera", "lighting", "robot", "sensor", "actuation", "language")
ROOT = ROOT

# ---- 摂動ベクトル（combination 行のみが 5 軸ぶんを持つ）
FEAT, VEC = [], {}
for l in open(f"{ROOT}/manifests/v0.1/langcomb_all.jsonl"):
    r = json.loads(l)
    if r["axis"] != "combination":
        continue
    p = r["perturb"]
    if not FEAT:
        FEAT = [f"{a}.{k}" for a in ("camera", "lighting", "robot", "sensor", "actuation")
                for k in p[a]]
    VEC[(r["suite"], r["task_id"], r["level"], r["config"])] = np.array(
        [p[f.split(".")[0]][f.split(".")[1]] for f in FEAT], float)

# ---- 結果の読み込み（fuji ローカル + tk 書き出し）
tk = collections.defaultdict(list)   # 統合済みなので追加の読み込み元は無い

def load(ds):
    o = {}
    for d in ds:
        for f in glob.glob(f"{RESULTS}/{d}/*.jsonl"):
            for l in open(f):
                r = json.loads(l); o[r["rollout_id"]] = r
        for r in tk.get(d, []):
            o[r["rollout_id"]] = r
    return list(o.values())

SPEC = {"MINERVA": ["minerva_eval", "minerva_comb"], "SmolVLA": ["smolvla_eval"],
        "VLA-JEPA": ["vlajepa_eval"], "pi05": ["pi05_eval"],
        "OpenVLA-OFT": ["oft_eval"], "UniVLA": ["univla_eval"],
        "PredVLA": ["predvla_s13_eval"]}
FLOOR = {("MINERVA", "L3"), ("SmolVLA", "L3"), ("UniVLA", "L3")}   # 予測 <4% の床セル

def labels(name):
    """(level, key) -> ('c'|'b'|None)。条件付け集合の外は None。"""
    tbl = collections.defaultdict(dict)
    for r in load(SPEC[name]):
        tbl[(r["suite"], r["task_id"], r["level"], r["config"])][r["axis"]] = r["success"]
    out = {}
    for k, v in tbl.items():
        if not all(a in v for a in AX) or "combination" not in v:
            continue
        all_alone = all(v[a] for a in AX)
        out[k] = ("b" if not v["combination"] else "ok") if all_alone else \
                 ("c" if v["combination"] else "ng")
    return out

# ---- レベルごとに z 化した特徴行列
def design(keys):
    X = np.array([VEC[k] for k in keys])
    return (X - X.mean(0)) / (X.std(0) + 1e-12)

def pbis(x, y):
    """点双列相関と両側 p（Fisher z 近似）。"""
    if y.sum() < 5 or (1 - y).sum() < 5:
        return float("nan"), float("nan")
    r = np.corrcoef(x, y)[0, 1]
    n = len(y)
    if abs(r) >= 1 or n < 6:
        return r, float("nan")
    z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(n - 3)
    return r, math.erfc(abs(z) / math.sqrt(2))

def scan(term_names, term_fn, tag):
    """方策×レベルごとに相関を取り、方策をまたいで集計する。"""
    acc = collections.defaultdict(list)
    for name in SPEC:
        lab = labels(name)
        for L in ("L1", "L2", "L3"):
            if (name, L) in FLOOR:
                continue
            for tgt, cond in (("c", ("c", "ng")), ("b", ("b", "ok"))):
                keys = [k for k, v in lab.items() if k[2] == L and v in cond]
                if len(keys) < 40:
                    continue
                y = np.array([1.0 if lab[k] == tgt else 0.0 for k in keys])
                if y.sum() < 5 or (1 - y).sum() < 5:
                    continue
                Z = design(keys); T = term_fn(Z)
                for j, tn in enumerate(term_names):
                    r, p = pbis(T[:, j], y)
                    if not math.isnan(r):
                        acc[(tgt, tn)].append(r)
    rows = []
    for (tgt, tn), rs in acc.items():
        rs = np.array(rs)
        if len(rs) < 6:
            continue
        # 方策×レベルをまたいだ平均相関と、符号の一致数
        same = max((rs > 0).sum(), (rs < 0).sum())
        # 符号一致の二項検定（両側）
        from math import comb
        n = len(rs)
        pb = min(1.0, 2 * sum(comb(n, i) for i in range(same, n + 1)) / 2 ** n)
        rows.append((tgt, tn, rs.mean(), n, same, pb))
    rows.sort(key=lambda t: t[5])
    print(f"\n=== {tag} （{len(rows)} 項目、符号一致の二項検定でソート）")
    print(f"{'':4s} {'項':38s} {'平均r':>7s} {'n':>3s} {'同符号':>5s} {'p':>8s}")
    for tgt, tn, m, n, same, pb in rows[:12]:
        print(f"  {tgt:2s} {tn:38s} {m:+7.3f} {n:3d} {same:4d}/{n} {pb:8.4f}")
    return rows

single = scan(FEAT, lambda Z: Z, "単一パラメータ")
pairs = list(itertools.combinations(range(len(FEAT)), 2))
pnames = [f"{FEAT[i]} x {FEAT[j]}" for i, j in pairs]
_ = scan(pnames, lambda Z: np.column_stack([Z[:, i] * Z[:, j] for i, j in pairs]), "パラメータ対の積（符号一致）")
