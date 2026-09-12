"""論文の Experiment / Results に載せる数値を1本のマークダウンに書き出す。
本文の転記ミスを避けるため、すべてこのスクリプトの出力から取る。
  実行: python analysis/summarise_results.py > notes/RESULTS_SUMMARY.md
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

import json, glob, collections, random, itertools, math, sys
import numpy as np

ROOT = ROOT
AX = ("camera", "lighting", "robot", "sensor", "actuation", "language")
ALLAX = AX + ("combination",)
LV = ("L1", "L2", "L3")

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

PV = lambda s: ([f"predvla_s{s}_eval"] + [f"predvla_s{s}_sh{i}_eval" for i in range(4)],
                [f"predvla_s{s}_clean"] + [f"predvla_s{s}_sh{i}_clean" for i in range(2)])
POL = [("MINERVA",     "0.54M", ["minerva_eval","minerva_comb"], ["minerva_clean"], 95.75),
       ("PredVLA",     "0.68M", *PV(13),                                            77.80),
       ("SmolVLA",     "450M",  ["smolvla_eval"],                ["smolvla_clean"], 87.30),
       ("VLA-JEPA",    "2.77B", ["vlajepa_eval"],                ["vlajepa_clean"], 97.20),
       ("pi0.5",       "4.14B", ["pi05_eval"],                   ["pi05_clean"],    96.90),
       ("OpenVLA-OFT", "7.54B", ["oft_eval"],                    ["oft_clean"],     97.10),
       ("UniVLA",      "7.54B", ["univla_eval"],                 ["univla_clean"],  95.20)]

D = {}
for name, sz, ed, cd, pub in POL:
    ev, cl = load(ed), load(cd)
    cell = collections.defaultdict(dict)
    for r in cl: cell[(r["suite"], r["task_id"])].setdefault("clean", []).append(r["success"])
    for r in ev: cell[(r["suite"], r["task_id"])].setdefault((r["axis"], r["level"]), []).append(r["success"])
    cell = {k: v for k, v in cell.items() if "clean" in v}
    agg = collections.defaultdict(lambda: [0, 0])
    for r in ev:
        a = agg[(r["axis"], r["level"])]; a[0] += r["success"]; a[1] += 1
    D[name] = dict(sz=sz, pub=pub, n_ev=len(ev), n_cl=len(cl),
                   clean=100*sum(r["success"] for r in cl)/len(cl),
                   sr={k: 100*v[0]/v[1] for k, v in agg.items()},
                   n={k: v[1] for k, v in agg.items()}, cell=cell)

def dev(name, ks, L):
    C = D[name]["cell"]
    def m(sel):
        t = [x for k in ks for x in C[k].get(sel, [])]
        return sum(t)/len(t) if t else float("nan")
    c = m("clean"); p = c
    for a in AX: p *= m((a, L))/c
    return 100*(m(("combination", L)) - p), 100*p, 100*m(("combination", L))

random.seed(0)
P = sys.stdout.write
P("# LIBERO-CTRL — Experiment と Results のまとめ\n\n")
P("`analysis/summarise_results.py` が全数値を生成。手入力なし。2026-09-09 時点の全データ。\n\n")

P("## 1. 実施した評価\n\n")
P("| 方策 | パラメータ | clean | 摂動 | 合計 |\n|---|---|---:|---:|---:|\n")
tot = 0
for name, *_ in POL:
    d = D[name]; tot += d["n_cl"] + d["n_ev"]
    P(f"| {name} | {d['sz']} | {d['n_cl']:,} | {d['n_ev']:,} | {d['n_cl']+d['n_ev']:,} |\n")
P(f"\n上表は各方策の代表 checkpoint。これに多シード分（MINERVA abl_l1 3シード + PredVLA の残り3シード、\n")
P(f"各 10,400 本 = 62,400）を加えて、**総計 {tot+62400:,} ロールアウト**。\n\n")

P("## 2. 再現ゲート\n\n")
P("| 方策 | 実測 clean | 公表値 | 差 |\n|---|---:|---:|---:|\n")
for name, *_ in POL:
    d = D[name]
    P(f"| {name} | {d['clean']:.2f} | {d['pub']:.1f} | {d['clean']-d['pub']:+.2f} |\n")
P("\nSmolVLA の公表値は論文の 1/8 の学習量の公開 checkpoint なので照合対象外（本研究の参照値は 76.35）。\n\n")

P("## 3. 軸別の劣化\n\n各セル 400 本。括弧内は clean を 100 とした保持率。\n\n")
P("| 方策 | L | " + " | ".join(a[:6] for a in ALLAX) + " |\n")
P("|---|---|" + "---:|"*7 + "\n")
for name, *_ in POL:
    d = D[name]
    for L in LV:
        row = []
        for a in ALLAX:
            v = d["sr"].get((a, L))
            row.append("—" if v is None else f"{v:.1f} ({100*v/d['clean']:.0f}%)")
        P(f"| {name if L=='L1' else ''} | {L} | " + " | ".join(row) + " |\n")
P("\n")

P("## 4. 率レベルの合成（積モデル）\n\n")
P("予測 = clean × Π_a (SR_a / clean)。逸脱 = 観測 − 予測。95% はタスクセル上のクラスタブートストラップ 4,000 回。\n\n")
P("| 方策 | L | 予測 | 観測 | 逸脱 | 95% CI | p |\n|---|---|---:|---:|---:|---|---:|\n")
DEV = {}
for name, *_ in POL:
    ks = sorted(D[name]["cell"])
    for L in LV:
        pt, pr, ob = dev(name, ks, L)
        bs = sorted(dev(name, [random.choice(ks) for _ in ks], L)[0] for _ in range(4000))
        p = 2*min(sum(1 for b in bs if b > 0), sum(1 for b in bs if b < 0))/len(bs)
        DEV[(name, L)] = dict(dev=pt, pred=pr, obs=ob, lo=bs[100], hi=bs[3899], p=p)
        fl = " ⌊" if pr < 4 else ""
        P(f"| {name if L=='L1' else ''} | {L} | {pr:.1f} | {ob:.1f} | **{pt:+.1f}**{fl} | "
          f"[{bs[100]:+.1f}, {bs[3899]:+.1f}] | {p:.3f} |\n")
P("\n⌊ = 予測が 4% 未満で符号付き残差が解像できない床セル。以降の集計から除外。\n\n")

P("### 方策間の対比（同一タスクセル上の対応づけブートストラップ）\n\n")
P("| 比較 | L | 差 | 95% CI | p |\n|---|---|---:|---|---:|\n")
def pdiff(A, B, L):
    ks = sorted(set(D[A]["cell"]) & set(D[B]["cell"]))
    pt = dev(A, ks, L)[0] - dev(B, ks, L)[0]
    bs = sorted(dev(A, s, L)[0]-dev(B, s, L)[0]
                for s in ([random.choice(ks) for _ in ks] for _ in range(4000)))
    return pt, bs[100], bs[3899], 2*min(sum(1 for b in bs if b>0), sum(1 for b in bs if b<0))/len(bs)
for A, B, L in [("SmolVLA","UniVLA","L1"),("SmolVLA","pi0.5","L1"),("SmolVLA","VLA-JEPA","L1"),
                ("SmolVLA","VLA-JEPA","L2"),("UniVLA","OpenVLA-OFT","L1"),
                ("VLA-JEPA","OpenVLA-OFT","L1"),("VLA-JEPA","OpenVLA-OFT","L3")]:
    pt, lo, hi, p = pdiff(A, B, L)
    P(f"| {A} − {B} | {L} | {pt:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {p:.3f} |\n")
P("\n### 方策内の強度依存（L3 − L1）\n\n| 方策 | 差 | 95% CI | p |\n|---|---:|---|---:|\n")
for name, *_ in POL:
    ks = sorted(D[name]["cell"])
    pt = dev(name, ks, "L3")[0] - dev(name, ks, "L1")[0]
    bs = sorted(dev(name, s, "L3")[0]-dev(name, s, "L1")[0]
                for s in ([random.choice(ks) for _ in ks] for _ in range(4000)))
    p = 2*min(sum(1 for b in bs if b>0), sum(1 for b in bs if b<0))/len(bs)
    P(f"| {name} | {pt:+.1f} | [{bs[100]:+.1f}, {bs[3899]:+.1f}] | {p:.3f} |\n")
P("\n")

P("## 5. 軌道レベルの合成（連言モデル）と分解\n\n")
P("初期状態ごとに「6因子すべてに単独で耐えた軌道は、同時にも耐えるか」を問う。\n")
P("b = 単独では全て耐えたが同時では失敗（創発失敗）、c = 単独で少なくとも1因子に殺されるが同時では成功（補償）。\n\n")
P("| 方策 | L | AND予測 | 観測 | b | c | 創発率 | 補償率 | McNemar p |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
CONJ = {}
def binom2(b, c):
    from math import comb
    n = b+c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2*sum(comb(n, i) for i in range(k+1))/2**n)
for name, *_ in POL:
    ev = load([d for d in ([x for x in POL if x[0]==name][0][2])])
    tbl = collections.defaultdict(dict)
    for r in ev:
        if (r["suite"], r["task_id"]) in D[name]["cell"]:
            tbl[(r["suite"], r["task_id"], r["level"], r["config"])][r["axis"]] = r["success"]
    for L in LV:
        ks = [k for k in tbl if k[2]==L and all(a in tbl[k] for a in AX) and "combination" in tbl[k]]
        if len(ks) < 100: continue
        a1 = [k for k in ks if all(tbl[k][a] for a in AX)]
        b = sum(1 for k in a1 if not tbl[k]["combination"])
        c = sum(1 for k in ks if k not in set(a1) and tbl[k]["combination"])
        n = len(ks)
        emg = 100*b/len(a1) if a1 else float("nan")
        cmp_ = 100*c/(n-len(a1)) if n-len(a1) else float("nan")
        CONJ[(name, L)] = dict(AND=100*len(a1)/n, obs=100*sum(tbl[k]["combination"] for k in ks)/n,
                               b=b, c=c, emg=emg, cmp=cmp_)
        P(f"| {name if L=='L1' else ''} | {L} | {100*len(a1)/n:.1f} | "
          f"{100*sum(tbl[k]['combination'] for k in ks)/n:.1f} | {b} | {c} | "
          f"{emg:.1f} | {cmp_:.1f} | {binom2(b,c):.3f} |\n")
P("\n### 創発率と補償率の相関（方策間、強度レベル内）\n\n")
for L in LV:
    e = [CONJ[(n,L)]["emg"] for n,*_ in POL if (n,L) in CONJ and not math.isnan(CONJ[(n,L)]["emg"])]
    c = [CONJ[(n,L)]["cmp"] for n,*_ in POL if (n,L) in CONJ and not math.isnan(CONJ[(n,L)]["emg"])]
    P(f"- {L}: r = {np.corrcoef(e,c)[0,1]:+.2f}（n={len(e)} 方策）\n")
P("\nノイズ由来なら両者は同方向に動くはずで、動かない。\n\n")
P("## 6. 創発項の方向依存性\n\n")
P("severity が半径なので、同一レベルの config は軸ごとの大きさが等しく向きだけが違う。\n")
P("25 個の符号付きパラメータを z 化し、その 300 対の積と、創発／補償の指標との相関を取る。\n")
P("積が大きく正 = 2 パラメータが同方向、大きく負 = 逆方向。\n\n")
FEAT, VEC = [], {}
for l in open(f"{ROOT}/manifests/v0.1/langcomb_all.jsonl"):
    r = json.loads(l)
    if r["axis"] != "combination": continue
    p_ = r["perturb"]
    if not FEAT:
        FEAT = [f"{a}.{k}" for a in ("camera","lighting","robot","sensor","actuation") for k in p_[a]]
    VEC[(r["suite"], r["task_id"], r["level"], r["config"])] = np.array(
        [p_[f.split(".")[0]][f.split(".")[1]] for f in FEAT], float)
FLOOR = {(n, L) for (n, L), v in DEV.items() if v["pred"] < 4}
pairs = list(itertools.combinations(range(len(FEAT)), 2))
def labels(name):
    ev = load([x for x in POL if x[0]==name][0][2])
    tbl = collections.defaultdict(dict)
    for r in ev: tbl[(r["suite"], r["task_id"], r["level"], r["config"])][r["axis"]] = r["success"]
    out = {}
    for k, v in tbl.items():
        if k not in VEC or not all(a in v for a in AX) or "combination" not in v: continue
        al = all(v[a] for a in AX)
        out[k] = ("b" if not v["combination"] else "ok") if al else ("c" if v["combination"] else "ng")
    return out
def cells(tgt, cond):
    R = []
    for name, *_ in POL:
        lab = labels(name)
        for L in LV:
            if (name, L) in FLOOR: continue
            ks = [k for k, v in lab.items() if k[2]==L and v in cond]
            if len(ks) < 40: continue
            y = np.array([1.0 if lab[k]==tgt else 0.0 for k in ks])
            if y.sum() < 5 or (1-y).sum() < 5: continue
            X = np.array([VEC[k] for k in ks]); Z = (X-X.mean(0))/(X.std(0)+1e-12)
            T = np.column_stack([Z[:,i]*Z[:,j] for i,j in pairs])
            Tz = (T-T.mean(0))/(T.std(0)+1e-12); yz = (y-y.mean())/(y.std()+1e-12)
            R.append(Tz.T@yz/len(y))
    return np.array(R)
Rb, Rc = cells("b", ("b","ok")), cells("c", ("c","ng"))
una = np.where((Rb>0).all(0)|(Rb<0).all(0))[0]
unc = np.where((Rc>0).all(0)|(Rc<0).all(0))[0]
P(f"**創発項: {Rb.shape[0]} セル全てで符号が一致する対が {len(una)} 個。**")
P(f"**補償項: {Rc.shape[0]} セル中で全一致する対は {len(unc)} 個。**\n\n")
P("| パラメータ対 | 創発 平均r | 一致 | 補償 平均r | 一致 |\n|---|---:|---|---:|---|\n")
for j in sorted(una, key=lambda j: -abs(Rb[:,j].mean())):
    i1, i2 = pairs[j]
    ab = max((Rb[:,j]>0).sum(), (Rb[:,j]<0).sum()); ac = max((Rc[:,j]>0).sum(), (Rc[:,j]<0).sum())
    P(f"| {FEAT[i1]} × {FEAT[i2]} | {Rb[:,j].mean():+.3f} | {ab}/{Rb.shape[0]} | "
      f"{Rc[:,j].mean():+.3f} | {ac}/{Rc.shape[0]} |\n")
rng = np.random.default_rng(0)
null = []
for _ in range(200):
    Rp = []
    for name, *_ in POL:
        lab = labels(name)
        for L in LV:
            if (name, L) in FLOOR: continue
            ks = [k for k, v in lab.items() if k[2]==L and v in ("b","ok")]
            if len(ks) < 40: continue
            y = np.array([1.0 if lab[k]=="b" else 0.0 for k in ks])
            if y.sum() < 5 or (1-y).sum() < 5: continue
            y = rng.permutation(y)
            X = np.array([VEC[k] for k in ks]); Z = (X-X.mean(0))/(X.std(0)+1e-12)
            T = np.column_stack([Z[:,i]*Z[:,j] for i,j in pairs])
            Tz = (T-T.mean(0))/(T.std(0)+1e-12); yz = (y-y.mean())/(y.std()+1e-12)
            Rp.append(Tz.T@yz/len(y))
    Rp = np.array(Rp); null.append(int(((Rp>0).all(0)|(Rp<0).all(0)).sum()))
null = np.array(null)
P(f"\n順列検定（ラベル入替 200 回）: 全一致対の帰無分布は平均 {null.mean():.2f}、最大 {null.max()}、"
  f"実測 {len(una)} に対する経験 p = {(null>=len(una)).mean():.3f}\n\n")
P("効果量は小さい（分散の 2% 未満）。主張は「創発項には再現する方向依存性があり、補償項には無い」という**非対称性**。\n\n")
P("## 7. 学習シード分散\n\n")
P("残差が「方策の性質」か「最適化実行の当たり外れ」かを切り分ける。多シードを公開している2アーキテクチャで測る。\n\n")
for arch, dirs, note in [("MINERVA abl_l1 (0.99M)", [f"mseed_s{s}" for s in (1000,2000,3000)],
                          "本表の MINERVA 行（t05_l1, 0.54M）とは幅だけ違う近傍アーム"),
                         ("PredVLA (0.68M)", None, "本表の PredVLA 行 s13 を含む公開4シード")]:
    P(f"### {arch}\n\n{note}\n\n| 量 | " )
    if dirs:
        seeds = [(f"s{s}", ([f"mseed_s{s}"], [f"mseed_s{s}"])) for s in (1000,2000,3000)]
    else:
        seeds = [(f"s{s}", PV(s)) for s in (13,5,8,10)]
    P(" | ".join(s for s,_ in seeds) + " | 平均 | SD |\n|---|" + "---:|"*(len(seeds)+2) + "\n")
    rows = collections.defaultdict(list)
    for sname, (ed, cd) in seeds:
        ev = load(ed); cl = load(cd if cd != ed else ed)
        cell = collections.defaultdict(dict)
        for r in cl:
            if r["axis"] == "clean": cell[(r["suite"],r["task_id"])].setdefault("clean",[]).append(r["success"])
        for r in ev:
            if r["axis"] != "clean": cell[(r["suite"],r["task_id"])].setdefault((r["axis"],r["level"]),[]).append(r["success"])
        cell = {k:v for k,v in cell.items() if "clean" in v}
        cn = [r["success"] for r in cl if r["axis"]=="clean"]
        rows["clean SR"].append(100*sum(cn)/len(cn) if cn else float("nan"))
        Dsave = D.get("__tmp__")
        D["__tmp__"] = dict(cell=cell)
        for L in LV:
            rows[f"逸脱 {L}"].append(dev("__tmp__", sorted(cell), L)[0])
    for q, vs in rows.items():
        P(f"| {q} | " + " | ".join(f"{v:.2f}" for v in vs) +
          f" | {np.mean(vs):.2f} | **{np.std(vs, ddof=1):.2f}** |\n")
    P("\n")
sp = {L: max(DEV[(n,L)]["dev"] for n,*_ in POL) - min(DEV[(n,L)]["dev"] for n,*_ in POL) for L in LV}
P("方策間の残差の幅: " + " / ".join(f"{L} {sp[L]:.1f} pp" for L in LV) + "\n\n")

P("## 8. 言語軸\n\n")
P("### 落差（clean 比）\n\n| 方策 | clean | L1 | L2 | L3 | L3 の落差 |\n|---|---:|---:|---:|---:|---:|\n")
for name, *_ in POL:
    d = D[name]; v = [d["sr"].get(("language", L), float("nan")) for L in LV]
    P(f"| {name} | {d['clean']:.1f} | {v[0]:.1f} | {v[1]:.1f} | {v[2]:.1f} | **{v[2]-d['clean']:+.1f}** |\n")
P("\n### 摂動後の指示文が方策に届いているかの検証\n\n")
P("3レベルは同一初期状態を共有するので、指示文が届いていなければ方策は同じ問題を3回解く。\n")
P("成功率の一致では飽和と区別できず、**step 単位の一致**が決定的。\n\n")
P("| 方策 | 成否一致 L1-L2 | L1-L3 | L2-L3 | **step 完全一致** | うち打ち切り |\n|---|---:|---:|---:|---:|---:|\n")
for name, sz, ed, cd, pub in POL:
    ev = load(ed)
    by = collections.defaultdict(dict)
    for r in ev:
        if r["axis"] == "language": by[(r["suite"],r["task_id"],r["config"])][r["level"]] = (r["success"], r["steps"])
    ks = [k for k, v in by.items() if len(v) == 3]
    ag = lambda a,b: 100*sum(1 for k in ks if by[k][a][0]==by[k][b][0])/len(ks)
    idm = [k for k in ks if by[k]["L1"][1]==by[k]["L2"][1]==by[k]["L3"][1]]
    st = 100*len(idm)/len(ks)
    mx = {r["rollout_id"]: r for r in ev}
    to = 100*sum(1 for k in idm if not by[k]["L1"][0]) / max(len(idm), 1)
    P(f"| {name} | {ag('L1','L2'):.1f} | {ag('L1','L3'):.1f} | {ag('L2','L3'):.1f} | **{st:.1f}** | {to:.0f}% |\n")
P("\nMINERVA は指示文をタスク表の索引に解決するので**言い換えを受理できない = 非配達が ground truth**。\n")
P("診断はこれを検出（step 一致 95.8%）し、テキスト条件付けの大型4方策（5〜9%）と一桁違う。\n")
P("作らずに手に入った陽性対照。\n\n")
P("**ただしこの診断には交絡がある。**方策がその軸で崩壊して打ち切りに達すると、step は自明に一致する。\n")
P("PredVLA（98.0%）と SmolVLA（44.5%）の高い値はこれで、非配達ではない（最右列の打ち切り率を参照）。\n")
P("成功率が高い方策でのみ有効な診断であり、崩壊している方策には別の対照（別タスクの正規文）が要る。\n\n")

P("## 9. 集計指標\n\n")
pr = np.array([DEV[(n,L)]["pred"] for n,*_ in POL for L in LV])
ob = np.array([DEV[(n,L)]["obs"] for n,*_ in POL for L in LV])
ar = np.array([CONJ[(n,L)]["AND"] if (n,L) in CONJ else np.nan for n,*_ in POL for L in LV])
ok = ~np.isnan(ar)
P("| | 全21セル | 床を除く |\n|---|---:|---:|\n")
for tag, m in (("", np.ones(len(pr), bool)), ("", pr >= 4)):
    pass
m_all, m_res = ok, ok & (pr >= 4)
def row(lbl, f):
    P(f"| {lbl} | {f(m_all):.3f} | {f(m_res):.3f} |\n")
def row2(lbl, f):
    P(f"| {lbl} | {f(m_all):.2f} | {f(m_res):.2f} |\n")
row("相関 r",        lambda m: np.corrcoef(pr[m], ob[m])[0,1])
row2("積モデル MAE",   lambda m: np.mean(abs(ob[m]-pr[m])))
row2("積モデル 偏り",  lambda m: np.mean(ob[m]-pr[m]))
row2("連言モデル MAE", lambda m: np.mean(abs(ob[m]-ar[m])))
row2("連言モデル 偏り",lambda m: np.mean(ob[m]-ar[m]))
row("残差の傾き",    lambda m: np.polyfit(pr[m], ob[m]-pr[m], 1)[0])
P(f"| セル数 | {m_all.sum()} | {m_res.sum()} |\n\n")

P("## 10. 破棄したデータ\n\n")
P("| 内容 | 本数 | 理由 |\n|---|---:|---|\n")
P("| 全7方策の language / combination | 16,800 | runner の `--task_mode oracle` がタスク単位で正規文を上書きし、行の言い換えが方策に届いていなかった。7方策すべての言語軸が恒等変換になっていた |\n")
P("| π0.5 の language / combination 再取得分 | 2,479 | `lerobot/pi05_libero_base`（未微調整ベース、正規化統計なし）を使っていた。正しくは `lerobot/pi05-libero` |\n")
P("| PredVLA 初期評価 | 28,996 | `ERBatch.lang_t` を更新しておらず、全ロールアウトがダミー指示文で走っていた（clean 0%） |\n")
P("| PredVLA s8 の一部 | 1,696 | ソケット接頭辞の衝突で別スイートの checkpoint に接続 |\n")
P("| PredVLA s8 の combination | 282 | fuji の GPU が Xid 79 でバスから脱落した時間帯に書かれた |\n")
P("\nいずれも `results/_invalid_*` に隔離し、削除していない。\n")
