"""摂動の向きと創発/補償の相関を、設計に整合した 10,000 回の置換で較正する。

置換の設計:
  ・(suite,task) セル内で config を並べ替える（タスク難易度は保存）
  ・**全方策に同一の並べ替えを適用**する。config は方策間で共有されているので、
    独立にシャッフルすると configuration difficulty 由来の方策間共分散が壊れる。
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

import json, glob, collections, itertools
import numpy as np
AX=("camera","lighting","robot","sensor","actuation","language"); LV=("L1","L2","L3")
tk=collections.defaultdict(list)
# 統合済みなので追加の読み込み元は無い
def load(ds):
    o={}
    for d in ds:
        for f in glob.glob(f"{RESULTS}/{d}/*.jsonl"):
            for l in open(f): r=json.loads(l); o[r["rollout_id"]]=r
        for r in tk.get(d,[]): o[r["rollout_id"]]=r
    return list(o.values())
FEAT,VEC=[],{}
for l in open("manifests/v0.1/langcomb_all.jsonl"):
    r=json.loads(l)
    if r["axis"]!="combination": continue
    p=r["perturb"]
    if not FEAT: FEAT=[f"{a}.{k}" for a in ("camera","lighting","robot","sensor","actuation") for k in p[a]]
    VEC[(r["suite"],r["task_id"],r["level"],r["config"])]=np.array(
        [p[f.split(".")[0]][f.split(".")[1]] for f in FEAT],float)
pairs=list(itertools.combinations(range(len(FEAT)),2))
SPEC={"MINERVA":["minerva_eval","minerva_comb"],"SmolVLA":["smolvla_eval"],"VLA-JEPA":["vlajepa_eval"],
      "pi0.5":["pi05_eval"],"OpenVLA-OFT":["oft_eval"],"UniVLA":["univla_eval"],
      "PredVLA":["predvla_s13_eval"]}
FLOOR={("MINERVA","L3"),("SmolVLA","L3"),("UniVLA","L3"),("PredVLA","L1"),("PredVLA","L2"),("PredVLA","L3")}
def lab(name):
    tbl=collections.defaultdict(dict)
    for r in load(SPEC[name]): tbl[(r["suite"],r["task_id"],r["level"],r["config"])][r["axis"]]=r["success"]
    o={}
    for k,v in tbl.items():
        if k not in VEC or not all(a in v for a in AX) or "combination" not in v: continue
        al=all(v[a] for a in AX)
        o[k]=("b" if not v["combination"] else "ok") if al else ("c" if v["combination"] else "ng")
    return o
LAB={n:lab(n) for n in SPEC}
NP_=10000; rng=np.random.default_rng(0)
def run(tgt,cond):
    cells=[]
    for n in SPEC:
        for L in LV:
            if (n,L) in FLOOR: continue
            ks=sorted([k for k,v in LAB[n].items() if k[2]==L and v in cond])
            if len(ks)<40: continue
            y=np.array([1.0 if LAB[n][k]==tgt else 0.0 for k in ks])
            if y.sum()<5 or (1-y).sum()<5: continue
            X=np.array([VEC[k] for k in ks]); Z=(X-X.mean(0))/(X.std(0)+1e-12)
            T=np.column_stack([Z[:,i]*Z[:,j] for i,j in pairs])
            T=(T-T.mean(0))/(T.std(0)+1e-12)
            grp=collections.defaultdict(list)
            for idx,k in enumerate(ks): grp[(k[0],k[1])].append(idx)
            cells.append((n,L,T,y,[np.array(v) for v in grp.values()]))
    obs=np.array([ (c[2].T@((c[3]-c[3].mean())/(c[3].std()+1e-12)))/len(c[3]) for c in cells])
    una=int(((obs>0).all(0)|(obs<0).all(0)).sum())
    # 同一の並べ替えを全セルに適用（タスクセル内で config を入替）
    null=np.zeros(NP_,int)
    Yp=[np.empty((len(c[3]),NP_)) for c in cells]
    for ci,c in enumerate(cells):
        y=c[3]
        for g in c[4]:
            blk=y[g]
            for t in range(NP_):
                Yp[ci][g,t]=rng.permutation(blk)
    for t0 in range(0,NP_,500):
        sl=slice(t0,min(t0+500,NP_))
        R=[]
        for ci,c in enumerate(cells):
            Y=Yp[ci][:,sl]
            Yz=(Y-Y.mean(0))/(Y.std(0)+1e-12)
            R.append(c[2].T@Yz/len(c[3]))
        R=np.stack(R)
        null[sl]=((R>0).all(0)|(R<0).all(0)).sum(0)
    return cells,obs,una,null
cb,ob_,ua,nb=run("b",("b","ok"))
cc,oc_,uc,nc=run("c",("c","ng"))
print(f"創発 b: {len(cb)} セル、全一致 {ua} 対 / {len(pairs)}")
print(f"  帰無（10,000 置換、設計整合）: 平均 {nb.mean():.3f}  95%点 {np.percentile(nb,95):.0f}  "
      f"最大 {nb.max()}  経験 p = {(nb>=ua).mean():.4f}")
print(f"補償 c: {len(cc)} セル、全一致 {uc} 対")
print(f"  帰無: 平均 {nc.mean():.3f}  最大 {nc.max()}  経験 p = {(nc>=max(uc,1)).mean():.4f}")
if ua:
    print("\n  全一致した対:")
    for j in sorted(np.where((ob_>0).all(0)|(ob_<0).all(0))[0], key=lambda j:-abs(ob_[:,j].mean())):
        i1,i2=pairs[j]
        print(f"    {FEAT[i1]:26s} × {FEAT[i2]:24s} 平均r={ob_[:,j].mean():+.3f}  "
              f"補償側 {oc_[:,j].mean():+.3f} ({max((oc_[:,j]>0).sum(),(oc_[:,j]<0).sum())}/{len(cc)})")
