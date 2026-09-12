"""libero-ctrl コマンド。軸単体・レベル別・フルを同じ manifest から回す。

  libero-ctrl run --policy mymod:MyPolicy --axis camera --level L2
  libero-ctrl run --policy mymod:MyPolicy --split eval          # 8,400 本
  libero-ctrl gate --results out/                               # 再現ゲート
"""
from __future__ import annotations
import argparse, importlib, json, os, sys, time
from collections import defaultdict


def _load_policy(spec: str, kw: list[str] | None = None):
    """'package.module:ClassName' を import して呼び出す。

    別プロセスの方策サーバを使う場合（依存が env と両立しないモデル）は
      --policy libero_ctrl.policy.remote:RemotePolicy --policy-kw sock_path=/tmp/oft_0.sock
    """
    if ":" not in spec:
        raise SystemExit("--policy は 'module:Class' の形で指定してください")
    mod, cls = spec.split(":", 1)
    kwargs = {}
    for item in kw or []:
        if "=" not in item: raise SystemExit(f"--policy-kw は k=v の形で: {item}")
        k, v = item.split("=", 1)
        kwargs[k] = v
    return getattr(importlib.import_module(mod), cls)(**kwargs)


def _done_ids(out_dir: str) -> set:
    """既に書かれた rollout_id。中断しても同じコマンドで再開できる。"""
    got = set()
    if not os.path.isdir(out_dir): return got
    for fn in os.listdir(out_dir):
        if not fn.endswith(".jsonl"): continue
        with open(os.path.join(out_dir, fn)) as f:
            for l in f:
                try: got.add(json.loads(l)["rollout_id"])
                except Exception: pass
    return got


def cmd_run(a):
    from .manifest import iter_rows
    from .env import make_task, close_task
    from .rollout import run_rollout

    from .manifest import env_seed
    axis = None if a.axis in (None, "all") else a.axis.split(",")
    rows = [r for r in iter_rows(a.split, axis=axis, level=a.level,
                                 suite=(a.suite.split(",") if a.suite else None),
                                 task_id=([int(x) for x in a.task.split(",")] if a.task else None))]
    if a.limit: rows = rows[:a.limit]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        cells = sorted({(r["suite"], r["task_id"]) for r in rows})
        mine = {c for k, c in enumerate(cells) if k % n == i}
        rows = [r for r in rows if (r["suite"], r["task_id"]) in mine]
    os.makedirs(a.out, exist_ok=True)
    done = _done_ids(a.out)
    rows = [r for r in rows if r["rollout_id"] not in done]
    if not rows:
        print(f"すべて完了済み（{len(done)} 本）"); return
    by_task = defaultdict(list)
    for r in rows: by_task[(r["suite"], r["task_id"])].append(r)

    policy = _load_policy(a.policy, a.policy_kw)
    t0, n = time.time(), 0
    for (suite, tid), rs in sorted(by_task.items()):
        # ★env_seed は manifest が決める（什器配置の固定）。ここを 0 にすると再現しない。
        task = make_task(suite, tid, res=a.res, seed=env_seed())
        path = os.path.join(a.out, f"{suite}_t{tid}.jsonl")
        try:
            with open(path, "a") as f:
                for r in rs:
                    out, _ = run_rollout(task, r, policy, res=a.res)
                    f.write(json.dumps(out) + "\n"); f.flush(); n += 1
                    if n % 50 == 0:
                        el = time.time() - t0
                        print(f"  {n}/{len(rows)}  {el/n:.1f}s/本  残り {(len(rows)-n)*el/n/60:.0f}分",
                              flush=True)
        finally:
            close_task(task)
    print(f"完了 {n} 本 / {time.time()-t0:.0f}s -> {a.out}")


def cmd_gate(a):
    """clean の結果を公表値と照合する（再現ゲート）。"""
    import glob
    rows = [json.loads(l) for f in glob.glob(os.path.join(a.results, "*.jsonl"))
            for l in open(f)]
    rows = [r for r in rows if r.get("axis") == "clean"]
    if not rows: raise SystemExit("clean の結果がありません（--split clean で回してください）")
    by = defaultdict(list)
    for r in rows: by[r["suite"]].append(r["success"])
    tot = sum(sum(v) for v in by.values()) / sum(len(v) for v in by.values()) * 100
    for s in sorted(by):
        print(f"  {s:16s} {100*sum(by[s])/len(by[s]):5.1f}%  (n={len(by[s])})")
    print(f"  {'aggregate':16s} {tot:5.2f}%  (n={len(rows)})")
    if a.published is not None:
        d = tot - a.published
        verdict = "PASS" if abs(d) <= a.tol else "FAIL"
        print(f"\n公表値 {a.published:.2f}% との差 {d:+.2f} pt  -> {verdict}（許容 ±{a.tol}）")
        if verdict == "FAIL":
            print("  ★観測の対応づけ（画像の向き・state 次元・カメラ本数）を先に疑ってください。")
            sys.exit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="libero-ctrl")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="rollout を回す")
    r.add_argument("--policy", required=True, help="'module:Class'。reset/act の2メソッドを持つこと")
    r.add_argument("--out", default="out", help="結果の出力先ディレクトリ")
    r.add_argument("--split", default="eval", choices=["eval", "clean"])
    r.add_argument("--axis", default=None, help="camera / lighting / robot / sensor / actuation / language / combination / all")
    r.add_argument("--level", default=None, help="L1 / L2 / L3。省略で全水準")
    r.add_argument("--suite", default=None, help="libero_spatial,... カンマ区切り")
    r.add_argument("--res", type=int, default=128)
    r.add_argument("--shard", default=None, help="i/N。タスク単位で分割")
    r.add_argument("--task", default=None, help="0,1,2。タスク番号で絞る（動作確認用）")
    r.add_argument("--limit", type=int, default=None, help="先頭 N 本だけ（動作確認用）")
    r.add_argument("--policy-kw", action="append", default=[], metavar="K=V",
                   help="方策のコンストラクタ引数。例 sock_path=/tmp/oft_0.sock")
    r.set_defaults(f=cmd_run)

    g = sub.add_parser("gate", help="clean 結果を公表値と照合")
    g.add_argument("--results", required=True)
    g.add_argument("--published", type=float, default=None)
    g.add_argument("--tol", type=float, default=5.0, help="許容差 [pt]")
    g.set_defaults(f=cmd_gate)

    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
