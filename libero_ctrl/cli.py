"""The libero-ctrl command. One axis, one level, or everything -- all from the same manifest.

  libero-ctrl run --policy mymod:MyPolicy --axis camera --level L2
  libero-ctrl run --policy mymod:MyPolicy --split eval          # 8,400 rollouts
  libero-ctrl gate --results out/                               # reproduction gate
"""
from __future__ import annotations
import argparse, importlib, json, os, sys, time
from collections import defaultdict


def _load_policy(spec: str, kw: list[str] | None = None):
    """Import 'package.module:ClassName' and instantiate it.

    For a policy that runs in its own process (a model whose dependencies cannot coexist with
    the env):
      --policy libero_ctrl.policy.remote:RemotePolicy --policy-kw sock_path=/tmp/oft_0.sock
    """
    if ":" not in spec:
        raise SystemExit("--policy must be given as 'module:Class'")
    mod, cls = spec.split(":", 1)
    kwargs = {}
    for item in kw or []:
        if "=" not in item: raise SystemExit(f"--policy-kw must be k=v: {item}")
        k, v = item.split("=", 1)
        kwargs[k] = v
    return getattr(importlib.import_module(mod), cls)(**kwargs)


def _done_ids(out_dir: str) -> set:
    """The rollout ids already written, so that the same command resumes after an interruption."""
    got = set()
    if not os.path.isdir(out_dir): return got
    for fn in os.listdir(out_dir):
        if not fn.endswith(".jsonl"): continue
        with open(os.path.join(out_dir, fn)) as f:
            for l in f:
                try: got.add(json.loads(l)["rollout_id"])
                except Exception: pass
    return got


def _subsample(rows, frac):
    """A random fraction of the design, drawn as whole paired units.

    The unit is what the paired analysis needs to stay together: one `(suite, task, level,
    config)` carries the six single-axis rollouts and the simultaneous one, all on the same
    initial state, and the decomposition of Section 4 is computed across that set. Drawing
    rollouts independently would give the same count and make S_conj -- "did this initial state
    survive every axis on its own" -- uncomputable. Nominal rows have no such structure and are
    drawn one at a time.

    Units are stratified by (level, suite) so every level and every suite keeps its share.

    Deliberately unseeded: every run draws a different subset, so two runs of the reduced
    benchmark are two independent samples of the same design rather than the same rollouts
    twice. The policy seed of a drawn rollout is still crc32(rollout_id), so a rollout that
    appears in both is the same rollout.
    """
    import random
    units = defaultdict(list)
    for r in rows:
        key = ((r["suite"], r["task_id"], r["level"], r["config"]) if r["axis"] != "clean"
               else (r["suite"], r["task_id"], r["level"], r["init_id"]))
        units[key].append(r)
    strata = defaultdict(list)
    for key in units:
        strata[(key[2], key[0])].append(key)          # (level, suite)
    out = []
    for stratum in sorted(strata):
        keys = strata[stratum]
        exact = len(keys) * frac
        if abs(exact - round(exact)) > 1e-9:
            raise SystemExit(
                f"--sample {frac} does not divide the design evenly: the {stratum} stratum holds "
                f"{len(keys)} of them and {frac} of that is {exact:.3f}.\n"
                f"  Use a fraction 1/N with N a divisor of {len(keys)}.")
        n = max(1, int(round(exact)))
        for key in random.sample(keys, n):
            out += units[key]
    return sorted(out, key=lambda r: r["rollout_id"])


def cmd_run(a):
    from .manifest import iter_rows
    from .env import make_task, close_task
    from .rollout import run_rollout

    from .manifest import env_seed
    axis = None if a.axis in (None, "all") else a.axis.split(",")
    rows = [r for r in iter_rows(a.split, path=a.rows, axis=axis, level=a.level,
                                 suite=(a.suite.split(",") if a.suite else None),
                                 task_id=([int(x) for x in a.task.split(",")] if a.task else None))]
    if a.limit: rows = rows[:a.limit]
    if a.sample: rows = _subsample(rows, a.sample)
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        cells = sorted({(r["suite"], r["task_id"]) for r in rows})
        mine = {c for k, c in enumerate(cells) if k % n == i}
        rows = [r for r in rows if (r["suite"], r["task_id"]) in mine]
    os.makedirs(a.out, exist_ok=True)
    done = _done_ids(a.out)
    rows = [r for r in rows if r["rollout_id"] not in done]
    if not rows:
        print(f"already complete ({len(done)} rollouts)"); return
    by_task = defaultdict(list)
    for r in rows: by_task[(r["suite"], r["task_id"])].append(r)

    policy = _load_policy(a.policy, a.policy_kw)
    # One progress line every 5% of the run, so a 100-rollout sample reports as often as the
    # 8,400-rollout split does. LIBERO prints "using task orders ..." once per env it builds,
    # which is once per (suite, task) group below, not once per rollout.
    step = max(1, min(50, len(rows) // 20))
    print(f"{len(rows)} rollouts over {len(by_task)} task cells -> {a.out}", flush=True)
    t0, n = time.time(), 0
    for (suite, tid), rs in sorted(by_task.items()):
        # env_seed comes from the manifest -- it is what pins the fixture placement.
        # Hard-coding it to 0 here silently breaks reproduction.
        task = make_task(suite, tid, res=a.res, seed=env_seed())
        path = os.path.join(a.out, f"{suite}_t{tid}.jsonl")
        try:
            with open(path, "a") as f:
                for r in rs:
                    out = run_rollout(task, r, policy, res=a.res)
                    f.write(json.dumps(out) + "\n"); f.flush(); n += 1
                    if n % step == 0 or n == len(rows):
                        el = time.time() - t0
                        left = (len(rows) - n) * el / n
                        eta = f"{left/60:.0f} min left" if left >= 90 else f"{left:.0f} s left"
                        print(f"  {n}/{len(rows)}  {100*n/len(rows):3.0f}%  "
                              f"{el/n:.1f}s each  {eta}", flush=True)
        finally:
            close_task(task)
    print(f"done: {n} rollouts in {time.time()-t0:.0f}s -> {a.out}")


def cmd_gate(a):
    """Check the nominal results against a published score: the reproduction gate."""
    import glob
    rows = [json.loads(l) for f in glob.glob(os.path.join(a.results, "*.jsonl"))
            for l in open(f)]
    rows = [r for r in rows if r.get("axis") == "clean"]
    if not rows: raise SystemExit("no nominal results here (run with --split clean first)")
    by = defaultdict(list)
    for r in rows: by[r["suite"]].append(r["success"])
    tot = sum(sum(v) for v in by.values()) / sum(len(v) for v in by.values()) * 100
    for s in sorted(by):
        print(f"  {s:16s} {100*sum(by[s])/len(by[s]):5.1f}%  (n={len(by[s])})")
    print(f"  {'aggregate':16s} {tot:5.2f}%  (n={len(rows)})")
    if a.published is not None:
        d = tot - a.published
        verdict = "PASS" if abs(d) <= a.tol else "FAIL"
        print(f"\n{d:+.2f} pt against the published {a.published:.2f}%  -> {verdict} "
              f"(tolerance +/-{a.tol})")
        if verdict == "FAIL":
            print("  Suspect the observation mapping first: image orientation, state\n"
                  "  dimensionality, number of cameras.")
            sys.exit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="libero-ctrl")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run rollouts")
    r.add_argument("--policy", required=True, help="'module:Class' with reset/act methods")
    r.add_argument("--out", default="out", help="output directory")
    r.add_argument("--split", default="eval", choices=["eval", "clean"])
    r.add_argument("--axis", default=None, help="camera / lighting / robot / sensor / actuation / language / combination / all")
    r.add_argument("--level", default=None, help="L1 / L2 / L3; omit for all levels")
    r.add_argument("--suite", default=None, help="libero_spatial,... comma separated")
    r.add_argument("--res", type=int, default=128)
    r.add_argument("--shard", default=None, help="i/N; splits by task")
    r.add_argument("--task", default=None, help="0,1,2; restrict to these task ids")
    r.add_argument("--limit", type=int, default=None, help="first N rollouts only")
    r.add_argument("--rows", default=None, metavar="PATH",
                   help="a manifest file to use instead of the split's own; "
                        "manifests/v0.1/minerva_recollect.jsonl is the eval design with the "
                        "canonical instruction, for a policy that cannot accept a paraphrase")
    r.add_argument("--sample", type=float, default=None, metavar="FRAC",
                   help="run a random FRAC of the design, drawn as whole paired units. "
                        "Unseeded: every run draws a different subset. FRAC must divide each "
                        "stratum exactly -- 1/N with N a divisor of 100. 0.05 is the reduced "
                        "benchmark, 100 nominal and 420 perturbed rollouts")
    r.add_argument("--policy-kw", action="append", default=[], metavar="K=V",
                   help="constructor argument for the policy, e.g. sock_path=/tmp/oft_0.sock")
    r.set_defaults(f=cmd_run)

    g = sub.add_parser("gate", help="check nominal results against a published score")
    g.add_argument("--results", required=True)
    g.add_argument("--published", type=float, default=None)
    g.add_argument("--tol", type=float, default=5.0, help="tolerance in points")
    g.set_defaults(f=cmd_gate)

    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
