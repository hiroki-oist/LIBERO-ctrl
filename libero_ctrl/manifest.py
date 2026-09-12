"""manifest（= 実験設計そのもの）を読むための小さな API。

manifest の 1 行が 1 rollout に対応し、行に含まれる情報だけで rollout が再現される。
実行時に乱数を引かないのが設計の要で、seed は rollout_id から決定論的に導く。
"""
from __future__ import annotations
import json, os, zlib
from typing import Iterator

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
MANIFEST_DIR = os.environ.get("LIBERO_CTRL_MANIFEST", os.path.join(_ROOT, "manifests", "v0.1"))

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
AXES = ("camera", "lighting", "robot", "sensor", "actuation", "language", "combination")
LEVELS = ("L1", "L2", "L3")
SPLITS = {"clean": "rollouts_clean.jsonl", "eval": "rollouts_eval.jsonl"}
#: severity 半径。正規化空間 Δp/σ のユークリッドノルム。
SEVERITY_RADIUS = {"L1": 2.0, "L2": 4.0, "L3": 8.0}


def severity_radius(level: str) -> float:
    """level -> 半径。L3 の単一パラメータ方向は宣言した物理範囲の端に一致する。"""
    return SEVERITY_RADIUS[level]


def rollout_seed(rollout_id: str) -> int:
    """★安定ハッシュ。Python の hash() は起動ごとに変わるので使わない。"""
    return zlib.crc32(rollout_id.encode())


def _require_manifest() -> str:
    if not os.path.isdir(MANIFEST_DIR):
        raise SystemExit(
            f"manifest が見つかりません: {MANIFEST_DIR}\n"
            "  リポジトリ直下で `pip install -e .` してください（manifest はパッケージ外に置いてあります）。\n"
            "  別の場所に置く場合は環境変数 LIBERO_CTRL_MANIFEST でディレクトリを指定してください。")
    return MANIFEST_DIR


def protocol() -> dict:
    """manifest.json の protocol ブロック（max_steps / num_steps_wait / env_seed）。"""
    with open(os.path.join(_require_manifest(), "manifest.json")) as f:
        return json.load(f)["protocol"]


def env_seed() -> int:
    """env を**構築する直前**に張るシード。★これを固定しないと什器（棚・コンロ等）の
    配置が実行ごとに変わり、set_init_state では戻らないため再現できない。"""
    return int(protocol()["env_seed"])


def load_rows(split: str = "eval") -> list[dict]:
    path = os.path.join(_require_manifest(), SPLITS[split])
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    for r in rows:
        r.setdefault("seed", rollout_seed(r["rollout_id"]))
    return rows


def iter_rows(split: str = "eval", *, axis=None, level=None, suite=None,
              task_id=None) -> Iterator[dict]:
    """条件で絞った manifest 行を返す。

      iter_rows(axis="camera", level="L2")          -> 400 行（単一軸 1 水準）
      iter_rows(level="L1")                         -> 2,800 行（全 7 条件 1 水準）
      iter_rows()                                   -> 8,400 行（フル）
      iter_rows("clean")                            -> 2,000 行（摂動なし）
    """
    def ok(r, key, want):
        if want is None: return True
        want = {want} if isinstance(want, str) else set(want)
        return r.get(key) in want
    for r in load_rows(split):
        if (ok(r, "axis", axis) and ok(r, "level", level)
                and ok(r, "suite", suite) and _ok_task(r, task_id)):
            yield r


def _ok_task(r, task_id) -> bool:
    if task_id is None: return True
    if isinstance(task_id, int): return r["task_id"] == task_id
    return r["task_id"] in set(task_id)
