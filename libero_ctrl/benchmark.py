"""LIBERO-CTRL の drop-in API。**既存の LIBERO 評価ループを2行変えるだけで使える。**

採用障壁を下げることが設計要件である。実装が面倒なベンチマークは使われない。

  # 既存の LIBERO
  from libero.libero import benchmark
  bm = benchmark.get_benchmark_dict()["libero_spatial"]()
  task = bm.get_task(i)
  env = OffScreenRenderEnv(bddl_file_name=..., camera_heights=256, camera_widths=256)
  env.reset(); env.set_init_state(init_states[j])
  for t in range(max_steps):
      obs, r, done, info = env.step(policy(obs, task.language))

  # LIBERO-CTRL（変更は import と env 生成の2行だけ）
  from libero_ctrl import benchmark
  bm = benchmark.get_benchmark_dict()["libero_ctrl_spatial"]()
  task = bm.get_task(i)                      # i は (task x axis x level x config) を走る
  env = bm.make_env(i, camera_heights=256, camera_widths=256)
  env.reset(); env.set_init_state(bm.get_init_state(i))
  for t in range(task.max_steps):
      obs, r, done, info = env.step(policy(obs, task.language))

**なぜ bddl の差し替えでは足りないか**: LIBERO-Plus は摂動を bddl（シーン定義）で表現できたが、
LIBERO-CTRL の sensor（観測の劣化）と actuation（行動の系統誤差）は bddl では原理的に書けない。
そこで env のラッパーが内側で全部を面倒みる:

  camera / lighting -> reset 後に sim.model.* を上書き
  robot             -> set_init_state に渡す状態を IK で作り替える
  sensor            -> step() が返す観測を劣化させる
  actuation         -> step() に渡された action を変換する
  language          -> task.language に摂動後の指示文が入っている
  什器配置          -> env 構築と reset の直前にシードを固定（§ LIBERO の再現性の穴）
"""
from __future__ import annotations
import json, os, sys
from dataclasses import dataclass
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

MANIFEST_DIR = os.environ.get("LIBERO_CTRL_MANIFEST", os.path.join(_ROOT, "manifests", "v0.1"))
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SPLITS = {"clean": "rollouts_clean.jsonl", "eval": "rollouts_eval.jsonl",
          "finetune": "rollouts_finetune.jsonl"}


@dataclass(frozen=True)
class CtrlTask:
    """LIBERO の Task と同じ気持ちで使える。`language` と `max_steps` が主。"""
    rollout_id: str
    language: str
    max_steps: int
    num_steps_wait: int
    suite: str
    task_id: int
    axis: str
    level: str
    config: int | None
    init_id: int
    init_slot: int
    split: str
    bddl_file: str
    problem_folder: str
    perturb: dict
    variant_id: str | None = None

    @property
    def seed(self) -> int:
        """方策内部の乱数に使う種。rollout_id から決定論的に決まる。"""
        from .manifest import rollout_seed
        return rollout_seed(self.rollout_id)

    # LIBERO の Task 互換のための別名
    @property
    def name(self) -> str: return self.rollout_id
    @property
    def init_states_file(self) -> str: return self.bddl_file.replace(".bddl", ".pruned_init")


class CtrlEnv:
    """LIBERO の OffScreenRenderEnv と同じインタフェースのラッパー。

    1 プロセスにつき1つだけ開くこと（複数の LIBERO env を同一プロセスで開くと
    EGL のレンダリングコンテキストが混ざって描画が壊れる）。
    """

    def __init__(self, task: CtrlTask, *, camera_heights=256, camera_widths=256,
                 env_seed: int | None = None, **kwargs):
        from ._task import Task as _T, seed_env
        self._seed = env_seed if env_seed is not None else _default_seed()
        seed_env(self._seed)                       # ★構築の直前
        self._t = _T(task.suite, task.task_id, H=camera_heights, W=camera_widths, seed=self._seed)
        self.res = (camera_heights, camera_widths)
        self._p = None
        self.configure(task)

    # ---- LIBERO 互換のプロパティ
    @property
    def env(self): return self._t.env.env
    @property
    def sim(self): return self._t.sim
    @property
    def task(self) -> CtrlTask: return self._task

    def configure(self, task: CtrlTask) -> None:
        """条件を差し替える。同じ (suite, task_id) の別条件に使い回せる。"""
        from .perturb import PerturbSpec, build
        if task.suite != self._t.suite_name or task.task_id != self._t.ti:
            raise ValueError("別タスクには使い回せない。新しい env を作ること。")
        self._task = task
        spec = PerturbSpec(axis=task.axis, level=task.level, config=task.config,
                           params=dict(task.perturb or {}), rollout_id=task.rollout_id)
        self._p = build(spec, suite=task.suite, shape=self.res)
        self._p.reset(task.seed)

    def reset(self):
        """LIBERO と同じく full reset。什器配置を固定するため直前にシードを張る。"""
        from .env import env_reset
        env_reset(self._t)
        self._t.reset_model()
        self._p.apply_model(self._t)               # camera / lighting
        obs = self.env._get_observations()
        return self._wrap_obs(obs)

    def set_init_state(self, state):
        """robot 軸はここで効く（IK で EEF をずらした状態にする）。"""
        st = self._p.transform_init_state(self._t, np.asarray(state))
        self._t.env.set_init_state(st)
        self._t.sim.forward()
        return self._wrap_obs(self.env._get_observations())

    def step(self, action):
        """actuation 軸を action に、sensor 軸を観測に適用する。"""
        a = self._p.transform_action(np.asarray(action, float))
        self.env.done = False
        obs, r, done, info = self._t.env.step(a)
        return self._wrap_obs(obs), r, done, info

    def check_success(self) -> bool:
        for fn in ("_check_success", "check_success"):
            f = getattr(self.env, fn, None)
            if callable(f) and f(): return True
        return False

    def close(self):
        from .env import close_task
        close_task(self._t)

    # ---- 内部
    def _wrap_obs(self, obs):
        """画像は robosuite の生の向きのまま返す（LIBERO 本家と同じ）。
        向きの規約はモデルごとに違うので、ここでは決め打ちしない。"""
        # ★ sensor 軸だけでなく combination 軸（Composite の中に SensorPerturb が
        #    入る）でも効かせる必要がある。無摂動時 transform_obs は恒等なので
        #    無条件に通してよい。
        out = dict(obs)
        for k in ("agentview_image", "robot0_eye_in_hand_image"):
            if k in out: out[k] = self._p.transform_obs(out[k])
        return out


def _manifest():
    return json.load(open(os.path.join(MANIFEST_DIR, "manifest.json")))


def _default_seed() -> int:
    from .manifest import env_seed
    return env_seed()


class CtrlBenchmark:
    """1 suite ぶんの条件列。`get_task(i)` / `get_init_state(i)` / `make_env(i)`。"""

    def __init__(self, suite: str, split: str = "eval"):
        if suite not in SUITES: raise ValueError(f"未知の suite: {suite}")
        if split not in SPLITS: raise ValueError(f"未知の split: {split}")
        self.suite, self.split = suite, split
        self._bddl = _bddl_map(suite)
        self.rows = [r for r in _read(os.path.join(MANIFEST_DIR, SPLITS[split]))
                     if r["suite"] == suite]
        self._init = _init_states(suite)

    # ---- LIBERO 互換
    @property
    def n_tasks(self) -> int: return len(self.rows)
    def __len__(self) -> int: return len(self.rows)

    def get_task(self, i: int) -> CtrlTask:
        r = self.rows[i]
        b, pf = self._bddl[r["task_id"]]
        return CtrlTask(rollout_id=r["rollout_id"], language=r["language"],
                        max_steps=r["max_steps"], num_steps_wait=r["num_steps_wait"],
                        suite=r["suite"], task_id=r["task_id"], axis=r["axis"],
                        level=r["level"], config=r.get("config"), init_id=r["init_id"],
                        init_slot=r["init_slot"], split=r["split"], bddl_file=b,
                        problem_folder=pf, perturb=r.get("perturb") or {},
                        variant_id=r.get("variant_id"))

    def get_init_state(self, i: int) -> np.ndarray:
        return np.asarray(self._init[self.rows[i]["task_id"]][self.rows[i]["init_id"]])

    def get_task_init_states(self, i: int) -> np.ndarray:
        return self.get_init_state(i)[None]

    def make_env(self, i: int, **kwargs) -> CtrlEnv:
        return CtrlEnv(self.get_task(i), **kwargs)

    def indices(self, *, axis=None, level=None, task_id=None) -> list[int]:
        """軸やレベルで絞り込む。全 9,200 を回さず一部だけ試すとき用。"""
        out = []
        for i, r in enumerate(self.rows):
            if axis is not None and r["axis"] != axis: continue
            if level is not None and r["level"] != level: continue
            if task_id is not None and r["task_id"] != task_id: continue
            out.append(i)
        return out


def _read(path):
    return [json.loads(l) for l in open(path)]


def _bddl_map(suite):
    from libero.libero import benchmark as _b
    bm = _b.get_benchmark_dict()[suite]()
    return {ti: (bm.get_task(ti).bddl_file, bm.get_task(ti).problem_folder) for ti in range(10)}


def _init_states(suite):
    import torch
    from libero.libero import benchmark as _b, get_libero_path
    bm = _b.get_benchmark_dict()[suite]()
    out = {}
    for ti in range(10):
        t = bm.get_task(ti)
        out[ti] = np.asarray(torch.load(os.path.join(get_libero_path("init_states"),
                                                     t.problem_folder, t.init_states_file),
                                        weights_only=False))
    return out


def get_benchmark_dict(split: str = "eval"):
    """LIBERO の `benchmark.get_benchmark_dict()` と同じ使い方。

      bm = get_benchmark_dict()["libero_ctrl_spatial"]()
    """
    return {f"libero_ctrl_{s.replace('libero_', '')}": (lambda s=s: CtrlBenchmark(s, split))
            for s in SUITES}


def get_benchmark(suite: str, split: str = "eval") -> CtrlBenchmark:
    return CtrlBenchmark(suite if suite.startswith("libero_") else f"libero_{suite}", split)
