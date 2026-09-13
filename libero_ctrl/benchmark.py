"""The drop-in API: an existing LIBERO evaluation loop adopts LIBERO-CTRL by changing two lines.

Keeping that cost at two lines is a design requirement, not a nicety. A benchmark that is
troublesome to wire up does not get used.

  # existing LIBERO
  from libero.libero import benchmark
  bm = benchmark.get_benchmark_dict()["libero_spatial"]()
  task = bm.get_task(i)
  env = OffScreenRenderEnv(bddl_file_name=..., camera_heights=256, camera_widths=256)
  env.reset(); env.set_init_state(init_states[j])
  for t in range(max_steps):
      obs, r, done, info = env.step(policy(obs, task.language))

  # LIBERO-CTRL (the import and the env construction are the only changes)
  from libero_ctrl import benchmark
  bm = benchmark.get_benchmark_dict()["libero_ctrl_spatial"]()
  task = bm.get_task(i)                      # i runs over (task x axis x level x config)
  env = bm.make_env(i, camera_heights=256, camera_widths=256)
  env.reset(); env.set_init_state(bm.get_init_state(i))
  for t in range(task.max_steps):
      obs, r, done, info = env.step(policy(obs, task.language))

Why shipping perturbed BDDL files is not enough: LIBERO-Plus could express its perturbations
as scene definitions, but two of ours cannot be written in BDDL even in principle. `sensor`
degrades the observation after rendering, and `actuation` perturbs the action on its way to the
controller. So the env wrapper takes care of all of it from the inside:

  camera / lighting -> overwrite sim.model.* after reset
  robot             -> rebuild the state handed to set_init_state, by IK
  sensor            -> degrade the observation returned by step()
  actuation         -> transform the action passed to step()
  language          -> task.language already holds the perturbed instruction
  fixture placement -> seed immediately before env construction and before reset
                       (this is the reproducibility hole LIBERO leaves open)
"""
from __future__ import annotations
import json, os, sys
from dataclasses import dataclass
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

MANIFEST_DIR = os.environ.get("LIBERO_CTRL_MANIFEST", os.path.join(_ROOT, "manifests", "v0.1"))
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SPLITS = {"clean": "rollouts_clean.jsonl", "eval": "rollouts_eval.jsonl"}


@dataclass(frozen=True)
class CtrlTask:
    """Used the way LIBERO's own Task is used; `language` and `max_steps` are what matter."""
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
        """Seed for the policy's own randomness, derived deterministically from rollout_id."""
        from .manifest import rollout_seed
        return rollout_seed(self.rollout_id)

    # aliases for LIBERO Task compatibility
    @property
    def name(self) -> str: return self.rollout_id
    @property
    def init_states_file(self) -> str: return self.bddl_file.replace(".bddl", ".pruned_init")


class CtrlEnv:
    """A wrapper with the same interface as LIBERO's OffScreenRenderEnv.

    Open exactly one per process
    (two live LIBERO envs in one process mix up their EGL rendering contexts and the renders
    come out wrong).
    """

    def __init__(self, task: CtrlTask, *, camera_heights=256, camera_widths=256,
                 env_seed: int | None = None, **kwargs):
        from ._task import Task as _T, seed_env
        self._seed = env_seed if env_seed is not None else _default_seed()
        seed_env(self._seed)                       # immediately before construction
        self._t = _T(task.suite, task.task_id, H=camera_heights, W=camera_widths, seed=self._seed)
        self.res = (camera_heights, camera_widths)
        self._p = None
        self.configure(task)

    # ---- LIBERO-compatible properties
    @property
    def env(self): return self._t.env.env
    @property
    def sim(self): return self._t.sim
    @property
    def task(self) -> CtrlTask: return self._task

    def configure(self, task: CtrlTask) -> None:
        """Swap in another condition. Reusable across conditions of the same (suite, task_id)."""
        from .perturb import PerturbSpec, build
        if task.suite != self._t.suite_name or task.task_id != self._t.ti:
            raise ValueError("cannot be reused for a different task; construct a new env")
        self._task = task
        spec = PerturbSpec(axis=task.axis, level=task.level, config=task.config,
                           params=dict(task.perturb or {}), rollout_id=task.rollout_id)
        self._p = build(spec, suite=task.suite, shape=self.res)
        self._p.reset(task.seed)

    def reset(self):
        """A full reset, as LIBERO does. The seed is applied just before it, so that the
        fixtures land in the same place."""
        from .env import env_reset
        env_reset(self._t)
        self._t.reset_model()
        self._p.apply_model(self._t)               # camera / lighting
        obs = self.env._get_observations()
        return self._wrap_obs(obs)

    def set_init_state(self, state):
        """This is where the robot axis applies: IK displaces the end effector."""
        st = self._p.transform_init_state(self._t, np.asarray(state))
        self._t.env.set_init_state(st)
        self._t.sim.forward()
        return self._wrap_obs(self.env._get_observations())

    def step(self, action):
        """Applies the actuation axis to the action and the sensor axis to the observation."""
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

    # ---- internals
    def _wrap_obs(self, obs):
        """Images come back in robosuite's raw orientation, as in LIBERO itself. The
        convention differs between released checkpoints, so it is not decided here."""
        # This has to fire on the combination axis too, where the SensorPerturb sits inside a
        # Composite -- not only when the perturbation *is* a SensorPerturb. transform_obs is
        # the identity when there is nothing to apply, so it is safe to always go through it.
        out = dict(obs)
        for k in ("agentview_image", "robot0_eye_in_hand_image"):
            if k in out: out[k] = self._p.transform_obs(out[k])
        return out




def _default_seed() -> int:
    from .manifest import env_seed
    return env_seed()


class CtrlBenchmark:
    """The conditions of one suite. `get_task(i)` / `get_init_state(i)` / `make_env(i)`."""

    def __init__(self, suite: str, split: str = "eval"):
        if suite not in SUITES: raise ValueError(f"unknown suite: {suite}")
        if split not in SPLITS: raise ValueError(f"unknown split: {split}")
        self.suite, self.split = suite, split
        self._bddl = _bddl_map(suite)
        self.rows = [r for r in _read(os.path.join(MANIFEST_DIR, SPLITS[split]))
                     if r["suite"] == suite]
        self._init = _init_states(suite)

    # ---- LIBERO compatibility
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
        """LIBERO returns an array of initial states per task; here a row fixes exactly one,
        so this returns it as a length-1 array. Present for drop-in compatibility."""
        return self.get_init_state(i)[None]

    def make_env(self, i: int, **kwargs) -> CtrlEnv:
        return CtrlEnv(self.get_task(i), **kwargs)

    def indices(self, *, axis=None, level=None, task_id=None) -> list[int]:
        """Narrow by axis, level or task, so that a subset can be tried without running all
        of them."""
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
    """Used exactly like LIBERO's own `benchmark.get_benchmark_dict()`.

      bm = get_benchmark_dict()["libero_ctrl_spatial"]()
    """
    return {f"libero_ctrl_{s.replace('libero_', '')}": (lambda s=s: CtrlBenchmark(s, split))
            for s in SUITES}


def get_benchmark(suite: str, split: str = "eval") -> CtrlBenchmark:
    return CtrlBenchmark(suite if suite.startswith("libero_") else f"libero_{suite}", split)
