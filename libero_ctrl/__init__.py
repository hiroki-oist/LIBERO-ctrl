"""LIBERO-CTRL: a controlled, paired, severity-calibrated robustness benchmark for VLA policies.

Three implementations, easiest first, depending on how much evaluation code you already have.

  Implementation 1 -- write a two-method policy and let us drive:
      class MyPolicy:
          def reset(self, language: str, *, seed: int) -> None: ...
          def act(self, agentview, wrist, obs) -> np.ndarray: ...
      $ libero-ctrl run --policy mymodule:MyPolicy --axis camera --level L2

  Implementation 2 -- swap the benchmark object (2 lines changed):
      from libero_ctrl import get_benchmark_dict
      bm = get_benchmark_dict(split="eval")["libero_ctrl_spatial"]()
      env = bm.make_env(i, camera_heights=256, camera_widths=256)
      # your existing loop over bm.get_task(i) / env.step(...) keeps working

  Implementation 3 -- keep your own loop, insert five hooks:
      from libero_ctrl import PerturbSpec, build, iter_rows
      p = build(PerturbSpec.from_row(row), suite=row["suite"], shape=(128,128))
      p.reset(row["seed"])
      env.reset();  p.apply_model(env)                    # camera, lighting
      st = p.transform_init_state(env, st)                # initial pose
      img = p.transform_obs(img)                          # sensor
      a   = p.transform_action(a)                         # actuation
      #     row["language"] is what you pass to the policy # language

All three read the same manifest, so they give identical rollouts.
"""
from .benchmark import (  # noqa: F401
    CtrlBenchmark, CtrlEnv, CtrlTask, get_benchmark, get_benchmark_dict,
)
from .perturb import Perturbation, PerturbSpec, build  # noqa: F401
from .manifest import AXES, LEVELS, SUITES, iter_rows, load_rows, severity_radius  # noqa: F401

__version__ = "0.1.0"
__all__ = [
    "get_benchmark", "get_benchmark_dict", "CtrlBenchmark", "CtrlEnv", "CtrlTask",
    "PerturbSpec", "Perturbation", "build",
    "iter_rows", "load_rows", "AXES", "LEVELS", "SUITES", "severity_radius",
]
