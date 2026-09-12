"""Where each axis is applied. Four hook points, plus the instruction string.

  model  : overwrite sim.model.* after the env has been reset   (camera, lighting)
  state  : rebuild the state handed to set_init_state           (robot)
  obs    : degrade the rendered observation                     (sensor)
  action : transform the action before it reaches the env       (actuation)
  text   : the instruction itself -- the manifest row carries it, the runner only
           passes it through                                    (language)

`Perturbation` exposes those four hooks and does nothing by default. Exactly one axis is
non-identity in any single rollout, and the runner asserts it.
"""
from dataclasses import dataclass, field
from typing import Any

AXES = ("camera", "lighting", "robot", "sensor", "actuation", "language",
        "combination", "clean")


@dataclass(frozen=True)
class PerturbSpec:
    """The perturbation a manifest row asks for. Nothing is sampled at run time."""
    axis: str
    level: str                      # L0 (clean) / L1 / L2 / L3
    config: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    rollout_id: str = ""

    def __post_init__(self):
        if self.axis not in AXES:
            raise ValueError(f"unknown axis: {self.axis}")

    @classmethod
    def from_row(cls, row: dict) -> "PerturbSpec":
        return cls(axis=row["axis"], level=row["level"], config=row.get("config"),
                   params=dict(row.get("perturb") or {}), rollout_id=row.get("rollout_id", ""))


class Perturbation:
    """Identity by default; each axis overrides only the hooks it needs."""
    axis = "clean"

    def apply_model(self, task) -> None: ...
    def transform_init_state(self, task, state):  return state
    def transform_obs(self, img): return img
    def transform_action(self, a): return a
    def reset(self, seed: int) -> None:
        """Called once at the start of a rollout: fix anything that must stay fixed within it."""

    @property
    def is_identity(self) -> bool:
        return type(self) is Perturbation


class Composite(Perturbation):
    """The seventh axis: all six single axes applied *simultaneously*.

    The other six axes each hold to "one rollout, one factor". This axis deliberately breaks
    that, because in deployment several things vary at once, and the question the benchmark
    asks is whether the simultaneous outcome is predictable from the single-axis ones.

    Severity is defined as "every axis at the same level". Six axes at radius r are at
    sqrt(6)*r in the joint space, so combination L1 is already stronger than any single-axis L3.

    Config j bundles config j of every axis (camera c_j + lighting c_j + ... + paraphrase j),
    which keeps the 1:1 correspondence between config and initial-state slot that the single
    axes have.
    """
    axis = "combination"

    def __init__(self, parts: list[Perturbation]):
        self.parts = parts

    def apply_model(self, task) -> None:
        for p in self.parts: p.apply_model(task)

    def transform_init_state(self, task, state):
        for p in self.parts: state = p.transform_init_state(task, state)
        return state

    def transform_obs(self, img):
        for p in self.parts: img = p.transform_obs(img)
        return img

    def transform_action(self, a):
        for p in self.parts: a = p.transform_action(a)
        return a

    def reset(self, seed: int) -> None:
        for k, p in enumerate(self.parts): p.reset(seed + 1000 * k)

    @property
    def ik_res_mm(self):
        for p in self.parts:
            v = getattr(p, "ik_res_mm", None)
            if v is not None: return v
        return None


def build(spec: PerturbSpec, *, suite: str, shape: tuple[int, int]) -> Perturbation:
    """Build the perturbation for a spec. `language` and `clean` are identity here, because
    the instruction is carried by the manifest row rather than applied to the env."""
    if spec.axis == "combination":
        # params maps axis name -> that axis's parameter dict
        parts = []
        for ax, pr in spec.params.items():
            if ax == "language": continue          # the paraphrase is in row["language"]
            parts.append(build(PerturbSpec(axis=ax, level=spec.level, config=spec.config,
                                           params=pr, rollout_id=spec.rollout_id),
                               suite=suite, shape=shape))
        return Composite(parts)
    if spec.axis in ("clean", "language"):
        return Perturbation()
    from .camera import CameraPerturb
    from .lighting import LightingPerturb
    from .robot import RobotPerturb
    from .sensor import SensorPerturb
    from .actuation import ActuationPerturb
    cls = {"camera": CameraPerturb, "lighting": LightingPerturb, "robot": RobotPerturb,
           "sensor": SensorPerturb, "actuation": ActuationPerturb}[spec.axis]
    return cls(spec, suite=suite, shape=shape)
