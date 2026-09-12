"""摂動の適用点は4種類ある。軸ごとに「どこに効くか」が違う。

  model  : env の reset 後に sim.model.* を書き換える（camera / lighting）
  state  : set_init_state に渡す状態そのものを作り替える（robot）
  obs    : 描画された観測を後段で劣化させる（sensor）
  action : 方策が出した action を実行前に変換する（actuation）
  text   : 指示文（language。manifest の行が実体を持つので runner は渡すだけ）

Perturbation はこの4つのフックを持ち、既定では何もしない。
1 rollout につき **ちょうど1軸だけ**が非恒等になる（runner が assert する）。
"""
from dataclasses import dataclass, field
from typing import Any

AXES = ("camera", "lighting", "robot", "sensor", "actuation", "language",
        "combination", "clean")


@dataclass(frozen=True)
class PerturbSpec:
    """manifest の1行が指定する摂動。実行時に乱数を引かない。"""
    axis: str
    level: str                      # L0(clean) / L1 / L2 / L3
    config: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    rollout_id: str = ""

    def __post_init__(self):
        if self.axis not in AXES:
            raise ValueError(f"未知の軸: {self.axis}")

    @classmethod
    def from_row(cls, row: dict) -> "PerturbSpec":
        return cls(axis=row["axis"], level=row["level"], config=row.get("config"),
                   params=dict(row.get("perturb") or {}), rollout_id=row.get("rollout_id", ""))


class Perturbation:
    """既定では恒等。軸ごとに必要なフックだけを上書きする。"""
    axis = "clean"

    def apply_model(self, task) -> None: ...
    def transform_init_state(self, task, state):  return state
    def transform_obs(self, img): return img
    def transform_action(self, a): return a
    def reset(self, seed: int) -> None:
        """1 rollout の開始時。rollout 内で固定すべき乱数要素をここで決める。"""

    @property
    def is_identity(self) -> bool:
        return type(self) is Perturbation


class Composite(Perturbation):
    """★軸7 Combination。6軸を**同時に**かける。

    他の6軸は「1 rollout = 1 因子」を守るが、この軸は意図的にそれを破る。
    実運用では複数の変動が同時に起きるので、単一因子の和で予測できるかを見るための軸である。
    severity は「全軸を同じレベルにする」で定義する（各軸が半径 r なら、
    結合空間では sqrt(6)*r 相当になるので、単一軸の L3 より強い）。

    config j は **各軸の config j を束ねたもの**（camera c_j + lighting c_j + ... + language 変種 j）。
    これで config と init slot の 1:1 対応が他の軸と揃う。
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
    """spec から摂動器を作る。language と clean は恒等（指示文は行が実体を持つ）。"""
    if spec.axis == "combination":
        # params は軸名 -> その軸のパラメータ辞書
        parts = []
        for ax, pr in spec.params.items():
            if ax == "language": continue          # 指示文は row の language に入っている
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
