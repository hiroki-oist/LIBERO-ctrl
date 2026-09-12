"""軸5 Sensor Corruption。観測の劣化（ノイズ・ぼかし・JPEG・動きぶれ）。env には触らない。

静的な要素（ぼかし・JPEG・動きぶれの向き）は rollout ごとに固定し、ノイズだけ毎ステップ引く。
レンズの汚れや光学ぼけは時間変化しないという物理に合わせる。
"""
import numpy as np
from . import Perturbation, PerturbSpec


class SensorPerturb(Perturbation):
    axis = "sensor"

    def __init__(self, spec: PerturbSpec, *, suite: str, shape):
        self.params = {k: float(v) for k, v in spec.params.items()}
        self.shape = shape
        self._c = None

    def reset(self, seed: int) -> None:
        from ._impl.sensor_model import SensorCorruption
        self._c = SensorCorruption(self.params, seed=seed, shape=self.shape)

    def transform_obs(self, img):
        if self._c is None: raise RuntimeError("reset(seed) を先に呼ぶこと")
        return self._c(img)
