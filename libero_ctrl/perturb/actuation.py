"""軸6 Actuation Error。「そこへ動かそうとしたが系統誤差で行かない」を action 変換で作る。

ゲイン誤差・固定バイアス・座標系の軸ずれ・一次遅れ・ランダムノイズ。model には触らない。
値域は実機（Franka Panda / SO-101 級）であり得る範囲に絞ってある。
"""
from . import Perturbation, PerturbSpec


class ActuationPerturb(Perturbation):
    axis = "actuation"

    def __init__(self, spec: PerturbSpec, *, suite: str, shape):
        self.params = {k: float(v) for k, v in spec.params.items()}
        self.suite = suite
        self._e = None

    def reset(self, seed: int) -> None:
        from ._impl.actuation_model import ActuationError
        self._e = ActuationError(self.params, seed=seed, suite=self.suite)

    def transform_action(self, a):
        if self._e is None: raise RuntimeError("reset(seed) を先に呼ぶこと")
        return self._e(a)
