"""Axis 6, actuation error: "the policy commanded that motion and systematic error kept it
from happening", expressed as a transform on the action.

Gain error, constant bias, axis misalignment, first-order lag and random noise. The model is
untouched. The ranges are restricted to what is plausible on real hardware of the Franka Panda
or SO-101 class.
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
        if self._e is None: raise RuntimeError("call reset(seed) first")
        return self._e(a)
