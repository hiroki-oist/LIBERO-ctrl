"""Axis 5, sensor corruption: noise, blur, JPEG artefacts and motion blur applied to the
observation. The env itself is untouched.

The static components -- blur, JPEG quality, motion-blur direction -- are fixed once per
rollout and only the noise is redrawn each step, matching the physical fact that a smudged lens
or an optical defocus does not change from frame to frame.
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
        if self._c is None: raise RuntimeError("call reset(seed) first")
        return self._c(img)
