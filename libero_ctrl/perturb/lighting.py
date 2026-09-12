"""軸2 Lighting。明るさ・色（暖寒/緑マゼンタ）・環境光・光源仰角。**影は出さない**。"""
import numpy as np
from . import Perturbation, PerturbSpec

KEYS = ["d_intensity_ev", "d_log2_warm", "d_tint_g", "d_log2_ambient", "d_elevation_deg"]


class LightingPerturb(Perturbation):
    axis = "lighting"

    def __init__(self, spec: PerturbSpec, *, suite: str, shape):
        self.d = np.array([float(spec.params[k]) for k in KEYS])

    def apply_model(self, task) -> None:
        task.set_light5(self.d, shadow=False)
