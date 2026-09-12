"""Axis 1, camera: move the agentview camera in spherical coordinates about its look-at
point. The wrist camera is left alone."""
import numpy as np
from . import Perturbation, PerturbSpec

KEYS = ["d_azimuth_deg", "d_elevation_deg", "d_distance_m", "d_lookat_x_m", "d_lookat_y_m"]


class CameraPerturb(Perturbation):
    axis = "camera"

    def __init__(self, spec: PerturbSpec, *, suite: str, shape):
        self.d = np.array([float(spec.params[k]) for k in KEYS])

    def apply_model(self, task) -> None:
        # The look-at point is the point on the optical axis closest to the centroid of the
        # movable objects, fixed by Task at construction. Intersecting a fixed z = 0.90 plane
        # instead fails, because table height differs per arena and the intersection can land
        # behind the camera.
        task.set_camera(self.d)
