"""Axis 3, robot initial state: defined in end-effector space rather than joint space, and
solved with damped least-squares IK.

Perturbing joint angles directly would move the end effector by wildly different amounts from
task to task for the same nominal noise, so severity is defined by the end-effector
displacement in millimetres instead.
"""
import numpy as np
from . import Perturbation, PerturbSpec


class RobotPerturb(Perturbation):
    axis = "robot"

    def __init__(self, spec: PerturbSpec, *, suite: str, shape):
        # the manifest stores six scalars, not two vectors
        P = spec.params
        self.dpos = np.array([P["eef_dx_m"], P["eef_dy_m"], P["eef_dz_m"]], float)
        self.drot = np.radians(np.array([P["eef_rx_deg"], P["eef_ry_deg"], P["eef_rz_deg"]], float))
        self.ik_res_mm = None

    def transform_init_state(self, task, state):
        from ._impl.ik import perturb_robot_eef
        st, ep, er, nc = perturb_robot_eef(task, state, self.dpos, self.drot)
        self.ik_res_mm, self.ik_res_deg, self.n_clamped = ep * 1000, er, nc
        return st
