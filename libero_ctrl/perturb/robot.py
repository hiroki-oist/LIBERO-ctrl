"""軸3 Robot Initial State。関節空間ではなく **EEF 空間**で定義し、減衰最小二乗 IK で解く。

関節に直接ノイズを載せると同じノイズ量でもタスクによって EEF の変位が桁で変わるため、
「EEF を何 mm ずらしたか」で severity を定義する。
"""
import numpy as np
from . import Perturbation, PerturbSpec


class RobotPerturb(Perturbation):
    axis = "robot"

    def __init__(self, spec: PerturbSpec, *, suite: str, shape):
        # manifest はスカラー6個で持つ（ベクトル2個ではない）
        P = spec.params
        self.dpos = np.array([P["eef_dx_m"], P["eef_dy_m"], P["eef_dz_m"]], float)
        self.drot = np.radians(np.array([P["eef_rx_deg"], P["eef_ry_deg"], P["eef_rz_deg"]], float))
        self.ik_res_mm = None

    def transform_init_state(self, task, state):
        from ._impl.ik import perturb_robot_eef
        st, ep, er, nc = perturb_robot_eef(task, state, self.dpos, self.drot)
        self.ik_res_mm, self.ik_res_deg, self.n_clamped = ep * 1000, er, nc
        return st
