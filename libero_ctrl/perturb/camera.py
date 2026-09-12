"""軸1 Camera。agentview を注視点まわりの球面座標で動かす。手先カメラは触らない。"""
import numpy as np
from . import Perturbation, PerturbSpec

KEYS = ["d_azimuth_deg", "d_elevation_deg", "d_distance_m", "d_lookat_x_m", "d_lookat_y_m"]


class CameraPerturb(Perturbation):
    axis = "camera"

    def __init__(self, spec: PerturbSpec, *, suite: str, shape):
        self.d = np.array([float(spec.params[k]) for k in KEYS])

    def apply_model(self, task) -> None:
        # 注視点は「カメラ光軸上で可動物体の重心に最も近い点」（Task が構築時に決めている）。
        # z=0.90 平面との交点だと arena ごとにテーブル高さが違い、交点がカメラ後方に出る。
        task.set_camera(self.d)
