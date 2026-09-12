"""90M プローブ（DINOv2 ViT-B/14 + task-id embedding + MLP）の推論アダプタ。

**言語エンコーダを持たない。** reset() は指示文を受け取るが**使わない**。
これが診断の要点で、指示文を無視する方策がどこまで解けるかを測る。
task-id は rollout_id から与える（runner が set_task で渡す）。
"""
import os, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from probe.train_probe import Probe, CHUNK

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "probe", "ckpt")


class ProbePolicy:
    """action chunking。chunk 内は再計算せずに順に出す（OpenVLA-OFT と同じ運用）。"""
    name = "probe90m"

    def __init__(self, suite: str, device="cuda", replan=CHUNK):
        self.dev = device
        self.bb = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
        for p in self.bb.parameters(): p.requires_grad_(False)
        d = torch.load(os.path.join(CKPT, f"{suite}.pt"), map_location=device, weights_only=False)
        self.m = Probe().to(device).eval(); self.m.load_state_dict(d["state"])
        self.mu = torch.tensor(d["mu"], device=device).float()
        self.sd = torch.tensor(d["sd"], device=device).float()
        self.pmu = torch.tensor(d["pmu"], device=device).float()
        self.psd = torch.tensor(d["psd"], device=device).float()
        self.chunk = d["chunk"]; self.replan = min(replan, d["chunk"])
        self.task_id = 0; self._buf = None; self._i = 0
        self.suite = suite

    def set_task(self, task_id: int): self.task_id = int(task_id)

    def reset(self, language: str, *, seed: int) -> None:
        self._buf = None; self._i = 0        # language は **使わない**（言語エンコーダ非搭載）

    @torch.no_grad()
    def _feat(self, a, w):
        x = torch.from_numpy(np.stack([a, w])).to(self.dev).permute(0, 3, 1, 2).float() / 255.
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        x = (x - MEAN.to(self.dev)) / STD.to(self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.bb(pixel_values=x).last_hidden_state
        f = torch.cat([h[:, 0], h[:, 1:].mean(1)], -1).float()      # (2, 1536)
        return torch.cat([f[0], f[1]]).unsqueeze(0)                  # (1, 3072)

    def _prop(self, obs):
        """デモ hdf5 の ee_pos(3) + ee_ori(3, axis-angle) + gripper_states(2) と同じ並び。"""
        from robosuite.utils.transform_utils import quat2axisangle
        import numpy as _np
        return _np.concatenate([obs["robot0_eef_pos"],
                                quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"]]).astype(_np.float32)

    @torch.no_grad()
    def act(self, agentview, wrist, obs):
        if self._buf is None or self._i >= self.replan:
            f = self._feat(agentview, wrist)
            t = torch.tensor([self.task_id], device=self.dev)
            p = torch.from_numpy(self._prop(obs)).to(self.dev).unsqueeze(0)
            p = (p - self.pmu) / self.psd
            y = self.m(f, t, p)[0] * self.sd + self.mu               # (chunk, 7)
            self._buf = y.float().cpu().numpy(); self._i = 0
        a = self._buf[self._i]; self._i += 1
        a = np.asarray(a, np.float64).copy()
        a[:6] = np.clip(a[:6], -1, 1)
        a[6] = 1.0 if a[6] > 0 else -1.0                             # gripper は二値
        return a
