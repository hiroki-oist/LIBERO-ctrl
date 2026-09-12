"""Client for a policy server running in a separate process.

This is how a policy whose dependencies cannot coexist with the env stack
(Python 3.10 / MuJoCo 2.3.7 / robosuite 1.4.0) is evaluated without changing anything on the
env side. Five of the seven policies in the paper need it; one of them requires
Python 3.13 and MuJoCo 3.3.2.
"""
import socket, numpy as np
from .wire import send, recv


class RemotePolicy:
    """Send an observation, receive an action. Orientation and normalisation are the
    server's responsibility, not the benchmark's."""

    def __init__(self, sock_path: str, name: str = "remote", task_string: str | None = None):
        self.name = name
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.connect(sock_path)
        send(self.s, dict(cmd="ping")); h, _ = recv(self.s)
        if not h.get("ok"): raise RuntimeError(h)
        self.n_params = h.get("params")
        self.task_string = task_string        # when set, this string is sent instead of the row's

    def set_task_string(self, s: str): self.task_string = s

    def reset(self, language: str, *, seed: int) -> None:
        # Use task_string when one was set; otherwise the row's own instruction.
        self._task = self.task_string if self.task_string is not None else language
        send(self.s, dict(cmd="reset", task=self._task, seed=int(seed)))
        h, _ = recv(self.s)
        if not h.get("ok"): raise RuntimeError(h.get("err"))

    def act(self, agentview, wrist, obs) -> np.ndarray:
        send(self.s, dict(cmd="act", task=self._task), dict(
            agentview=np.ascontiguousarray(agentview, np.uint8),
            wrist=np.ascontiguousarray(wrist, np.uint8),
            eef_pos=np.asarray(obs["robot0_eef_pos"], np.float32),
            eef_quat=np.asarray(obs["robot0_eef_quat"], np.float32),
            eef_mat=np.asarray(obs.get("robot0_eef_mat", np.eye(3)), np.float32),
            grip_qpos=np.asarray(obs["robot0_gripper_qpos"], np.float32),
            grip_qvel=np.asarray(obs["robot0_gripper_qvel"], np.float32),
            joint_pos=np.asarray(obs["robot0_joint_pos"], np.float32),
            joint_vel=np.asarray(obs["robot0_joint_vel"], np.float32)))
        h, arr = recv(self.s)
        if not h.get("ok"): raise RuntimeError(h.get("err"))
        return np.asarray(arr["action"], np.float64)

    def close(self):
        try: self.s.close()
        except Exception: pass
