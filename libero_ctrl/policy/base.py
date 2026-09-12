"""方策アダプタの規約。

runner は **manifest の行にある指示文をそのまま渡す**。
env や bddl のファイル名から導出してはならない（LIBERO-Plus はそれで
全非言語軸の指示文に摂動タグが混入していた）。
"""
from typing import Protocol
import numpy as np


class Policy(Protocol):
    name: str

    def reset(self, language: str, *, seed: int) -> None:
        """1 rollout の開始。指示文はここで受け取る。"""

    def act(self, agentview: np.ndarray, wrist: np.ndarray, obs: dict) -> np.ndarray:
        """(7,) の action を返す。"""


class ScriptedNoop:
    """疎通確認用。何もしない（gripper だけ開く）。"""
    name = "noop"

    def reset(self, language: str, *, seed: int) -> None: self.lang = language

    def act(self, agentview, wrist, obs):
        a = np.zeros(7); a[-1] = -1.0; return a
