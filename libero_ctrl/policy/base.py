"""The contract a policy adapter has to satisfy.

The runner passes the instruction **exactly as the manifest row gives it**. It must never be
derived from the env or from a BDDL filename: LIBERO-Plus did that, and perturbation tags
leaked into the instruction on every non-language axis as a result.
"""
from typing import Protocol
import numpy as np


class Policy(Protocol):
    name: str

    def reset(self, language: str, *, seed: int) -> None:
        """Start of one rollout. The instruction arrives here."""

    def act(self, agentview: np.ndarray, wrist: np.ndarray, obs: dict) -> np.ndarray:
        """Return a (7,) action."""


class ScriptedNoop:
    """A do-nothing policy for wiring checks; it only opens the gripper."""
    name = "noop"

    def reset(self, language: str, *, seed: int) -> None: self.lang = language

    def act(self, agentview, wrist, obs):
        a = np.zeros(7); a[-1] = -1.0; return a
