"""Implementation 1 -- touch no env at all; write a policy with two methods.

    class MyPolicy:
        def reset(self, language: str, *, seed: int) -> None: ...
        def act(self, agentview, wrist, obs) -> np.ndarray: ...

That is enough to drive it from the CLI:

    libero-ctrl run --policy 01_policy_adapter:MyPolicy --axis camera --level L2

A policy whose dependencies cannot share a process with the env (a different Python or MuJoCo
version) uses the unix-socket servers in examples/servers/ instead. The seven policies in the
paper span Python 3.10-3.13 and MuJoCo 2.3.7-3.3.2, and were all evaluated that way.
"""
import numpy as np


class MyPolicy:
    def __init__(self):
        # load the trained model here
        self.model = None

    def reset(self, language: str, *, seed: int) -> None:
        """Called at the start of each rollout. `language` is the perturbed instruction.

        The seed is derived deterministically from the rollout id. Seeding the policy's own
        randomness here makes the rollout reproducible; leaving it unseeded makes the policy
        stochastic, which is measured and reported rather than treated as an error.
        """
        self.language = language
        self.rng = np.random.default_rng(seed)

    def act(self, agentview, wrist, obs) -> np.ndarray:
        """agentview / wrist: (H,W,3) uint8 in robosuite's raw orientation (OpenGL, bottom-up).

        Released checkpoints disagree about the image convention -- OpenVLA-OFT expects a
        180-degree rotation -- so convert here for your own model; the benchmark does not guess.
        `obs` is the robosuite observation dict as-is, so robot0_eef_pos, robot0_gripper_qpos,
        robot0_joint_pos and the rest are all available.
        """
        return self.rng.uniform(-0.2, 0.2, 7)
