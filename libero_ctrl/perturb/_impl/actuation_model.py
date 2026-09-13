"""Axis 6, actuation error: systematic error between what the policy commands and what the
arm does, implemented purely as a transform on the action.

Five parameters; none of them touches the model.
  0 gain_log2     commanded displacement executed at 2^d times scale   systematic
  1 bias_frac     constant offset along a fixed direction              systematic
  2 misalign_deg  command frame rotated by theta about a fixed axis    systematic
  3 lag_tau       first-order lag time constant, in control steps      systematic
  4 noise_frac    random noise sigma                                   random

Bias and noise are anchored to the magnitude of the demonstration actions. Rotational actions
are about a seventh the size of translational ones (RMS 0.10 against 0.72), so each group is
scaled by its own RMS; using a single scale would make one of them dominate. The gripper
dimension is binary (always +/-1) and is never perturbed.
"""
import numpy as np

KEYS = ["gain_log2", "bias_frac", "misalign_deg", "lag_tau", "noise_frac"]
# One unit is anchored to end-effector tracking error. Each parameter's worst-case
# end-effector displacement per unit radius was measured by open-loop replay of the
# demonstration actions, then inverted so that r = 8 along a single parameter gives 20 mm.
#
# Two classes of real hardware were used as the reference:
#   research-grade (Franka Panda): repeatability ~0.1 mm, absolute accuracy ~1 mm,
#     tracking error 2-10 mm, worst case ~20 mm
#   low-cost (SO-101, Feetech STS3215): repeatability 0.17 deg, backlash 1-2 mm on a 100 mm
#     lever (1.30 deg under load) -> roughly 10-30 mm at the end effector
# so L1 = 10 mm (a good low-cost arm, or an ordinary research one), L2 = 20 mm (worst case for
# research-grade, typical for SO-101), L3 = 40 mm (SO-101 under load, poorly calibrated).
#
# What r = 8 along a single parameter amounts to, all of it plausible on a low-cost arm:
# gain x1.082, bias 5.95%, misalignment 7.17 deg, lag 2.60 steps (130 ms), noise sigma 18.8%.
UNIT = np.array([0.01432, 0.00744, 0.89606, 0.32468, 0.02350])
# lag and noise depend only on magnitude, so they are sampled from a half-space
HALF_SPACE = [3, 4]
MULT = {"L1": 2.0, "L2": 4.0, "L3": 8.0}
# measured on the demonstrations, per suite: RMS action magnitude (translation, rotation)
DEMO_RMS = {"libero_spatial": (0.785, 0.102), "libero_object": (0.697, 0.072),
            "libero_goal": (0.739, 0.143), "libero_10": (0.581, 0.113)}


def params(level, direction):
    """`direction` is a five-dimensional unit vector; the lag and noise components are taken
    in absolute value."""
    v = np.asarray(direction, float).copy()
    for i in HALF_SPACE: v[i] = abs(v[i])
    return dict(zip(KEYS, MULT[level] * v * UNIT))


def _rot(axis, ang):
    a = axis / np.linalg.norm(axis); c, s = np.cos(ang), np.sin(ang)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(a, a)


class ActuationError:
    """The action transform for one rollout. The bias direction and misalignment axis are
    drawn once from the seed and then held fixed."""

    def __init__(self, p, seed, suite="libero_spatial"):
        self.p = p
        tr, ro = DEMO_RMS.get(suite, DEMO_RMS["libero_spatial"])
        self.scale = np.array([tr, tr, tr, ro, ro, ro])
        rs = np.random.default_rng(seed)
        self.step_rng = np.random.default_rng(seed + 1)
        d = rs.normal(size=6); self.bias_dir = d / np.linalg.norm(d)
        ax = rs.normal(size=3); self.axis = ax / np.linalg.norm(ax)
        self.gain = 2.0 ** p.get("gain_log2", 0.0)
        self.bias = p.get("bias_frac", 0.0) * self.bias_dir * self.scale
        self.R = _rot(self.axis, np.radians(p.get("misalign_deg", 0.0))) \
            if abs(p.get("misalign_deg", 0.0)) > 1e-9 else None
        self.tau = abs(p.get("lag_tau", 0.0))
        self.alpha = 1.0 / (1.0 + self.tau) if self.tau > 1e-9 else 1.0
        self.sigma = abs(p.get("noise_frac", 0.0)) * self.scale
        self.state = np.zeros(6)          # lag starts at zero; seeding it with the command
                                          # would cancel the lag on the first step

    def __call__(self, a):
        a = np.asarray(a, float).copy()
        u = a[:6].copy()
        if self.gain != 1.0: u = u * self.gain
        if self.R is not None:
            u[:3] = self.R @ u[:3]; u[3:6] = self.R @ u[3:6]
        u = u + self.bias
        if self.alpha < 1.0:
            self.state = self.alpha * u + (1.0 - self.alpha) * self.state
            u = self.state.copy()
        if np.any(self.sigma > 1e-12):
            u = u + self.step_rng.normal(0.0, 1.0, 6) * self.sigma
        a[:6] = u
        return np.clip(a, -1.0, 1.0)
