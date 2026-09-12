"""Axis 5, sensor corruption: degradation applied to the observation, never to the env.

Four parameters:
  0 noise_sigma   additive Gaussian noise sigma, in [0,1] pixel units   redrawn each step
  1 blur_sigma    Gaussian blur sigma, px at a reference resolution 128 static
  2 jpeg_log2     JPEG quality = 100 * 2^(-d)                           static
  3 motion_len    motion-blur kernel length, px at reference 128        static (as is its angle)

Occlusion (a black patch) was dropped from this axis during design. It removes information
rather than degrading it, so covering the target object destroys the task outright, which
violates the solvability requirement -- the same reason the camera axis does not sample
directions that move the camera into the scene. The four that remain are homogeneous: every one
of them is a sensor degradation process (shot noise, defocus, compression, motion blur).

The two parameters measured in pixels scale with resolution; noise and JPEG quality do not.
The static components are fixed once per rollout and only the noise is redrawn each step, which
matches the physics: a smudged lens or an optical defocus does not change between frames.
"""
import io, numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, convolve

REF_RES = 128
KEYS = ["noise_sigma", "blur_sigma", "jpeg_log2", "motion_len"]
# One unit, calibrated on a pilot so that r = 8 is "severe but solvable" and r = 16 breaks down.
UNIT = np.array([0.010, 0.25, 0.415, 0.75])
MULT = {"L1": 2.0, "L2": 4.0, "L3": 8.0}
JPEG_Q0 = 100.0


def params(level, direction):
    v = np.asarray(direction, float)
    return dict(zip(KEYS, MULT[level] * v * UNIT))


def _motion_kernel(length, angle_rad):
    n = max(1, int(round(length)))
    if n <= 1: return None
    k = np.zeros((n, n)); c = (n - 1) / 2.0
    for i in range(n):
        t = i - c
        x = int(round(c + t * np.cos(angle_rad))); y = int(round(c + t * np.sin(angle_rad)))
        if 0 <= x < n and 0 <= y < n: k[y, x] = 1.0
    return None if k.sum() == 0 else k / k.sum()


class SensorCorruption:
    """The corruption for one rollout. The static components are drawn once from the seed."""

    def __init__(self, p, seed, shape):
        """p: the dict returned by params(). shape: (H, W)."""
        self.p = p; H, W = shape[:2]; self.H, self.W = H, W
        self.sc = min(H, W) / REF_RES
        rs = np.random.default_rng(seed)
        self.step_rng = np.random.default_rng(seed + 1)
        self.mk = _motion_kernel(abs(p["motion_len"]) * self.sc, rs.uniform(0, np.pi)) \
            if abs(p["motion_len"]) * self.sc >= 1.5 else None
        self.blur = abs(p["blur_sigma"]) * self.sc
        d = abs(p["jpeg_log2"])
        self.q = int(np.clip(round(JPEG_Q0 * 2.0 ** (-d)), 1, 100)) if d > 1e-3 else None
    def __call__(self, img_u8):
        """Applied in physical order: motion blur -> Gaussian blur -> JPEG -> noise, i.e.
        optics, then encoding, then readout."""
        x = img_u8.astype(np.float64) / 255.0
        if self.mk is not None:
            for c in range(3): x[..., c] = convolve(x[..., c], self.mk, mode="nearest")
        if self.blur > 1e-3:
            for c in range(3): x[..., c] = gaussian_filter(x[..., c], self.blur, mode="nearest")
        if self.q is not None:
            buf = io.BytesIO()
            Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8)).save(
                buf, format="JPEG", quality=self.q)
            buf.seek(0)
            x = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float64) / 255.0
        s = abs(self.p["noise_sigma"])
        if s > 1e-5:
            x = x + self.step_rng.normal(0.0, s, x.shape)
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)
