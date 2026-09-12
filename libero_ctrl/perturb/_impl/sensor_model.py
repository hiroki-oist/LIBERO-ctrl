"""軸5 Sensor Corruption。observation の劣化を runner 側で適用する（env に触らない）。

パラメータ空間（4次元）:
  0 noise_sigma   加算 Gaussian ノイズ σ（画素値 [0,1]）        毎ステップ変化
  1 blur_sigma    Gaussian ぼかし σ（px, 基準解像度 128）        静的
  2 jpeg_log2     JPEG quality = 100 * 2^(-d)                    静的
  3 motion_len    motion blur カーネル長（px, 基準 128）          静的（向きは rollout 固定）

遮蔽（黒パッチ）は 2026-09-03 に軸から外した。情報を削る摂動なので対象物を覆えば
タスクが消滅し、設計原則 Solvable に反する（camera 軸で近づく方向をサンプルしないのと同じ）。
残る4つはすべて「センサの劣化過程」（ショットノイズ・デフォーカス・圧縮・動きぶれ）で
質が揃っている。

px 単位（1, 3）は解像度に比例させる。noise と jpeg は解像度非依存。
静的な要素（ぼかし・JPEG・motion blur の向き・遮蔽の位置）は rollout ごとに固定し、
ノイズだけ毎ステップ引く。レンズの汚れや光学ぼけは時間変化しないという物理に合わせる。
"""
import io, numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, convolve

REF_RES = 128
KEYS = ["noise_sigma", "blur_sigma", "jpeg_log2", "motion_len"]
# 1 unit（pilot 校正済み: r=8 が「強いが解ける」、r=16 で破綻するよう調整）
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
    """1 rollout ぶんの劣化器。静的要素を seed から一度だけ決める。"""

    def __init__(self, p, seed, shape):
        """p: params() の dict。shape: (H, W)。"""
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
        """適用順: motion blur -> gauss blur -> JPEG -> ノイズ（光学 -> 符号化 -> 読み出し）。"""
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
