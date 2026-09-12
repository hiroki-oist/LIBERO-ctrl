"""軸6 Actuation Error。「指令した先に行かない」系統誤差を runner 側の action 変換で作る。

パラメータ空間（5次元）。すべて action 変換だけで、model には触らない。
  0 gain_log2     指令変位が 2^d 倍で実行される（スケール誤差）         系統
  1 bias_frac     固定方向への一定オフセット（キャリブレーション誤差）  系統
  2 misalign_deg  指令座標系が固定軸まわりに θ 回転している            系統
  3 lag_tau       一次遅れ時定数（制御ステップ）                        系統
  4 noise_frac    ランダムノイズ σ                                      ランダム

bias と noise は **デモの action 大きさ**にアンカーする。回転の action は並進の約 1/7
（‖a_rot‖ RMS 0.10 対 ‖a_trans‖ RMS 0.72）なので、並進群・回転群それぞれの RMS で
スケールしないと片方だけ過大になる。gripper 次元（常に ±1 の二値）は摂動しない。
"""
import numpy as np

KEYS = ["gain_log2", "bias_frac", "misalign_deg", "lag_tau", "noise_frac"]
# 1 unit は **EEF の追従誤差**にアンカーする。デモ action の開ループ再生で
# 各パラメータの「r あたりの EEF 最大ずれ」を実測し、単独 r=8 で 20mm になるよう逆算した。
#
# 参照する実機は2クラス:
#   研究用（Franka Panda）: 繰り返し ~0.1mm / 絶対精度 ~1mm / 追従誤差 2-10mm / 最悪 ~20mm
#   低価格（SO-101, Feetech STS3215）: 繰り返し 0.17deg / バックラッシュ 1-2mm@100mm レバー
#     （負荷時 1.30deg）-> EEF 換算で概ね 10-30mm
# -> L1 = 10mm（良好な低価格 / 並の研究用）, L2 = 20mm（研究用の最悪 = 典型的 SO-101）,
#    L3 = 40mm（負荷とバックラッシュが乗った SO-101、校正不良）
#
# 単独 r=8 での実量（すべて低価格アームとしてあり得る水準）:
#   gain x1.082 / bias 5.95% / misalign 7.17deg / lag 2.60 step (130ms) / noise sigma 18.8%
# 意図的に強くしすぎない。実機であり得ない値（gain x2 等）にすると軸の説得力が失われる。
UNIT = np.array([0.01432, 0.00744, 0.89606, 0.32468, 0.02350])
# lag と noise は絶対値しか効かない（±で同じ）ので半空間からサンプルする
HALF_SPACE = [3, 4]
MULT = {"L1": 2.0, "L2": 4.0, "L3": 8.0}
# デモ実測（suite ごと。translation / rotation の action 大きさ RMS）
DEMO_RMS = {"libero_spatial": (0.785, 0.102), "libero_object": (0.697, 0.072),
            "libero_goal": (0.739, 0.143), "libero_10": (0.581, 0.113)}


def params(level, direction):
    """direction は単位球面上の 5 次元ベクトル。lag と noise の成分は絶対値を取る。"""
    v = np.asarray(direction, float).copy()
    for i in HALF_SPACE: v[i] = abs(v[i])
    return dict(zip(KEYS, MULT[level] * v * UNIT))


def _rot(axis, ang):
    a = axis / np.linalg.norm(axis); c, s = np.cos(ang), np.sin(ang)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(a, a)


class ActuationError:
    """1 rollout ぶんの action 変換。固定方向・固定軸は seed から一度だけ決める。"""

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
        self.state = np.zeros(6)          # ★遅れの初期値はゼロ（指令値で初期化すると遅れが消える）

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
