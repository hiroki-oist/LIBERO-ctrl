"""Level 2 — env を一切触らず、2 メソッドの方策だけ書く。

    class MyPolicy:
        def reset(self, language: str, *, seed: int) -> None: ...
        def act(self, agentview, wrist, obs) -> np.ndarray: ...

これだけで CLI から回せる:

    libero-ctrl run --policy 03_policy_adapter:MyPolicy --axis camera --level L2

依存が衝突して同じプロセスに同居できない方策（Python や MuJoCo のバージョン違い）は
examples/servers/ の unix socket サーバ方式を使う。論文の 7 方策は Python 3.10/3.13、
MuJoCo 2.3.7/3.3.2 が混在しており、全てこの方式で評価した。
"""
import numpy as np


class MyPolicy:
    def __init__(self):
        # ここで学習済みモデルを読み込む
        self.model = None

    def reset(self, language: str, *, seed: int) -> None:
        """1 rollout の開始時に呼ばれる。language は摂動後の指示文。

        ★seed は rollout_id から決定論的に導かれる。方策内部の乱数をここで固定すれば
          rollout は完全に再現可能になる。固定しない場合は確率的な方策として扱われる
          （論文の測定では 7 方策中 3 本がこれに該当した）。
        """
        self.language = language
        self.rng = np.random.default_rng(seed)

    def act(self, agentview, wrist, obs) -> np.ndarray:
        """agentview / wrist: (H,W,3) uint8、robosuite の生の向き（OpenGL 下から上）。

        ★画像の向きの規約はモデルごとに違う。OpenVLA-OFT は 180 度回転を要求する。
          ここで各自のモデルに合わせて変換すること（ベンチマーク側では決め打ちしない）。
        obs は robosuite の observation dict そのままなので、
        robot0_eef_pos / robot0_gripper_qpos / robot0_joint_pos などが使える。
        """
        return self.rng.uniform(-0.2, 0.2, 7)
