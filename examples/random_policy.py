"""動作確認用のランダム方策。利用者が実装すべき最小の形。"""
import numpy as np

class RandomPolicy:
    def reset(self, language: str, *, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.language = language          # 言語軸はこの文字列だけが変わる

    def act(self, agentview, wrist, obs) -> np.ndarray:
        # agentview / wrist: (H,W,3) uint8。obs は robosuite の生 observation。
        return self.rng.uniform(-0.2, 0.2, 7)
