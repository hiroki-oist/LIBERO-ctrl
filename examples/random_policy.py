"""The smallest policy that runs: random actions. This is the shape a user has to implement."""
import numpy as np

class RandomPolicy:
    def reset(self, language: str, *, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.language = language          # on the language axis, this string is what changes

    def act(self, agentview, wrist, obs) -> np.ndarray:
        # agentview / wrist: (H,W,3) uint8. obs is the raw robosuite observation dict.
        return self.rng.uniform(-0.2, 0.2, 7)
