from __future__ import annotations

from dataclasses import dataclass

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional runtime dependency
    gym = None
    spaces = None


if gym is not None:

    @dataclass
    class PortfolioState:
        fused_signal: float
        uncertainty: float
        current_weight: float
        estimated_cost: float


    class PortfolioEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=float)
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=float)
            self._observation = [0.0, 0.0, 0.0, 0.0]

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            self._observation = [0.0, 0.0, 0.0, 0.0]
            return self._observation, {}

        def step(self, action):
            reward = float(action[0]) - abs(self._observation[3])
            terminated = True
            truncated = False
            info = {}
            return self._observation, reward, terminated, truncated, info
