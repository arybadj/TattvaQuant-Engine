"""PPO agent wrapper for the portfolio environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.rl.environment import PortfolioPPOTrainer, PPOTrainingConfig


@dataclass
class PPOAgent:
    """Thin wrapper around the PPO trainer so the RL package exposes an agent entrypoint."""

    env_factory: Any
    config: PPOTrainingConfig = field(default_factory=PPOTrainingConfig)

    def train(self, hyperparams: dict[str, Any] | None = None) -> dict[str, Any]:
        trainer = PortfolioPPOTrainer(env_factory=self.env_factory, config=self.config)
        return trainer.train(hyperparams=hyperparams)

    def optimize(self, n_trials: int = 100) -> dict[str, Any]:
        trainer = PortfolioPPOTrainer(env_factory=self.env_factory, config=self.config)
        return trainer.optimize(n_trials=n_trials)
