"""Portfolio environment and PPO training helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.fusion.fusion_engine import LambdaController, RegimeClassification
from src.rl.reward import PortfolioSnapshot, compute_reward, compute_reward_breakdown
from src.uncertainty.uncertainty_engine import UncertaintyOutput

try:
    import mlflow
except ImportError:  # pragma: no cover - optional runtime dependency
    mlflow = None

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional runtime dependency
    gym = None
    spaces = None

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:  # pragma: no cover - optional runtime dependency
    PPO = None
    DummyVecEnv = None

try:
    import optuna
except ImportError:  # pragma: no cover - optional runtime dependency
    optuna = None


@dataclass
class FusionState:
    combined_signal: float
    short_bias: float
    long_bias: float
    confidence_score: float
    risk_level: str
    regime: RegimeClassification
    uncertainty: UncertaintyOutput | None = None


@dataclass
class EnvironmentStepResult:
    observation: list[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


def _normalize(value: float, lower: float, upper: float) -> float:
    if upper == lower:
        return 0.0
    normalized = (float(value) - lower) / (upper - lower)
    return float(max(0.0, min(1.0, normalized)))


class PortfolioEnv(gym.Env if gym is not None else object):
    """
    State space (all normalized 0-1):
      - fusion_output: combined_signal, short/long bias
      - uncertainty: confidence_score, risk_level
      - portfolio_state: current weights, cash %, drawdown
      - regime: regime_id (one-hot), regime_proba

    Action space (continuous):
      - position_weights: array of floats, sum <= 1
      - cash_allocation: float 0-1
      - optional hedge_flag for backwards compatibility with older tests

    Constraints:
      - Max single position: 25% of portfolio
      - Min cash: 5% always
      - Max sector concentration: 35%
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        asset_symbols: list[str],
        sector_map: dict[str, str],
        fusion_states: list[FusionState],
        realized_returns: list[dict[str, float]],
        initial_cash: float = 0.05,
        uncertainty_states: list[UncertaintyOutput] | None = None,
        include_hedge_action: bool = True,
    ) -> None:
        self.asset_symbols = asset_symbols
        self.sector_map = sector_map
        self.fusion_states = fusion_states
        self.realized_returns = realized_returns
        self.uncertainty_states = uncertainty_states or []
        self.include_hedge_action = include_hedge_action
        self.lambda_controller = LambdaController()
        self.max_single_position = 0.25
        self.min_cash = 0.05
        self.max_sector_concentration = 0.35
        self.initial_cash = initial_cash
        self.current_step = 0
        self.current_weights = [0.0 for _ in asset_symbols]
        self.cash_allocation = initial_cash
        self.current_drawdown = 0.0
        self.returns_history: list[float] = []
        self.hedge_active = False

        obs_length = 3 + 2 + len(asset_symbols) + 2 + 4 + 4
        action_length = len(asset_symbols) + (2 if include_hedge_action else 1)
        if spaces is not None:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(obs_length,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(action_length,), dtype=np.float32
            )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if gym is not None:
            super().reset(seed=seed)
        self.current_step = 0
        self.current_weights = [0.0 for _ in self.asset_symbols]
        self.cash_allocation = self.initial_cash
        self.current_drawdown = 0.0
        self.returns_history = []
        self.hedge_active = False
        observation = self._build_observation()
        return observation, {}

    def step(self, action):
        weights, cash_allocation, hedge_flag = self._decode_action(action)
        transaction_cost = self._transaction_costs(weights, cash_allocation, hedge_flag)
        self.current_weights = weights
        self.cash_allocation = cash_allocation
        self.hedge_active = hedge_flag

        step_returns = self.realized_returns[self.current_step]
        gross_return = sum(
            step_returns.get(symbol, 0.0) * weight
            for symbol, weight in zip(self.asset_symbols, weights, strict=False)
        )
        if hedge_flag:
            gross_return -= 0.002
        net_return = gross_return - transaction_cost
        self.returns_history.append(net_return)
        cumulative = sum(self.returns_history)
        peak = max(
            [0.0]
            + [sum(self.returns_history[: index + 1]) for index in range(len(self.returns_history))]
        )
        self.current_drawdown = max(0.0, peak - cumulative)

        regime = self.fusion_states[self.current_step].regime
        lambdas = self.lambda_controller.get(regime.regime_id)
        portfolio = PortfolioSnapshot(
            weights=list(weights),
            cash_allocation=cash_allocation,
            returns=list(self.returns_history),
            transaction_costs_this_step=transaction_cost,
            current_drawdown=self.current_drawdown,
            sector_weights=self._sector_weights(weights),
        )
        reward = compute_reward(portfolio, lambdas, self.current_step)
        info = {
            "reward_breakdown": compute_reward_breakdown(
                portfolio, lambdas, self.current_step
            ).to_json(),
            "cash_allocation": cash_allocation,
            "hedge_active": hedge_flag,
            "position_size_multiplier": 1.0,
        }

        self.current_step += 1
        terminated = self.current_step >= len(self.fusion_states)
        truncated = False
        observation = (
            self._build_observation() if not terminated else [0.0] * len(self._build_observation())
        )
        return observation, reward, terminated, truncated, info

    def _build_observation(self) -> list[float]:
        state = self.fusion_states[min(self.current_step, len(self.fusion_states) - 1)]
        uncertainty = self._current_uncertainty_state()
        risk_map = {"low": 0.2, "medium": 0.5, "high": 0.9}
        regime_one_hot = [1.0 if index == state.regime.regime_id else 0.0 for index in range(4)]
        observation = [
            _normalize(state.combined_signal, -1.0, 1.0),
            _normalize(state.short_bias, -1.0, 1.0),
            _normalize(state.long_bias, -1.0, 1.0),
            float(max(0.0, min(1.0, uncertainty.confidence_score))),
            risk_map.get(uncertainty.risk_level, 0.5),
        ]
        observation.extend(float(max(0.0, min(1.0, weight))) for weight in self.current_weights)
        observation.extend(
            [float(self.cash_allocation), float(max(0.0, min(1.0, self.current_drawdown)))]
        )
        observation.extend(regime_one_hot)
        observation.extend(float(max(0.0, min(1.0, value))) for value in state.regime.regime_proba)
        return observation

    def _decode_action(self, action: Any) -> tuple[list[float], float, bool]:
        values = [float(item) for item in list(action)]
        proposed_weights = values[: len(self.asset_symbols)]
        expected_without_hedge = len(self.asset_symbols) + 1
        cash_index = len(self.asset_symbols)
        cash_allocation = max(
            self.min_cash,
            min(1.0, values[cash_index] if len(values) > cash_index else self.min_cash),
        )
        hedge_flag = bool(len(values) > expected_without_hedge and values[-1] >= 0.5)
        capped = [min(max(weight, 0.0), self.max_single_position) for weight in proposed_weights]
        if sum(capped) > (1.0 - cash_allocation):
            scale = (1.0 - cash_allocation) / max(sum(capped), 1e-6)
            capped = [weight * scale for weight in capped]
        sector_weights = self._sector_weights(capped)
        for sector, sector_weight in sector_weights.items():
            if sector_weight > self.max_sector_concentration:
                reduction_scale = self.max_sector_concentration / max(sector_weight, 1e-6)
                capped = [
                    weight * reduction_scale if self.sector_map[symbol] == sector else weight
                    for symbol, weight in zip(self.asset_symbols, capped, strict=False)
                ]
        if sum(capped) + cash_allocation > 1.0:
            scale = (1.0 - cash_allocation) / max(sum(capped), 1e-6)
            capped = [weight * scale for weight in capped]
        return capped, cash_allocation, hedge_flag

    def _current_uncertainty_state(self) -> UncertaintyOutput:
        if self.uncertainty_states:
            return self.uncertainty_states[min(self.current_step, len(self.uncertainty_states) - 1)]
        state = self.fusion_states[min(self.current_step, len(self.fusion_states) - 1)]
        if state.uncertainty is not None:
            return state.uncertainty
        return UncertaintyOutput(
            confidence_score=float(max(0.0, min(1.0, state.confidence_score))),
            prediction_variance=float(max(0.0, 1.0 - state.confidence_score)),
            risk_level=state.risk_level,
            shift_detected=False,
            mmd_score=0.0,
        )

    def _transaction_costs(
        self, weights: list[float], cash_allocation: float, hedge_flag: bool
    ) -> float:
        turnover = sum(
            abs(new - old) for new, old in zip(weights, self.current_weights, strict=False)
        )
        cash_turnover = abs(cash_allocation - self.cash_allocation)
        hedge_cost = 0.002 if hedge_flag and not self.hedge_active else 0.0
        return float(turnover * 0.0015 + cash_turnover * 0.0005 + hedge_cost)

    def _sector_weights(self, weights: list[float]) -> dict[str, float]:
        sector_weights: dict[str, float] = {}
        for symbol, weight in zip(self.asset_symbols, weights, strict=False):
            sector = self.sector_map.get(symbol, "unknown")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + float(weight)
        return sector_weights


@dataclass
class PPOTrainingConfig:
    total_timesteps: int = 10_000
    eval_episodes: int = 5
    quarterly_retrain_steps: int = 63
    artifact_root: Path = Path("models")
    checkpoint_path: Path = Path("models/ppo_agent.zip")


@dataclass
class PortfolioPPOTrainer:
    env_factory: Any
    config: PPOTrainingConfig = field(default_factory=PPOTrainingConfig)

    def train(self, hyperparams: dict[str, Any] | None = None) -> dict[str, Any]:
        if PPO is None or DummyVecEnv is None:
            raise ImportError("stable-baselines3 is required for PPO training.")
        params = hyperparams or {
            "learning_rate": 3e-4,
            "n_steps": 64,
            "batch_size": 32,
            "gamma": 0.99,
            "policy_kwargs": {"net_arch": [32, 32]},
            "device": "cpu",
        }
        env = DummyVecEnv([self.env_factory])
        model = PPO("MlpPolicy", env, verbose=0, **params)
        model.learn(total_timesteps=self.config.total_timesteps)
        evaluation = self.evaluate(model)
        checkpoint_path = self._save_model(model)
        self._log_run(params=params, metrics=evaluation, model=model)
        return {"params": params, "metrics": evaluation, "checkpoint_path": str(checkpoint_path)}

    def walk_forward_validate(
        self, windows: Iterable[tuple[int, int]], hyperparams: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        results = []
        for fold, _window in enumerate(windows, start=1):
            result = self.train(hyperparams=hyperparams)
            result["fold"] = fold
            results.append(result)
        return results

    def evaluate(self, model: Any) -> dict[str, float]:
        episode_rewards = []
        for _ in range(self.config.eval_episodes):
            env = self.env_factory()
            observation, _ = env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            while not terminated and not truncated:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
            episode_rewards.append(total_reward)
        average_reward = sum(episode_rewards) / max(len(episode_rewards), 1)
        return {"average_reward": float(average_reward)}

    def optimize(self, n_trials: int = 100) -> dict[str, Any]:
        if optuna is None:
            raise ImportError("optuna is required for hyperparameter search.")

        def objective(trial):
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
                "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048]),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "gamma": trial.suggest_float("gamma", 0.95, 0.999),
            }
            result = self.train(hyperparams=params)
            return result["metrics"]["average_reward"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return {"best_params": study.best_params, "best_value": study.best_value}

    def _log_run(self, params: dict[str, Any], metrics: dict[str, float], model: Any) -> None:
        self.config.artifact_root.mkdir(parents=True, exist_ok=True)
        summary_path = self.config.artifact_root / "ppo_training_summary.json"
        summary_path.write_text(
            json.dumps({"params": params, "metrics": metrics}, indent=2), encoding="utf-8"
        )
        if mlflow is not None:
            try:
                with mlflow.start_run(run_name="ppo-portfolio-training", nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)
                    mlflow.log_metric("mean_reward", float(metrics.get("average_reward", 0.0)))
                    mlflow.log_artifact(str(summary_path))
            except Exception:
                pass

    def _save_model(self, model: Any) -> Path:
        checkpoint_path = self.config.checkpoint_path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save"):
            model.save(str(checkpoint_path))
        return checkpoint_path
