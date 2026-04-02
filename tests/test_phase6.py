from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.fusion.fusion_engine import RegimeClassification
from src.rl.agent import PPOAgent
from src.rl.environment import FusionState, PPOTrainingConfig, PortfolioEnv
from src.uncertainty.uncertainty_engine import (
    DistributionalShiftDetector,
    DistributionalShiftWarning,
    MonteCarloDropout,
    UncertaintyOutput,
)

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def _artifact_dir(name: str) -> Path:
    target = Path("data/test_artifacts/phase6") / name
    target.mkdir(parents=True, exist_ok=True)
    return target


def _regime() -> RegimeClassification:
    return RegimeClassification(
        regime_id=0,
        regime_label="bull",
        regime_proba=[0.70, 0.10, 0.15, 0.05],
        as_of_date=date(2026, 3, 30),
    )


def _uncertainty(confidence: float = 0.8, risk_level: str = "low") -> UncertaintyOutput:
    return UncertaintyOutput(
        confidence_score=confidence,
        prediction_variance=0.05,
        risk_level=risk_level,
        shift_detected=False,
        mmd_score=0.01,
    )


def _fusion_states() -> list[FusionState]:
    regime = _regime()
    return [
        FusionState(
            combined_signal=0.65,
            short_bias=0.35,
            long_bias=0.75,
            confidence_score=0.8,
            risk_level="low",
            regime=regime,
            uncertainty=_uncertainty(0.8, "low"),
        ),
        FusionState(
            combined_signal=0.55,
            short_bias=0.45,
            long_bias=0.70,
            confidence_score=0.72,
            risk_level="medium",
            regime=regime,
            uncertainty=_uncertainty(0.72, "medium"),
        ),
    ]


def _returns() -> list[dict[str, float]]:
    return [
        {"AAPL": 0.012, "MSFT": 0.009, "NVDA": 0.018},
        {"AAPL": -0.004, "MSFT": 0.006, "NVDA": 0.011},
    ]


def _env(include_hedge_action: bool = True) -> PortfolioEnv:
    return PortfolioEnv(
        asset_symbols=["AAPL", "MSFT", "NVDA"],
        sector_map={"AAPL": "tech", "MSFT": "tech", "NVDA": "tech"},
        fusion_states=_fusion_states(),
        uncertainty_states=[state.uncertainty for state in _fusion_states() if state.uncertainty is not None],
        realized_returns=_returns(),
        include_hedge_action=include_hedge_action,
    )


@pytest.mark.skipif(torch is None, reason="torch is required for Monte Carlo dropout")
def test_monte_carlo_dropout_returns_confidence_score_between_zero_and_one() -> None:
    class DropoutModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dropout = nn.Dropout(p=0.5)

        def forward(self, inputs):
            base = torch.tensor([0.25], dtype=torch.float32)
            return self.dropout(base)

    result = MonteCarloDropout(n_samples=100).run(DropoutModel(), inputs=torch.tensor([1.0]))
    assert 0.0 <= result["confidence_score"] <= 1.0


def test_distributional_shift_detector_flags_large_mmd_correctly() -> None:
    target = _artifact_dir("shift")
    detector = DistributionalShiftDetector(
        reference_path=target / "reference.pkl",
        alert_path=target / "alert.json",
        threshold=0.1,
    )
    training = pd.DataFrame([{"x1": 0.0, "x2": 0.0}, {"x1": 0.1, "x2": 0.1}, {"x1": 0.05, "x2": 0.02}])
    live = pd.DataFrame([{"x1": 3.0, "x2": 3.2}, {"x1": 2.8, "x2": 3.1}])
    detector.fit(training_features=training, as_of_date=date(2026, 3, 1))
    with pytest.warns(DistributionalShiftWarning):
        result = detector.evaluate(live_features=live, as_of_date=date(2026, 3, 30))
    assert result.shift_detected is True
    assert result.mmd_score > 0.1


def test_portfolio_env_resets_and_steps_without_error() -> None:
    env = _env()
    observation, info = env.reset()
    assert isinstance(info, dict)
    assert len(observation) > 0
    next_obs, reward, terminated, truncated, step_info = env.step([0.2, 0.2, 0.2, 0.4, 0.0])
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(step_info, dict)
    assert len(next_obs) > 0


def test_portfolio_env_enforces_max_position_size_of_25_percent() -> None:
    env = _env()
    weights, cash, hedge_flag = env._decode_action([0.9, 0.8, 0.7, 0.05, 0.0])
    assert all(weight <= 0.25 for weight in weights)
    assert cash >= 0.05
    assert hedge_flag is False


@pytest.mark.skipif(torch is None, reason="torch is required for PPO training")
def test_ppo_agent_trains_for_10000_steps_and_saves_checkpoint() -> None:
    target = _artifact_dir("ppo")
    config = PPOTrainingConfig(
        total_timesteps=10_000,
        eval_episodes=1,
        artifact_root=target / "models",
        checkpoint_path=target / "models" / "ppo_agent.zip",
    )
    agent = PPOAgent(env_factory=lambda: _env(include_hedge_action=False), config=config)
    result = agent.train()
    assert (target / "models" / "ppo_agent.zip").exists()
    assert result["checkpoint_path"].endswith("ppo_agent.zip")


def test_uncertainty_output_risk_level_is_valid() -> None:
    output = _uncertainty(confidence=0.55, risk_level="medium")
    assert output.risk_level in {"low", "medium", "high"}
