from __future__ import annotations

from datetime import date

from src.fusion.fusion_engine import LambdaController, RegimeClassification
from src.rl.environment import FusionState, PortfolioEnv
from src.rl.reward import CVaR, PortfolioSnapshot, compute_reward, rolling_sharpe


def test_compute_reward_matches_formula_components() -> None:
    lambdas = LambdaController().get(3)
    portfolio = PortfolioSnapshot(
        weights=[0.25, 0.20, 0.10],
        cash_allocation=0.45,
        returns=[0.01, -0.02, 0.03, -0.01],
        transaction_costs_this_step=0.004,
        current_drawdown=0.12,
    )
    reward = compute_reward(portfolio, lambdas, t=3)
    pnl = portfolio.returns[3]
    sharpe_pen = -max(0.0, 1.0 - rolling_sharpe(3, portfolio.returns, 30))
    cvar_pen = -lambdas.cvar * CVaR(portfolio.returns, alpha=0.05)
    tc_pen = -portfolio.transaction_costs_this_step
    dd_pen = -lambdas.dd * (portfolio.current_drawdown ** 2)
    conc_pen = -lambdas.hhi * sum(weight ** 2 for weight in portfolio.weights)
    expected = pnl + sharpe_pen + cvar_pen + tc_pen + dd_pen + conc_pen
    assert abs(reward - expected) < 1e-9


def test_portfolio_env_enforces_cash_and_position_caps() -> None:
    regime = RegimeClassification(regime_id=0, regime_label="bull", regime_proba=[0.7, 0.1, 0.15, 0.05], as_of_date=date(2026, 3, 30))
    env = PortfolioEnv(
        asset_symbols=["AAPL", "MSFT", "NVDA"],
        sector_map={"AAPL": "tech", "MSFT": "tech", "NVDA": "tech"},
        fusion_states=[
            FusionState(
                combined_signal=0.7,
                short_bias=0.2,
                long_bias=0.8,
                confidence_score=0.9,
                risk_level="low",
                regime=regime,
            )
        ],
        realized_returns=[{"AAPL": 0.01, "MSFT": 0.02, "NVDA": 0.03}],
    )
    weights, cash, hedge = env._decode_action([0.9, 0.8, 0.7, 0.0, 1.0])
    assert all(weight <= 0.25 for weight in weights)
    assert cash >= 0.05
    assert sum(weights) + cash <= 1.000001
    assert hedge is True


def test_portfolio_env_step_uses_regime_lambdas() -> None:
    regime = RegimeClassification(regime_id=3, regime_label="crisis", regime_proba=[0.05, 0.10, 0.10, 0.75], as_of_date=date(2026, 3, 30))
    env = PortfolioEnv(
        asset_symbols=["AAPL", "XOM"],
        sector_map={"AAPL": "tech", "XOM": "energy"},
        fusion_states=[
            FusionState(
                combined_signal=0.4,
                short_bias=0.6,
                long_bias=0.4,
                confidence_score=0.5,
                risk_level="high",
                regime=regime,
            )
        ],
        realized_returns=[{"AAPL": -0.02, "XOM": 0.01}],
    )
    observation, _ = env.reset()
    assert len(observation) > 0
    next_obs, reward, terminated, truncated, info = env.step([0.25, 0.25, 0.5, 1.0])
    assert isinstance(reward, float)
    assert terminated is True
    assert truncated is False
    assert info["reward_breakdown"]["dd_pen"] <= 0.0
    assert info["hedge_active"] is True
    assert isinstance(next_obs, list)
