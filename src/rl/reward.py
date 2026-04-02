"""Risk-adjusted PPO reward helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


TARGET_SHARPE = 1.0


@dataclass
class PortfolioSnapshot:
    weights: list[float]
    cash_allocation: float
    returns: list[float] = field(default_factory=list)
    transaction_costs_this_step: float = 0.0
    current_drawdown: float = 0.0
    sector_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class RewardBreakdown:
    pnl: float
    sharpe_pen: float
    cvar_pen: float
    tc_pen: float
    dd_pen: float
    conc_pen: float
    total_reward: float

    def to_json(self) -> dict[str, float]:
        return asdict(self)


def portfolio_return(t: int, returns: Sequence[float]) -> float:
    if not returns:
        return 0.0
    index = min(max(t, 0), len(returns) - 1)
    return float(returns[index])


def rolling_sharpe(t: int, returns: Sequence[float], window: int = 30) -> float:
    if not returns:
        return 0.0
    end = min(max(t + 1, 1), len(returns))
    start = max(0, end - window)
    sample = list(float(value) for value in returns[start:end])
    if not sample:
        return 0.0
    mean = sum(sample) / len(sample)
    variance = sum((value - mean) ** 2 for value in sample) / max(len(sample), 1)
    std = variance ** 0.5
    if std == 0:
        return 0.0
    return float((mean / std) * (252.0 ** 0.5))


def CVaR(returns: Sequence[float], alpha: float = 0.05) -> float:
    if not returns:
        return 0.0
    ordered = sorted(float(value) for value in returns)
    cutoff = max(1, int(len(ordered) * alpha))
    tail = ordered[:cutoff]
    return abs(sum(tail) / max(len(tail), 1))


def herfindahl_index(weights: Sequence[float]) -> float:
    return float(sum(float(weight) ** 2 for weight in weights))


def compute_reward(portfolio: PortfolioSnapshot, lambdas: Any, t: int) -> float:
    # Core return
    pnl = portfolio_return(t, portfolio.returns)

    # Sharpe penalty (rolling 30-day)
    sharpe_pen = -max(0.0, TARGET_SHARPE - rolling_sharpe(t, portfolio.returns, 30))

    # Tail risk - CVaR at 95th percentile
    cvar_pen = -float(lambdas.cvar) * CVaR(portfolio.returns, alpha=0.05)

    # Transaction cost penalty (actual, not estimated)
    tc_pen = -float(portfolio.transaction_costs_this_step)

    # Drawdown penalty - non-linear (gets worse as DD deepens)
    dd = float(portfolio.current_drawdown)
    dd_pen = -float(lambdas.dd) * (dd ** 2)

    # Concentration penalty (Herfindahl index)
    hhi = sum(float(weight) ** 2 for weight in portfolio.weights)
    conc_pen = -float(lambdas.hhi) * hhi

    return float(pnl + sharpe_pen + cvar_pen + tc_pen + dd_pen + conc_pen)


def compute_reward_breakdown(portfolio: PortfolioSnapshot, lambdas: Any, t: int) -> RewardBreakdown:
    pnl = portfolio_return(t, portfolio.returns)
    sharpe_pen = -max(0.0, TARGET_SHARPE - rolling_sharpe(t, portfolio.returns, 30))
    cvar_pen = -float(lambdas.cvar) * CVaR(portfolio.returns, alpha=0.05)
    tc_pen = -float(portfolio.transaction_costs_this_step)
    dd = float(portfolio.current_drawdown)
    dd_pen = -float(lambdas.dd) * (dd ** 2)
    hhi = herfindahl_index(portfolio.weights)
    conc_pen = -float(lambdas.hhi) * hhi
    total_reward = pnl + sharpe_pen + cvar_pen + tc_pen + dd_pen + conc_pen
    return RewardBreakdown(
        pnl=float(pnl),
        sharpe_pen=float(sharpe_pen),
        cvar_pen=float(cvar_pen),
        tc_pen=float(tc_pen),
        dd_pen=float(dd_pen),
        conc_pen=float(conc_pen),
        total_reward=float(total_reward),
    )
