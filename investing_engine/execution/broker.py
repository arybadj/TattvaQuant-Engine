from __future__ import annotations

from dataclasses import dataclass

from investing_engine.models import DecisionContext, OrderRequest


@dataclass
class ExecutionEngine:
    min_trade_threshold: float = 0.01

    def build_orders(self, decisions: list[DecisionContext]) -> list[OrderRequest]:
        orders: list[OrderRequest] = []
        for decision in decisions:
            turnover = abs(decision.target_weight - decision.current_weight)
            trade_required = turnover >= self.min_trade_threshold and decision.reward_estimate > 0.0
            orders.append(
                OrderRequest(
                    symbol=decision.symbol,
                    as_of_date=decision.as_of_date,
                    current_weight=decision.current_weight,
                    target_weight=decision.target_weight if trade_required else decision.current_weight,
                    turnover=turnover if trade_required else 0.0,
                    estimated_cost=decision.estimated_cost,
                    trade_required=trade_required,
                )
            )
        return orders
