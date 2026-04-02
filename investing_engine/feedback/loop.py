from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from investing_engine.models import DecisionContext, FeedbackRecord, OrderRequest


@dataclass
class FeedbackLoop:
    root: Path

    def capture(self, decisions: list[DecisionContext], orders: list[OrderRequest]) -> list[FeedbackRecord]:
        if not decisions:
            return []
        self.root.mkdir(parents=True, exist_ok=True)
        order_map = {order.symbol: order for order in orders}
        feedback = []
        for decision in decisions:
            order = order_map[decision.symbol]
            realized_return = decision.expected_return - decision.estimated_cost - (decision.total_uncertainty * 0.01)
            record = FeedbackRecord(
                symbol=decision.symbol,
                as_of_date=decision.as_of_date,
                realized_return=realized_return,
                realized_cost=order.estimated_cost,
                reward=decision.reward_estimate,
                metadata={"trade_required": order.trade_required},
            )
            feedback.append(record)
        target = self.root / f"feedback_{decisions[0].as_of_date.isoformat()}.json"
        target.write_text(
            json.dumps([item.model_dump(mode="json") for item in feedback], indent=2),
            encoding="utf-8",
        )
        return feedback
