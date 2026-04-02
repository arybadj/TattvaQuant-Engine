from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class EventPublisher:
    bootstrap_servers: str
    topic: str = "investing-engine-events"

    def publish(self, payload: dict[str, Any]) -> None:
        try:
            from kafka import KafkaProducer
        except ImportError:
            return

        try:
            producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            )
            producer.send(self.topic, payload)
            producer.flush(timeout=5)
            producer.close()
        except Exception:
            return
