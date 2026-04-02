from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from investing_engine.models import FeatureVector


@dataclass
class FeatureStore:
    root: Path
    redis_url: str | None = None

    def persist(self, features: list[FeatureVector]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"features_{features[0].as_of_date.isoformat()}.parquet"
        rows = [feature.model_dump(mode="json") for feature in features]
        try:
            import pandas as pd

            frame = pd.DataFrame(rows)
            frame.to_parquet(target, index=False)
        except Exception:
            target = self.root / f"features_{features[0].as_of_date.isoformat()}.json"
            target.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        self._persist_online(rows)
        return target

    def _persist_online(self, rows: list[dict[str, Any]]) -> None:
        if not self.redis_url:
            return
        try:
            import redis
        except ImportError:
            return
        try:
            client = redis.Redis.from_url(self.redis_url)
            for row in rows:
                key = f"feature:{row['as_of_date']}:{row['symbol']}"
                client.hset(key, mapping={field: json.dumps(value) for field, value in row.items()})
        except Exception:
            return

    def load_rows(self, as_of_date: str) -> list[dict[str, Any]]:
        parquet_path = self.root / f"features_{as_of_date}.parquet"
        json_path = self.root / f"features_{as_of_date}.json"
        if parquet_path.exists():
            try:
                import polars as pl

                return pl.read_parquet(parquet_path).to_dicts()
            except Exception:
                pass
        if json_path.exists():
            return json.loads(json_path.read_text(encoding="utf-8"))
        return []
