"""Fundamental-model ensemble and typed output signals."""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.data.timegate import LookaheadError, TimeGate

PIOTROSKI_COMPONENT_COLUMNS = [
    "piotroski_positive_roa",
    "piotroski_positive_cfo",
    "piotroski_delta_roa",
    "piotroski_cfo_gt_net_income",
    "piotroski_lower_leverage",
    "piotroski_higher_current_ratio",
    "piotroski_no_new_shares",
    "piotroski_higher_gross_margin",
    "piotroski_higher_asset_turnover",
]

QUALITY_FEATURE_COLUMNS = [
    *PIOTROSKI_COMPONENT_COLUMNS,
    "roe_3y_average",
    "free_cash_flow_yield",
    "debt_to_equity_delta",
    "revenue_growth_yoy",
    "ebitda_margin",
    "earnings_surprise_pct",
]

NON_FEATURE_KEYS = {"symbol", "as_of_date", "available_at", "industry_context"}


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _sigmoid(value: float, scale: float = 1.0) -> float:
    scaled = max(min(value / max(scale, 1e-9), 60.0), -60.0)
    return float(1.0 / (1.0 + math.exp(-scaled)))


@dataclass
class CompanyQualitySignal:
    fundamental_score: float
    valuation_score: float
    health_score: float


@dataclass
class IndustryHistorySignal:
    industry_cagr_5y: float
    stability_score: float


@dataclass
class FutureIndustrySignal:
    future_growth_score: float
    disruption_risk: float
    macro_tailwind: float


@dataclass
class FundamentalSignal:
    long_term_strength: float
    growth_potential: float
    risk_score: float
    company_quality: CompanyQualitySignal
    industry_history: IndustryHistorySignal
    future_industry: FutureIndustrySignal
    fundamental_score: float = 0.5
    valuation_score: float = 0.5
    financial_health: float = 0.5

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WalkForwardFoldResult:
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    n_train: int
    n_test: int
    sharpe: float
    max_train_available_at: datetime
    no_lookahead: bool


@dataclass
class WalkForwardBacktestResult:
    sharpe_series: list[float]
    mean_sharpe: float
    folds: list[WalkForwardFoldResult]


class CompanyQualityModel:
    train_months = 36
    test_months = 3
    step_months = 1

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.regressor = None
        self.classifier = None
        self.prediction_bounds = (-0.05, 0.05)
        self.fitted = False

    def fit(self, frame: pd.DataFrame, target_column: str = "forward_30d_return") -> CompanyQualityModel:
        data = self._prepare_frame(frame, require_targets=True, target_column=target_column)
        x_train = data.loc[:, QUALITY_FEATURE_COLUMNS]
        y_reg = data[target_column].astype(float)
        y_cls = data["direction_label"].astype(int)
        self.regressor, self.classifier = self._build_models(y_reg=y_reg, y_cls=y_cls)
        self.regressor.fit(x_train, y_reg)
        self.classifier.fit(x_train, y_cls)
        in_sample = np.asarray(self.regressor.predict(x_train), dtype=float)
        self.prediction_bounds = (float(np.quantile(in_sample, 0.05)), float(np.quantile(in_sample, 0.95)))
        self.fitted = True
        return self

    def predict(self, features: dict[str, float]) -> CompanyQualitySignal:
        return self.predict_company_quality(features)

    def predict_company_quality(self, features: dict[str, float]) -> CompanyQualitySignal:
        row = self._feature_row(features)
        if any(pd.isna(value) for value in row.values()):
            raise ValueError("Feature rows must not contain NaN values.")
        if self.fitted:
            frame = pd.DataFrame([row]).loc[:, QUALITY_FEATURE_COLUMNS]
            reg_pred = float(np.asarray(self.regressor.predict(frame), dtype=float)[0])
            cls_prob = float(self._positive_probability(frame)[0])
            low, high = self.prediction_bounds
            model_score = 0.5 if math.isclose(low, high) else _clip01((reg_pred - low) / (high - low))
            fundamental_score = _clip01((0.6 * model_score) + (0.4 * cls_prob))
        else:
            fundamental_score = _clip01(
                (row["piotroski_f_score"] / 9.0) * 0.35
                + _sigmoid(row["roe_3y_average"], 0.12) * 0.20
                + _sigmoid(row["free_cash_flow_yield"], 0.08) * 0.20
                + _sigmoid(row["revenue_growth_yoy"], 0.12) * 0.15
                + _sigmoid(row["earnings_surprise_pct"], 0.10) * 0.10
            )
        valuation_score = _clip01(
            (
                _sigmoid(row["free_cash_flow_yield"], 0.08)
                + _sigmoid(row["roe_3y_average"], 0.12)
                + _sigmoid(row["ebitda_margin"], 0.18)
                + _sigmoid(row["earnings_surprise_pct"], 0.10)
            )
            / 4.0
        )
        health_score = _clip01(
            (
                (row["piotroski_f_score"] / 9.0)
                + (1.0 - _sigmoid(row["debt_to_equity_delta"], 0.12))
                + _sigmoid(row["revenue_growth_yoy"], 0.12)
                + _sigmoid(row["ebitda_margin"], 0.18)
            )
            / 4.0
        )
        return CompanyQualitySignal(fundamental_score=fundamental_score, valuation_score=valuation_score, health_score=health_score)

    def predict_signal(self, features: dict[str, float], industry_context: str = "") -> FundamentalSignal:
        company = self.predict_company_quality(features)
        history = IndustryHistoryModel().predict(features)
        future = FutureIndustryModel(api_key=None).predict(features, industry_context=industry_context)
        return FundamentalSignal(
            long_term_strength=_clip01((company.fundamental_score * 0.5) + (company.health_score * 0.3) + (history.stability_score * 0.2)),
            growth_potential=_clip01((company.valuation_score * 0.25) + (history.industry_cagr_5y * 0.25) + (future.future_growth_score * 0.5)),
            risk_score=_clip01((1.0 - company.health_score) * 0.5 + future.disruption_risk * 0.35 + (1.0 - future.macro_tailwind) * 0.15),
            company_quality=company,
            industry_history=history,
            future_industry=future,
            fundamental_score=company.fundamental_score,
            valuation_score=company.valuation_score,
            financial_health=company.health_score,
        )

    def build_training_frame(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        price_source: str = "auto",
        price_frame: pd.DataFrame | None = None,
        fundamentals_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        fundamentals = fundamentals_frame.copy() if fundamentals_frame is not None else self.generate_synthetic_fundamentals(symbols, start_date, end_date)
        fundamentals = self._prepare_frame(fundamentals, require_targets=False)
        prices = self._load_prices(symbols, start_date - timedelta(days=10), end_date + timedelta(days=45), price_source, price_frame, fundamentals)
        return self._attach_targets(fundamentals, prices)

    def walk_forward_validate(self, frame: pd.DataFrame, target_column: str = "forward_30d_return") -> WalkForwardBacktestResult:
        data = self._prepare_frame(frame, require_targets=True, target_column=target_column)
        periods = sorted(data["as_of_date"].dt.to_period("M").unique())
        if len(periods) < self.train_months + self.test_months:
            raise ValueError("Dataset does not contain enough monthly history for walk-forward validation.")
        folds: list[WalkForwardFoldResult] = []
        sharpes: list[float] = []
        for anchor in range(self.train_months, len(periods) - self.test_months + 1, self.step_months):
            train_periods = periods[anchor - self.train_months : anchor]
            test_periods = periods[anchor : anchor + self.test_months]
            train = data.loc[data["as_of_date"].dt.to_period("M").isin(train_periods)].copy()
            test = data.loc[data["as_of_date"].dt.to_period("M").isin(test_periods)].copy()
            train_end = train_periods[-1].to_timestamp(how="end").to_pydatetime().replace(microsecond=0)
            if bool((train["available_at"] > pd.Timestamp(train_end)).any()):
                bad = train.loc[train["available_at"] > pd.Timestamp(train_end)].iloc[0]
                raise LookaheadError(f"training fold includes unavailable data: {bad['symbol']} {pd.Timestamp(bad['available_at']).isoformat()}")
            fold_model = CompanyQualityModel(random_state=self.random_state + anchor).fit(train, target_column=target_column)
            x_test = test.loc[:, QUALITY_FEATURE_COLUMNS]
            probs = fold_model._positive_probability(x_test)
            reg = np.asarray(fold_model.regressor.predict(x_test), dtype=float)
            realized = test[target_column].to_numpy(dtype=float)
            strategy = np.where(probs >= 0.5, 1.0, -1.0) * np.clip(np.abs(probs - 0.5) * 2.0, 0.25, 1.0) * (1.0 + np.clip(reg, -0.2, 0.2)) * realized
            vol = float(strategy.std(ddof=0))
            sharpe = 0.0 if vol < 1e-12 else float((strategy.mean() / vol) * math.sqrt(len(strategy)))
            sharpes.append(sharpe)
            folds.append(
                WalkForwardFoldResult(
                    fold=len(folds) + 1,
                    train_start=train["as_of_date"].min().to_pydatetime().replace(microsecond=0),
                    train_end=train_end,
                    test_start=test["as_of_date"].min().to_pydatetime().replace(microsecond=0),
                    test_end=test_periods[-1].to_timestamp(how="end").to_pydatetime().replace(microsecond=0),
                    n_train=int(len(train)),
                    n_test=int(len(test)),
                    sharpe=sharpe,
                    max_train_available_at=train["available_at"].max().to_pydatetime().replace(microsecond=0),
                    no_lookahead=True,
                )
            )
        return WalkForwardBacktestResult(sharpe_series=sharpes, mean_sharpe=float(np.mean(sharpes)), folds=folds)

    def generate_synthetic_fundamentals(self, symbols: list[str], start_date: date, end_date: date, seed: int | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state if seed is None else seed)
        months = pd.date_range(start=start_date, end=end_date, freq="MS")
        rows: list[dict[str, Any]] = []
        for symbol_index, symbol in enumerate(symbols):
            base = 0.45 + (0.05 * symbol_index)
            for step, as_of_date in enumerate(months):
                seasonal = math.sin((step + 1 + symbol_index) / 3.0)
                quality = base + (0.10 * seasonal) + rng.normal(0.0, 0.03)
                roe = float(np.clip(0.10 + (quality * 0.18) + rng.normal(0.0, 0.01), -0.10, 0.40))
                fcf = float(np.clip(0.03 + (quality * 0.08) + rng.normal(0.0, 0.01), -0.05, 0.20))
                debt = float(np.clip(-0.02 - (quality * 0.08) + rng.normal(0.0, 0.03), -0.40, 0.40))
                revenue = float(np.clip(0.05 + (quality * 0.14) + (0.02 * seasonal) + rng.normal(0.0, 0.015), -0.15, 0.45))
                margin = float(np.clip(0.12 + (quality * 0.15) + rng.normal(0.0, 0.012), 0.02, 0.45))
                surprise = float(np.clip(0.01 + (quality * 0.07) + rng.normal(0.0, 0.01), -0.15, 0.20))
                components = {
                    "piotroski_positive_roa": float(roe > 0.0),
                    "piotroski_positive_cfo": float(fcf > 0.0),
                    "piotroski_delta_roa": float(roe > 0.11),
                    "piotroski_cfo_gt_net_income": float(fcf > roe * 0.3),
                    "piotroski_lower_leverage": float(debt < 0.0),
                    "piotroski_higher_current_ratio": float(quality > 0.35),
                    "piotroski_no_new_shares": float(quality > 0.30),
                    "piotroski_higher_gross_margin": float(margin > 0.14),
                    "piotroski_higher_asset_turnover": float(revenue > 0.06),
                }
                rows.append(
                    {
                        "symbol": symbol,
                        "as_of_date": pd.Timestamp(as_of_date).normalize(),
                        "available_at": datetime.combine(pd.Timestamp(as_of_date).date(), time(hour=8)),
                        **components,
                        "piotroski_f_score": float(sum(components.values())),
                        "roe_3y_average": roe,
                        "free_cash_flow_yield": fcf,
                        "debt_to_equity_delta": debt,
                        "revenue_growth_yoy": revenue,
                        "ebitda_margin": margin,
                        "earnings_surprise_pct": surprise,
                        "sector_cagr_3y": float(np.clip(0.08 + (0.02 * seasonal), 0.0, 0.30)),
                        "sector_cagr_5y": float(np.clip(0.10 + (0.015 * math.cos((step + 1) / 5.0)), 0.0, 0.30)),
                        "cyclicality_flag": float(np.clip(0.45 - (0.25 * quality), 0.0, 1.0)),
                        "tam_growth_estimate": float(np.clip(0.12 + (0.05 * quality), 0.0, 0.40)),
                        "patent_filings_trend": float(np.clip(0.05 + (0.03 * quality), 0.0, 0.25)),
                        "rate_regime": float(np.clip(0.2 + math.cos((step + 1) / 4.0), -1.0, 1.0)),
                        "currency_momentum_usdinr": float(np.clip(rng.normal(0.01, 0.02), -0.10, 0.10)),
                    }
                )
        return pd.DataFrame(rows)

    def _prepare_frame(self, frame: pd.DataFrame, require_targets: bool, target_column: str = "forward_30d_return") -> pd.DataFrame:
        data = frame.copy()
        for column in ("symbol", "as_of_date", "available_at"):
            if column not in data.columns:
                raise ValueError(f"Missing required column: {column}")
        data["as_of_date"] = pd.to_datetime(data["as_of_date"]).dt.normalize()
        data["available_at"] = pd.to_datetime(data["available_at"])
        if "piotroski_f_score" not in data.columns and all(column in data.columns for column in PIOTROSKI_COMPONENT_COLUMNS):
            data["piotroski_f_score"] = data[PIOTROSKI_COMPONENT_COLUMNS].sum(axis=1).astype(float)
        if "piotroski_f_score" in data.columns:
            base_score = data["piotroski_f_score"].astype(float).clip(0.0, 9.0) / 9.0
            for column in PIOTROSKI_COMPONENT_COLUMNS:
                data[column] = data[column] if column in data.columns else base_score
        missing = [column for column in QUALITY_FEATURE_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        if data.loc[:, QUALITY_FEATURE_COLUMNS].isna().any().any():
            raise ValueError("Feature and target columns must not contain NaN values.")
        TimeGate().validate_no_lookahead(data.loc[:, ["symbol", "as_of_date", "available_at", *QUALITY_FEATURE_COLUMNS]].copy())
        if require_targets:
            if target_column not in data.columns:
                raise ValueError(f"Missing required target column: {target_column}")
            data["direction_label"] = (data[target_column].astype(float) > 0.0).astype(int) if "direction_label" not in data.columns else data["direction_label"].astype(int)
            if data.loc[:, [target_column, "direction_label"]].isna().any().any():
                raise ValueError("Feature and target columns must not contain NaN values.")
        return data.sort_values(["as_of_date", "symbol"]).reset_index(drop=True)

    def _feature_row(self, features: dict[str, float]) -> dict[str, float]:
        row = {key: float(value) for key, value in features.items() if key not in NON_FEATURE_KEYS}
        score = float(row.get("piotroski_f_score", sum(row.get(column, 0.0) for column in PIOTROSKI_COMPONENT_COLUMNS)))
        row["piotroski_f_score"] = score
        for column in PIOTROSKI_COMPONENT_COLUMNS:
            row.setdefault(column, score / 9.0)
        for column in QUALITY_FEATURE_COLUMNS:
            row.setdefault(column, 0.0)
        row.setdefault("sector_cagr_3y", 0.0)
        row.setdefault("sector_cagr_5y", row["sector_cagr_3y"])
        row.setdefault("cyclicality_flag", 0.0)
        row.setdefault("tam_growth_estimate", 0.0)
        row.setdefault("patent_filings_trend", 0.0)
        row.setdefault("rate_regime", 0.0)
        row.setdefault("currency_momentum_usdinr", 0.0)
        return row

    def _build_models(self, y_reg: pd.Series, y_cls: pd.Series) -> tuple[Any, Any]:
        try:
            from xgboost import XGBClassifier, XGBRegressor

            regressor = XGBRegressor(objective="reg:squarederror", n_estimators=180, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=self.random_state, verbosity=0)
            classifier = XGBClassifier(objective="binary:logistic", n_estimators=160, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=self.random_state, eval_metric="logloss", verbosity=0)
        except ImportError:  # pragma: no cover
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

            regressor = GradientBoostingRegressor(random_state=self.random_state)
            classifier = GradientBoostingClassifier(random_state=self.random_state)
        if y_cls.nunique() < 2:
            from sklearn.dummy import DummyClassifier

            classifier = DummyClassifier(strategy="constant", constant=int(y_cls.iloc[0]))
        return regressor, classifier

    def _positive_probability(self, frame: pd.DataFrame) -> np.ndarray:
        probs = np.asarray(self.classifier.predict_proba(frame), dtype=float)
        if probs.ndim == 2 and probs.shape[1] == 2:
            return probs[:, 1]
        constant = int(getattr(self.classifier, "classes_", np.array([0]))[-1])
        return np.full(len(frame), float(constant), dtype=float)

    def _load_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        price_source: str,
        price_frame: pd.DataFrame | None,
        fundamentals: pd.DataFrame,
    ) -> pd.DataFrame:
        if price_frame is not None:
            prices = price_frame.copy()
            prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
            return prices.sort_values(["symbol", "date"]).reset_index(drop=True)
        if price_source in {"auto", "yfinance"}:
            try:
                import yfinance as yf

                raw = yf.download(symbols, start=start_date.isoformat(), end=(end_date + timedelta(days=1)).isoformat(), auto_adjust=True, progress=False, threads=False)
                close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
                if isinstance(close, pd.Series):
                    close = close.to_frame(name=symbols[0])
                if "Close" in close.columns and len(symbols) == 1:
                    close = close.rename(columns={"Close": symbols[0]})
                rows = [{"symbol": str(symbol), "date": pd.Timestamp(index).normalize(), "close": float(value)} for symbol in close.columns for index, value in close[symbol].dropna().items()]
                if rows:
                    return pd.DataFrame(rows)
                if price_source == "yfinance":
                    raise ValueError("No price data returned by yfinance.")
            except Exception:
                if price_source == "yfinance":
                    raise
        rng = np.random.default_rng(self.random_state + 101)
        rows = []
        for symbol, group in fundamentals.groupby("symbol"):
            history = group.sort_values("as_of_date")
            price = 100.0 + (hash(symbol) % 25)
            for day_index, day in enumerate(pd.bdate_range(start=start_date, end=end_date)):
                latest = history.loc[history["as_of_date"] <= day]
                driver = 0.5 if latest.empty else float((latest["piotroski_f_score"].iloc[-1] / 9.0) + latest["roe_3y_average"].iloc[-1] + latest["free_cash_flow_yield"].iloc[-1])
                price *= max(0.80, 1.0 + 0.0002 + (driver * 0.0007) + (0.0004 * math.sin(day_index / 21.0)) + rng.normal(0.0, 0.007))
                rows.append({"symbol": str(symbol), "date": pd.Timestamp(day).normalize(), "close": float(price)})
        return pd.DataFrame(rows)

    def _attach_targets(self, fundamentals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
        for symbol, price_group in prices.groupby("symbol"):
            calendar = price_group.set_index("date")["close"].sort_index()
            aligned = calendar.reindex(pd.date_range(calendar.index.min(), calendar.index.max(), freq="D")).ffill()
            for record in fundamentals.loc[fundamentals["symbol"] == symbol].to_dict(orient="records"):
                as_of = pd.Timestamp(record["as_of_date"]).normalize()
                exit_date = as_of + pd.Timedelta(days=30)
                if as_of not in aligned.index or exit_date not in aligned.index:
                    continue
                start_price = float(aligned.loc[as_of])
                end_price = float(aligned.loc[exit_date])
                if start_price <= 0.0:
                    continue
                forward_return = (end_price / start_price) - 1.0
                rows.append({**record, "forward_30d_return": float(forward_return), "direction_label": int(forward_return > 0.0)})
        return self._prepare_frame(pd.DataFrame(rows), require_targets=True)


class IndustryHistoryModel:
    def predict(self, features: dict[str, float]) -> IndustryHistorySignal:
        cagr_3y = float(features.get("sector_cagr_3y", 0.0))
        cagr_5y = float(features.get("sector_cagr_5y", cagr_3y))
        cyclicality_flag = float(features.get("cyclicality_flag", 0.0))
        stability_score = max(min(1.0 - cyclicality_flag + min(cagr_5y, 0.5), 1.0), 0.0)
        return IndustryHistorySignal(industry_cagr_5y=cagr_5y, stability_score=stability_score)


class FutureIndustryModel:
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        api_base: str = "https://api.openai.com/v1/responses",
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base

    def predict(self, features: dict[str, float], industry_context: str) -> FutureIndustrySignal:
        if not self.api_key:
            return self._heuristic_predict(features, industry_context)
        prompt = (
            "Analyze industry outlook from macro features and context. "
            "Return JSON with future_growth_score, disruption_risk, macro_tailwind as floats in [0,1]."
        )
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": prompt}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps({"macro_features": features, "industry_context": industry_context}),
                        }
                    ],
                },
            ],
        }
        request = urllib.request.Request(
            self.api_base,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = json.loads(response.read().decode("utf-8"))
            text = raw.get("output", [{}])[0].get("content", [{}])[0].get("text", "{}")
            parsed = json.loads(text)
            return FutureIndustrySignal(
                future_growth_score=float(parsed.get("future_growth_score", 0.5)),
                disruption_risk=float(parsed.get("disruption_risk", 0.5)),
                macro_tailwind=float(parsed.get("macro_tailwind", 0.5)),
            )
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError):
            return self._heuristic_predict(features, industry_context)

    def _heuristic_predict(self, features: dict[str, float], industry_context: str) -> FutureIndustrySignal:
        tam_trend = float(features.get("tam_growth_estimate", 0.0))
        patent_trend = float(features.get("patent_filings_trend", 0.0))
        rate_regime = float(features.get("rate_regime", 0.0))
        currency_momentum = float(features.get("currency_momentum_usdinr", 0.0))
        context_penalty = 0.15 if "disruption" in industry_context.lower() else 0.0
        future_growth_score = max(min(0.5 + tam_trend * 0.4 + patent_trend * 0.3 + rate_regime * 0.1, 1.0), 0.0)
        disruption_risk = max(min(0.4 + context_penalty - patent_trend * 0.2, 1.0), 0.0)
        macro_tailwind = max(min(0.5 + rate_regime * 0.15 - currency_momentum * 0.1, 1.0), 0.0)
        return FutureIndustrySignal(
            future_growth_score=future_growth_score,
            disruption_risk=disruption_risk,
            macro_tailwind=macro_tailwind,
        )


class FundamentalModelEnsemble:
    def __init__(
        self,
        company_quality: CompanyQualityModel | None = None,
        industry_history: IndustryHistoryModel | None = None,
        future_industry: FutureIndustryModel | None = None,
    ) -> None:
        self.company_quality = company_quality or CompanyQualityModel()
        self.industry_history = industry_history or IndustryHistoryModel()
        self.future_industry = future_industry or FutureIndustryModel()

    def predict(self, feature_row: dict[str, float], industry_context: str = "") -> FundamentalSignal:
        company = self.company_quality.predict_company_quality(feature_row)
        history = self.industry_history.predict(feature_row)
        future = self.future_industry.predict(feature_row, industry_context=industry_context)
        long_term_strength = _clip01((company.fundamental_score * 0.45) + (company.health_score * 0.25) + (history.stability_score * 0.30))
        growth_potential = _clip01((history.industry_cagr_5y * 0.35) + (future.future_growth_score * 0.45) + (future.macro_tailwind * 0.20))
        risk_score = _clip01((1.0 - company.health_score) * 0.35 + future.disruption_risk * 0.40 + abs(feature_row.get("debt_to_equity_delta", 0.0)) * 0.25)
        return FundamentalSignal(
            long_term_strength=float(long_term_strength),
            growth_potential=float(growth_potential),
            risk_score=float(risk_score),
            company_quality=company,
            industry_history=history,
            future_industry=future,
            fundamental_score=float(company.fundamental_score),
            valuation_score=float(company.valuation_score),
            financial_health=float(company.health_score),
        )
