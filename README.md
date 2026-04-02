[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=github-actions&logoColor=white)](.github/workflows/ci.yml)

# Institution-Grade AI Investing Engine

Modular quant + AI hybrid investing system prototype built around strict point-in-time controls, exposed through a FastAPI service and packaged for local Docker-based development.

## Architecture

```text
Client / cURL / Postman
          |
          v
FastAPI app (src.api.app)
  - GET  /health
  - GET  /portfolio
  - POST /inference
  - POST /decide
          |
          v
LivePipeline (src.execution.execution_engine)
  -> TimeGate point-in-time data access
  -> FeatureStore feature materialization
  -> Parallel intelligence layer
     - market model
     - event model
     - fundamental model
  -> Regime classification + attention fusion
  -> Uncertainty engine
  -> Portfolio environment + execution engine
  -> PaperTradingBroker + FinalDecision response
          |
          +--> Redis (feature/cache support)
          |
          +--> Postgres (feedback audit persistence)
```

## Setup

1. Create a local environment:

```bash
python -m venv .venv
```

2. Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create your environment file:

```bash
cp .env.example .env
```

5. Start the API locally:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

## Run With Docker

1. Create `.env` from `.env.example`.
2. Build and start the stack:

```bash
docker compose up --build
```

This starts:

- `engine` on `http://localhost:8000`
- `postgres` on `localhost:5432`
- `redis` on `localhost:6379`

To stop everything:

```bash
docker compose down
```

To remove the Postgres volume as well:

```bash
docker compose down -v
```

## API Endpoints

Health check:

```bash
curl http://localhost:8000/health
```

Portfolio state:

```bash
curl http://localhost:8000/portfolio
```

Inference request:

```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "decision": "BUY",
    "confidence": 0.81,
    "position_size": 0.18,
    "expected_return_min": 0.02,
    "expected_return_max": 0.09,
    "expected_return_median": 0.05,
    "fundamentals_reason": "Margins and cash flow remain resilient.",
    "market_reason": "Momentum and trend remain constructive.",
    "sentiment_reason": "Recent event flow is supportive.",
    "regime_reason": "The regime is stable and risk is moderate.",
    "risk_factors": ["regime:neutral"],
    "exit_conditions": ["Review on regime change"],
    "estimated_cost_bps": 12.5,
    "shift_warning": false,
    "recommended_range": "3-6 months",
    "dynamic_adjustment": true
  }'
```

Decision request:

```bash
curl -X POST http://localhost:8000/decide \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "as_of_date": "2026-03-30"
  }'
```

## CI Pipeline

GitHub Actions runs on:

- Every push to `main`
- Every pull request

The workflow includes:

- `lint` with `ruff check src/` and `black --check src/`
- `test` with `pytest tests/ --no-cov -x` on Python 3.11
- `docker-build` that builds the image, starts the container, and checks `GET /health`

## Notes

- The Docker image uses `python:3.11-slim` for a stable CI/runtime base.
- The container entrypoint is `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`.
- `TimeGate` remains the core guardrail for point-in-time correctness across the pipeline.
