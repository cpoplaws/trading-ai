# Quantlytics

Quantlytics is a trading platform demo that packages research, paper trading flows, and operational tooling around the **Itera** internal engine.

> Status: **demoable and hardened prototype**, not a fully managed production SaaS.

## What this repo currently supports

- Streamlit dashboard for strategy/risk visibility.
- FastAPI service for market, portfolio, risk, and agent endpoints.
- Paper trading and backtest demos.
- Optional live-trading demo paths when credentials are provided.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-secure.txt
pip install -e .
```

## Launch modes

```bash
# Dashboard (default)
quantlytics --mode dashboard

# Paper trading demo
quantlytics --mode paper

# Live trading demo (requires credentials)
quantlytics --mode live

# Backtest flow
quantlytics --mode backtest

# Runtime status
quantlytics --mode status

# API only
quantlytics-api
```

Backward compatibility launcher:

```bash
python start.py --mode dashboard
```

## Modes and truthfulness

- **Demo/simulated features**: Many examples and dashboards use mocked or delayed data paths.
- **Paper trading**: Uses simulated execution with no real funds.
- **Live trading**: Available as an integration demo; users must provide broker keys and risk controls.

## Security baseline in this repo

- API keys are validated against configured key lists (`QUANTLYTICS_API_KEYS`).
- CORS is restricted via `QUANTLYTICS_CORS_ORIGINS`.
- Lightweight in-memory per-client rate limiting is enabled.
- Docker compose requires explicit secret env vars (no fallback default passwords).

See `SECURITY.md` for details and remaining limitations.

## Docker demo

```bash
cp .env.template .env
# set required secrets in .env
docker compose up --build
```

## Brand architecture

- External platform brand: **Quantlytics**
- Internal engine layer: **Itera** (e.g., Itera agents / Itera engine)

