# Phase 1: Build the Base System

## 📌 Purpose
Create a modular, stable, automated trading AI foundation that can:
- Pull financial data
- Engineer features
- Train a machine learning model
- Generate basic trading signals
- Backtest performance

---

<!-- Reminder: Remove cached versions of `trading-ai.code-workspace` and `phase_0_command_center.md` from the repository if they are not meant to be tracked. -->

## 🎯 Major Deliverables
- Data ingestion pipeline (`fetch_data.py`)
- Feature engineering pipeline (`feature_generator.py`)
- Modeling pipeline (`train_model.py`)
- Strategy execution logic (`simple_strategy.py`)
- Backtesting engine (`backtest.py`)
- Logging system for predictions and trades
- Daily retrain job (`daily_retrain.py`)
- Simple scheduler (`scheduler.py`)

---

## 🛠️ Tools / Tech Required
- Python 3.11+
- pandas
- numpy
- scikit-learn
- yfinance
- matplotlib
- joblib
- schedule (for job automation)

---

## 🗺️ Step-by-Step Plan
1. Build `/src/data_ingestion/fetch_data.py`
2. Build `/src/feature_engineering/feature_generator.py`
3. Build `/src/modeling/train_model.py`
4. Build `/src/strategy/simple_strategy.py`
5. Build `/src/backtesting/backtest.py`
6. Build `/src/execution/daily_retrain.py`
7. Build `/src/execution/scheduler.py`
8. Log model outputs and backtest results daily to `/logs/`
9. Validate with a single ticker (e.g., AAPL)
10. Paper trade predictions internally (no live broker yet)

---

## ✅ Success Criteria
- Daily model retraining is successful 95%+ of the time
- Predictions generate clean buy/sell outputs
- Backtests run without manual intervention
- Logs capture predictions and trade outcomes
- System is modular (no hardcoded spaghetti scripts)

---

## ⚠️ Risks & How to Handle
| Risk | Solution |
|:-----|:---------|
| Overfitting model to tiny datasets | Use rolling train/test splits, validate generalization |
| API downtime | Build retry logic into data ingestion |
| Code bloating | Enforce modular, single-responsibility principle on all scripts |

---

> Phase 1 is complete when you have a self-retraining, fully logged AI that generates and backtests signals daily.