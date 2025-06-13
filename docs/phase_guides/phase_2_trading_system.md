# Phase 2: Build the Trading System (Broker Connectivity)

## 📌 Purpose
Connect the AI model to a real brokerage account to start:
- Simulated paper trading (first)
- Small real-money live trading (later)
while enforcing risk management and safe order execution.

---

## 🎯 Major Deliverables
- Broker API integration (Alpaca first, IBKR later)
- `/execution/broker_interface.py` module
- Paper trading mode (safe environment)
- Real trading mode (small live testing)
- Simple order manager (open, cancel, modify orders)
- Portfolio tracker (real-time PnL, exposure)

---

## 🛠️ Tools / Tech Required
- Python `requests` for REST API
- Alpaca API keys (paper trading account)
- IBKR `ib_insync` library (later)
- GitHub branch: `feature/broker-interface`

---

## 🗺️ Step-by-Step Plan
1. Create `/src/execution/broker_interface.py`
2. Build order sending functions (buy, sell, limit, market)
3. Add error handling (timeouts, retries)
4. Create `/src/execution/portfolio_tracker.py`
5. Log all trades inside `/logs/trades.log`
6. Simulate paper trades for 1 week
7. (Optional) Start real trading with small capital

---

## ✅ Success Criteria
- Paper trades fire automatically based on AI model decisions
- No missed or duplicated trades
- Portfolio PnL tracked cleanly
- Switchable mode: "PAPER" vs "LIVE"

---

## ⚠️ Risks & How to Handle
| Risk | Solution |
|:-----|:---------|
| Double-firing trades | Add deduplication check |
| API limits reached | Implement rate limiter in broker interface |
| Unclear trade logs | Log every trade event (timestamp, symbol, side, qty, price) |

---
> Phase 2 is complete when the AI can execute trades automatically in a live broker environment — even if starting small.