# Strategy Examples

Examples demonstrating trading strategies and backtesting.

## Files

### `demo_crypto_paper_trading.py`
Paper trading demo for cryptocurrency strategies.
- Uses fake money to simulate trading
- Tests multiple strategies simultaneously
- Safe for testing without risk

**Run**: `python examples/strategies/demo_crypto_paper_trading.py`

---

### `demo_live_trading.py`
Live trading demo (requires real broker API keys).
- Connects to real exchanges (Alpaca, Binance, Coinbase)
- Executes real trades
- **WARNING**: Uses real money!

**Run**: `python examples/strategies/demo_live_trading.py`

---

### `simple_backtest_demo.py`
Simple backtesting example.
- Tests strategies on historical data
- Quick performance evaluation
- No API keys needed

**Run**: `python examples/strategies/simple_backtest_demo.py`

---

### `run_trading_demo.py`
General trading system demo.
- Shows overall system capabilities
- Good starting point for new users

**Run**: `python examples/strategies/run_trading_demo.py`

---

## Quick Start

```bash
# Start with simple backtest (no API keys needed)
python examples/strategies/simple_backtest_demo.py

# Then try paper trading (safe, fake money)
python examples/strategies/demo_crypto_paper_trading.py

# Finally, live trading (requires API keys and USES REAL MONEY)
python examples/strategies/demo_live_trading.py
```

## Strategies Demonstrated

All examples showcase these strategies:
1. Mean Reversion
2. Momentum
3. RSI
4. MACD
5. Bollinger Bands
6. ML Ensemble (if dependencies installed)
7. And more...

---

**Tip**: Use `python start.py` from the root directory for a unified experience!
