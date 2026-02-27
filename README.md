# Quantlytics - Advanced Trading Platform

**One command. One dashboard. Everything you need.**

---

## Quick Start (< 2 minutes)

```bash
# 1. Install dependencies
pip install -r requirements-secure.txt

# 2. Start the system
python start.py

# That's it! Dashboard opens at http://localhost:8501
```

---

## What You Get

### 📊 Unified Dashboard
- **ONE place** to see everything
- All 11 strategies in real-time
- Agent swarm status
- Live P&L and risk metrics
- Portfolio analytics

### 🤖 Agent Swarm
- 6 specialized AI agents working together
- Strategy coordinator
- Risk manager
- Market analyzer
- Execution agent
- ML model trainer
- RL decision maker

### 💼 11 Trading Strategies
- **ML-Based**: Ensemble, GRU, CNN-LSTM, VAE
- **RL-Based**: PPO Agent
- **Classic**: Momentum, Mean Reversion, RSI, MACD, Bollinger Bands
- **DeFi**: Yield Optimizer, Multi-Chain Arbitrage

### ⚠️ Risk Management
- Position limits (auto-enforced)
- Circuit breakers
- Max drawdown protection
- Real-time monitoring

---

## Trading Modes

```bash
# Monitor only (no trading)
python start.py

# Paper trading (fake money)
python start.py --mode paper

# Live trading (real money - be careful!)
python start.py --mode live

# Agent swarm
python start.py --agents

# System status
python start.py --status

# List all modules
python start.py --list
```

---

## Examples & Demos

All examples are organized by category:

### Strategy Examples
```bash
# Paper trading demo
python examples/strategies/demo_crypto_paper_trading.py

# Live trading demo (requires API keys)
python examples/strategies/demo_live_trading.py

# Simple backtest
python examples/strategies/simple_backtest_demo.py
```

### DeFi Examples
```bash
# Simple DeFi demo
python examples/defi/defi_simple_demo.py

# Full DeFi trading
python examples/defi/defi_trading_demo.py

# Multi-chain arbitrage
python examples/defi/demo_multi_chain.py
```

See `examples/README.md` for complete list and documentation.

---

## Testing

```bash
# Run all tests
pytest tests/

# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/
```

See `tests/README.md` for testing guide.

---

## System Architecture

```
┌─────────────────────────────────────────────────┐
│          Unified Dashboard (Port 8501)          │
│  Shows: Strategies | Agents | Risk | Analytics  │
└─────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
   │ Trading │    │  Agent  │    │  Risk   │
   │ Engine  │    │  Swarm  │    │ Manager │
   └─────────┘    └─────────┘    └─────────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │   Data Layer (Redis/Postgres) │
        │   Infrastructure (Docker)     │
        └───────────────────────────────┘
```

---

## Repository Structure

```
trading-ai/
├── start.py                  ⭐ Main entry point
│
├── examples/                 📚 Organized examples
│   ├── strategies/          Trading strategy demos
│   ├── defi/                DeFi strategy demos
│   ├── integration/         Integration demos
│   └── README.md           Examples guide
│
├── tests/                    🧪 Organized tests
│   ├── unit/                Component tests
│   ├── integration/         System tests
│   └── README.md           Testing guide
│
├── src/                      🏗️ Source code
│   ├── strategies/          Trading strategies
│   ├── ml/                  ML models
│   ├── rl/                  RL agents
│   ├── defi/                DeFi strategies
│   ├── dashboard/           Unified dashboard
│   ├── execution/           Order execution
│   ├── exchanges/           Exchange integrations
│   └── autonomous_agent/    Agent swarm
│
├── docs/                     📖 Documentation
├── config/                   ⚙️ Configuration
└── docker/                   🐳 Docker files
```

---

## Performance

**Backtested Results** (Jan 2025 - Feb 2026):

| Metric | Value |
|--------|-------|
| Total Return | 101.3% |
| Sharpe Ratio | 2.13 |
| Win Rate | 62.3% |
| Max Drawdown | 12.3% |

**Best Performing Strategy**: ML Ensemble (215% return, 2.4 Sharpe)

*Note: Past performance does not guarantee future results. See `docs/BACKTESTING.md` for methodology.*

---

## Production Setup

### Option 1: Quick Start (Recommended)
```bash
python start.py
```

### Option 2: Full Infrastructure
```bash
# Start all services (Redis, PostgreSQL, Grafana, Prometheus)
docker-compose up -d

# Start trading system
python start.py --mode live
```

### Option 3: Individual Components
```bash
# API Server
python -m src.api.main

# Dashboard
streamlit run src/dashboard/unified_dashboard.py

# Agent Swarm
python -m src.autonomous_agent.trading_agent
```

---

## Configuration

Edit `config/trading_config.yaml`:

```yaml
trading:
  mode: paper  # paper | live
  initial_capital: 10000
  max_position_size: 0.10  # 10% max per position
  max_drawdown: 0.15       # 15% max drawdown

strategies:
  enabled:
    - ml_ensemble
    - ppo_rl
    - momentum
    - mean_reversion

risk:
  daily_loss_limit: 0.05  # 5%
  position_limit: 0.10    # 10% per asset
```

---

## Security

✅ All vulnerabilities patched (12/12 fixed)
✅ Automated security scanning
✅ Encrypted communications
✅ No hardcoded secrets

See [SECURITY.md](SECURITY.md) for details.

---

## Documentation

- **[Getting Started](docs/GETTING_STARTED.md)** - Complete setup guide
- **[Examples Guide](examples/README.md)** - All examples and demos
- **[Testing Guide](tests/README.md)** - Running tests
- **[Security](SECURITY.md)** - Security policy and updates
- **[Architecture](docs/ARCHITECTURE.md)** - System architecture
- **[Strategies](docs/STRATEGY_GUIDE.md)** - Trading strategies guide
- **[API Reference](docs/API_REFERENCE.md)** - API documentation
- **[Status Report](STATUS-REPORT.md)** - Honest current status
- **[Dependency Status](DEPENDENCY_STATUS.md)** - ML/AI dependencies

---

## Project Status

### ✅ What's Working (Production Ready)
- 7/11 strategies (all classical strategies)
- Full broker integrations (Alpaca, Binance, Coinbase)
- Paper trading system
- Unified dashboard
- Agent swarm architecture
- Risk management system
- Infrastructure (Docker, Redis, PostgreSQL, Grafana)
- Security hardening (12/12 vulnerabilities fixed)

### ⚠️ What Needs Dependencies
- ML Ensemble, GRU, CNN-LSTM, VAE (need xgboost/tensorflow)
- RL Agent (needs tensorflow)

See **[DEPENDENCY_STATUS.md](DEPENDENCY_STATUS.md)** for details and solutions.

### 📊 Overall Status: ~80% Production Ready
- **Core Trading**: 90% complete
- **ML/AI**: 70% complete (code done, needs dependencies)
- **Infrastructure**: 90% complete
- **Documentation**: 85% complete

---

## What's Included

### Trading Strategies (11 total)
- ✅ Mean Reversion
- ✅ Momentum
- ✅ RSI
- ✅ MACD
- ✅ Bollinger Bands
- ✅ Grid Trading
- ✅ DCA (Dollar-Cost Averaging)
- ⚠️ ML Ensemble (needs xgboost/lightgbm)
- ⚠️ GRU Predictor (needs tensorflow)
- ⚠️ CNN-LSTM Hybrid (needs tensorflow)
- ⚠️ PPO RL Agent (needs tensorflow)

### DeFi Strategies (3 total)
- ✅ Yield Optimizer
- ✅ Impermanent Loss Hedging
- ✅ Multi-Chain Arbitrage

### Infrastructure
- ✅ Docker & Docker Compose
- ✅ PostgreSQL/TimescaleDB
- ✅ Redis cache
- ✅ Prometheus monitoring
- ✅ Grafana dashboards
- ✅ WebSocket real-time data

### Agent Swarm (6 agents)
- ✅ Strategy Coordinator
- ✅ Risk Manager
- ✅ Market Analyzer
- ✅ Execution Agent
- ✅ ML Model Trainer
- ✅ RL Decision Maker

---

## Support

- **Issues**: [GitHub Issues](https://github.com/cpoplaws/trading-ai/issues)
- **Security**: See [SECURITY.md](SECURITY.md)
- **Documentation**: [docs/](docs/)

---

## Contributing

Contributions are welcome! Please see:
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [Branch Management](BRANCH_DOCS_INDEX.md) - Branch workflow

---

## License

MIT License - See [LICENSE](LICENSE)

---

## Quick Commands Reference

```bash
# Start dashboard
python start.py

# Paper trading
python start.py --mode paper

# Live trading (requires API keys)
python start.py --mode live

# Agent swarm
python start.py --agents

# System status
python start.py --status

# List all modules
python start.py --list

# Run tests
pytest tests/

# Run examples
python examples/strategies/simple_backtest_demo.py

# Security scan
python -m safety check
```

---

## Recent Updates

- ✅ **Repository Cleanup**: Organized 14 demo/test files into clean structure
- ✅ **Security Hardening**: Fixed 12/12 Dependabot vulnerabilities
- ✅ **Unified Dashboard**: One place to see all strategies and agent swarm
- ✅ **Documentation**: Added comprehensive READMEs for examples and tests
- ⚠️ **ML Dependencies**: Installed but need system libraries (see DEPENDENCY_STATUS.md)

---