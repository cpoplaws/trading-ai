# Trading-AI: Advanced Crypto/Web3 Trading Platform

Welcome to the Trading-AI Project — a comprehensive autonomous trading system now transformed into an **advanced crypto/Web3 platform** with multi-chain support, DeFi integration, on-chain analytics, and sophisticated crypto trading strategies.

## 📜 Project Vision

Build a scalable, modular AI trading system for **cryptocurrency and DeFi markets** that combines:
- 🌐 Multi-chain blockchain support (Ethereum, Polygon, Arbitrum, BSC, Avalanche, Base, Solana)
- 🔄 DEX aggregation for optimal trade execution
- 📊 On-chain analytics and whale tracking
- 🤖 Machine learning with crypto-specific features
- ⚡ Advanced strategies (funding rate arbitrage, cross-DEX arbitrage, yield optimization)
- 🛡️ Comprehensive risk management

> **📖 New Contributors?** Check out our [Branch Management Documentation Index](BRANCH_DOCS_INDEX.md) for a complete guide to contributing, branch workflow, and project standards.

## 🚀 Quick Start

### 🪙 Crypto Multi-Chain Demo

```bash
# Install crypto dependencies
pip install -r requirements.txt
pip install -r requirements-crypto.txt

# Run multi-chain demo
python demo_multi_chain.py
```

### Launch the Dashboard

```bash
# Start the real-time dashboard
./run_dashboard.sh
# Dashboard opens at http://localhost:8501
```

### Run Trading Demos

```bash
# Traditional stock trading
python demo_live_trading.py

# DeFi trading on BSC/PancakeSwap
python defi_trading_demo.py

# Multi-chain operations
python demo_multi_chain.py

# 🆕 Crypto paper trading & backtesting
python demo_crypto_paper_trading.py
```

### Option 1: Docker (Recommended for Production)

```bash
# Clone and setup
git clone https://github.com/cpoplaws/trading-ai.git
cd trading-ai

# Build and run
docker-compose up --build
```

### Option 3: Local Development

```bash
# Setup environment
./setup.sh

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python src/execution/daily_retrain.py
```

## 📊 Implementation Status

### 🌟 **NEW: CRYPTO/WEB3 PLATFORM** (40% Complete)

**Multi-Chain Infrastructure** ✅ (100%)
- ✅ Unified chain manager for 7+ blockchains
- ✅ Ethereum mainnet + L2s (Arbitrum, Optimism, Base)
- ✅ Polygon/MATIC integration
- ✅ Binance Smart Chain (BSC)
- ✅ Avalanche C-Chain
- ✅ Solana blockchain support
- ✅ Gas estimation and optimization
- ✅ Multi-chain balance queries

**Crypto Data Sources** ✅ (70%)
- ✅ Binance API (spot + futures, funding rates, liquidations)
- ✅ CoinGecko API (prices, market data, trending coins)
- ✅ Fear & Greed Index
- ✅ 24h statistics and order books
- ✅ Open interest and long/short ratios
- ⚠️ Dune Analytics, The Graph (planned)
- ⚠️ Glassnode on-chain metrics (planned)

**DEX Aggregation** ⚡ (40%)
- ✅ DEX aggregator framework (1inch, Paraswap, 0x ready)
- ✅ Cross-DEX price comparison
- ✅ Arbitrage opportunity detection
- ✅ Price impact calculation
- ✅ Optimal route finding
- ⚠️ Uniswap V3, Curve, Balancer (in progress)

**Crypto Strategies** ⚡ (30%)
- ✅ Funding rate arbitrage (perpetual futures)
- ✅ Position sizing and risk management
- ✅ Signal generation and backtesting
- ⚠️ Cross-exchange arbitrage (planned)
- ⚠️ Grid trading, yield optimization (planned)
- ⚠️ Whale following, liquidation hunting (planned)

**On-Chain Analytics** ⚡ (25%)
- ✅ Wallet tracker for whale monitoring
- ✅ Large transaction detection
- ✅ Wallet behavior analysis
- ⚠️ Smart money detection (planned)
- ⚠️ Token flow analyzer (planned)
- ⚠️ Rug pull detector (planned)

**Crypto ML Features** ✅ (60%)
- ✅ NVT Ratio (Network Value to Transactions)
- ✅ MVRV (Market Value to Realized Value)
- ✅ SOPR (Spent Output Profit Ratio)
- ✅ Funding rate momentum
- ✅ Exchange netflow analysis
- ✅ Whale activity scoring
- ✅ BTC dominance trends
- ✅ Altcoin season index
- ⚠️ GARCH volatility models (planned)
- ⚠️ Regime detection (planned)

**Infrastructure** ⚡ (60%)
- ✅ Multi-channel alerting (Telegram, Discord, Slack)
- ✅ Trade and whale alerts
- ✅ Error notifications
- ✅ **Crypto paper trading engine** 🆕
- ✅ **Historical data fetcher for crypto assets** 🆕
- ✅ **Comprehensive backtesting framework** 🆕
- ⚠️ WebSocket manager (planned)
- ⚠️ Rate limiting (planned)
- ⚠️ Redis caching (planned)

**Configuration** ✅ (100%)
- ✅ Comprehensive crypto settings YAML
- ✅ Multi-chain RPC configurations
- ✅ Token watchlists (BTC, ETH, SOL, etc.)
- ✅ Strategy parameters
- ✅ Risk management settings
- ✅ Alert configurations
- ✅ Updated .env template with all crypto APIs

**🆕 Paper Trading Infrastructure** ✅ (100%)
- ✅ **Crypto Paper Trading Engine** - Realistic simulation with gas costs, slippage
- ✅ **Historical Data Fetcher** - Multi-source data (Binance, CoinGecko, simulated)
- ✅ **Backtesting Framework** - Complete strategy testing and comparison
- ✅ **Performance Analytics** - Sharpe ratio, drawdown, win rate, returns
- ✅ **Multi-Asset Support** - Portfolio management across chains
- ✅ **Technical Indicators** - 15+ indicators auto-generated
- ✅ **Strategy Library** - SMA crossover, RSI mean reversion, extensible
- ✅ **Comprehensive Demo** - Full working example with documentation

### ✅ **FULLY IMPLEMENTED** (Production Ready)

**Phase 1: Core Trading System** (100%)
- ✅ Data ingestion via yfinance (OHLCV data)
- ✅ 15+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.)
- ✅ RandomForest ML model with train/test validation
- ✅ Trading signal generation (BUY/SELL/HOLD)
- ✅ Comprehensive backtesting engine (476 lines)
- ✅ Automated daily pipeline with scheduling
- ✅ Docker containerization + CI/CD
- ✅ Test suite (9 passing tests)

**Advanced Strategies Suite** (75%)
- ✅ Kelly Criterion position sizing
- ✅ Black-Scholes options pricing & Greeks
- ✅ Multi-timeframe analysis (1min, 5min, 1h, daily)
- ✅ Ensemble ML models (RandomForest + GradientBoosting + LSTM)
- ✅ Mean reversion detection
- ✅ 3,388 lines of sophisticated strategy code

**Phase 3: Intelligence Network** (60%) 🆕
- ✅ NewsAPI integration with real API support (`sentiment_analyzer.py`)
- ✅ Reddit sentiment via PRAW (with fallback simulation)
- ✅ FRED API for macro data (CPI, Fed rates, unemployment, GDP)
- ✅ Economic regime detection
- ✅ Yield curve analysis
- ⚠️ Twitter sentiment (simulated - requires paid tier)

**Phase 6: Command Center Dashboard** (85%) 🆕
- ✅ Full Streamlit dashboard (`dashboard.py` - 620+ lines)
- ✅ Real-time portfolio overview
- ✅ Interactive price charts with signals
- ✅ Backtest performance visualization
- ✅ Advanced strategy analysis
- ✅ System status monitoring
- ✅ Multi-page navigation

**DeFi/Blockchain (Legacy)** (50%)
- ✅ Binance Smart Chain integration
- ✅ PancakeSwap DEX interaction
- ✅ Token swaps & price queries
- ⚠️ Production features (MEV protection, automated strategies)

### ⚠️ **PARTIALLY IMPLEMENTED** (In Development)

**Phase 4: Deep Learning** (40%)
- ✅ LSTM neural networks (TensorFlow)
- ✅ Hybrid LSTM + RandomForest ensemble
- ❌ TimesNet, Autoformer, Informer (not implemented)
- ❌ PyTorch transformers (not implemented)

**Phase 7: Infrastructure** (25%)
- ✅ Docker & Docker Compose
- ✅ GitHub Actions CI/CD
- ❌ Kubernetes (not implemented)
- ❌ Prometheus/Grafana monitoring (not implemented)

### ❌ **NOT IMPLEMENTED** (Planned)

**Phase 5: Reinforcement Learning** (0%)
- ❌ PPO/DDPG agents
- ❌ Custom trading environment
- ❌ Smart execution optimization

**Phases 8-10: Frontier Research** (0%)
- ❌ Quantum ML (empty research folders)
- ❌ Federated learning (empty research folders)
- ❌ Neurosymbolic AI (empty research folders)

## 📈 Overall Completion: ~65%

| Component | Status | Files | Lines of Code |
|:----------|:-------|:------|:--------------|
| **Crypto/Web3 Platform** | ⚡ 40% | 20+ | 5,000+ |
| - Multi-Chain Infrastructure | ✅ 100% | 6 | 1,800+ |
| - Crypto Data Sources | ✅ 70% | 2 | 900+ |
| - DEX Aggregation | ⚡ 40% | 1 | 500+ |
| - Crypto Strategies | ⚡ 30% | 1 | 400+ |
| - On-Chain Analytics | ⚡ 25% | 1 | 400+ |
| - Crypto ML Features | ✅ 60% | 1 | 500+ |
| - Infrastructure (Alerts) | ⚡ 40% | 1 | 400+ |
| **Core Trading System** | ✅ 100% | 10+ | 2,000+ |
| **Advanced Strategies** | ✅ 75% | 6 | 3,388 |
| **Broker Integration** | ⚠️ 70% | 3 | 900+ |
| **Dashboard & UI** | ✅ 85% | 1 | 620+ |
| **Real Data APIs** | ⚠️ 60% | 2 | 800+ |
| **DeFi Integration (Legacy)** | ⚠️ 50% | 2 | 819 |
| **Deep Learning** | ⚠️ 40% | 3 | 1,200+ |
| **Infrastructure** | ⚠️ 25% | Various | N/A |
| **RL Agents** | ❌ 0% | 0 | 0 |
| **Research** | ❌ 0% | 0 | 0 |

**Total Implemented:** ~15,000+ lines of production code

## 🗺️ Evolution Framework

- **Phase 1:** ✅ Base Trading System (100% Complete)
- **Phase 2:** ⚠️ Broker Integration (70% Complete)
- **Phase 3:** ⚠️ Intelligence Network (60% Complete - Real APIs Added!)
- **Phase 4:** ⚠️ Deep Learning (40% Complete - LSTM Only)
- **Phase 5:** ❌ RL Agents (0% - Planned)
- **Phase 6:** ✅ Dashboard (85% Complete - Fully Functional!)
- **Phase 7:** ⚠️ Infrastructure (25% - Docker Only)
- **Phase 8-10:** ❌ Research (0% - Future Work)
- **Phase 11-12:** ❌ Business Scaling (0% - Future Work)
- **🆕 Phase Crypto:** ⚡ **Crypto/Web3 Transformation (40% Complete)**

## 🛠️ Technology Stack

### Crypto/Web3 Stack 🆕
- **Blockchains:** Ethereum, Polygon, Arbitrum, Optimism, Base, BSC, Avalanche, Solana
- **Web3:** web3.py, eth-account, solana-py
- **DEXs:** Uniswap, PancakeSwap, 1inch aggregation
- **Data:** Binance API, CoinGecko, Fear & Greed Index
- **On-Chain:** Wallet tracking, whale monitoring
- **Strategies:** Funding rate arbitrage, cross-DEX arbitrage
- **ML Features:** NVT, MVRV, SOPR, funding momentum
- **Alerts:** Telegram, Discord, Slack

**Core Framework:**
- **Languages:** Python 3.11+
- **Data:** pandas, numpy, yfinance, TA-Lib
- **ML/AI:** scikit-learn (RandomForest, ensembles), TensorFlow (LSTM)
- **Visualization:** matplotlib, seaborn, plotly, Streamlit

**Real Data Sources:**
- **Crypto:** Binance API (spot + futures), CoinGecko API 🆕
- **Market Data:** yfinance (stocks, free, real-time)
- **News:** NewsAPI (100 requests/day free tier)
- **Social:** Reddit via PRAW (60 requests/min free)
- **Macro:** FRED API (unlimited, free)
- **On-Chain:** Fear & Greed Index, wallet tracking 🆕

**Trading Infrastructure:**
- **Crypto Exchanges:** Binance, DEX aggregators 🆕
- **Blockchains:** Multi-chain support (7+ networks) 🆕
- **DEXs:** PancakeSwap, Uniswap (aggregation ready) 🆕
- **Broker:** Alpaca (paper trading + live)
- **Containerization:** Docker, Docker Compose
- **CI/CD:** GitHub Actions
- **Alerts:** Telegram, Discord, Slack 🆕

**Dashboard & UI:**
- **Framework:** Streamlit (multi-page app)
- **Charts:** Plotly (interactive candlesticks, indicators)
- **Real-time:** Auto-refresh data feeds

**Planned/Future:**
- PyTorch transformers (Phase 4)
- Reinforcement learning (stable-baselines3, Phase 5)
- Kubernetes orchestration (Phase 7)
- Quantum ML, federated learning (Phases 8-10)

Note on native dependencies:
- `TA-Lib` requires system-level C library. For Ubuntu/Debian: `apt-get install -y build-essential libta-lib0 libta-lib-dev`. Docker container avoids local setup issues.

## 🗂️ Project Structure

```
trading-ai/
├── src/
│   ├── blockchain/              # 🆕 Multi-chain infrastructure
│   │   ├── chain_manager.py     # Unified chain abstraction
│   │   ├── ethereum_interface.py
│   │   ├── polygon_interface.py
│   │   ├── avalanche_interface.py
│   │   ├── base_interface.py
│   │   ├── solana_interface.py
│   │   └── bsc_interface.py
│   │
│   ├── defi/                    # 🆕 DEX aggregation & DeFi
│   │   ├── dex_aggregator.py    # Multi-DEX price aggregation
│   │   └── pancakeswap_trader.py
│   │
│   ├── crypto_data/             # 🆕 Crypto data sources
│   │   ├── binance_client.py    # Spot + futures + funding rates
│   │   └── coingecko_client.py  # Market data + token metadata
│   │
│   ├── crypto_strategies/       # 🆕 Crypto-specific strategies
│   │   └── funding_rate_arbitrage.py
│   │
│   ├── onchain/                 # 🆕 On-chain analytics
│   │   └── wallet_tracker.py    # Whale & smart money tracking
│   │
│   ├── crypto_ml/               # 🆕 Crypto ML features
│   │   └── crypto_features.py   # NVT, MVRV, SOPR, etc.
│   │
│   ├── infrastructure/          # 🆕 System infrastructure
│   │   └── alerting.py          # Multi-channel alerts
│   │
│   ├── risk/                    # 🆕 Risk management (planned)
│   │
│   ├── data_ingestion/          # ✅ yfinance + macro data (FRED)
│   ├── feature_engineering/     # ✅ 15+ technical indicators
│   ├── modeling/                # ✅ RandomForest + LSTM
│   ├── strategy/                # ✅ Signal generation
│   ├── execution/               # ✅ Broker interface + portfolio tracker
│   ├── advanced_strategies/     # ✅ Options, sentiment, Kelly Criterion
│   ├── backtesting/             # ✅ Performance analysis
│   ├── monitoring/              # ✅ Streamlit dashboard
│   └── utils/                   # ✅ Logging, config
│
├── config/
│   ├── settings.yaml            # Traditional trading config
│   └── crypto_settings.yaml     # 🆕 Crypto/Web3 config
│
├── data/                        # ✅ Raw + processed market data
├── models/                      # ✅ Trained ML models (.joblib)
├── signals/                     # ✅ Generated trading signals
├── backtests/                   # ✅ Backtest reports
├── logs/                        # ✅ Application logs
├── tests/                       # ✅ pytest test suite
├── docs/                        # ✅ Phase guides + documentation
├── research/                    # ❌ Empty (future experiments)
│
├── demo_multi_chain.py          # 🆕 Multi-chain demo
├── demo_live_trading.py         # ✅ End-to-end demo
├── defi_trading_demo.py         # ✅ DeFi trading demo
├── run_dashboard.sh             # ✅ Dashboard launcher
├── .env.template                # 🆕 Updated with crypto APIs
├── requirements.txt             # ✅ Core dependencies
└── requirements-crypto.txt      # 🆕 Crypto-specific dependencies
```
│   ├── backtesting/             # ✅ Performance analysis
│   ├── monitoring/              # ✅ Streamlit dashboard NEW!
│   ├── defi/                    # ⚠️ BSC + PancakeSwap
│   └── utils/                   # ✅ Logging, config
├── data/                        # ✅ Raw + processed market data
├── models/                      # ✅ Trained ML models (.joblib)
├── signals/                     # ✅ Generated trading signals
├── backtests/                   # ✅ Backtest reports
├── logs/                        # ✅ Application logs
├── tests/                       # ✅ pytest test suite
├── docs/                        # ✅ Phase guides + documentation
├── research/                    # ❌ Empty (future experiments)
├── demo_live_trading.py         # ✅ End-to-end demo NEW!
├── run_dashboard.sh             # ✅ Dashboard launcher NEW!
└── .env.example                 # ✅ API key template NEW!
```

## 🔧 Configuration

### 1. Set Up API Keys 🆕 Updated

Copy `.env.template` to `.env` and add your keys:

```bash
cp .env.template .env
```

Edit `.env` with your API keys:

```bash
# ===== CRYPTO EXCHANGES =====
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key

# ===== BLOCKCHAIN RPCs =====
ETHEREUM_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY
POLYGON_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY
ARBITRUM_RPC_URL=https://arb-mainnet.g.alchemy.com/v2/YOUR_KEY
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com

# Private keys (NEVER commit real keys!)
ETH_PRIVATE_KEY=your_ethereum_wallet_private_key
SOLANA_PRIVATE_KEY=your_solana_wallet_private_key

# ===== CRYPTO DATA PROVIDERS =====
COINGECKO_API_KEY=your_coingecko_api_key
GLASSNODE_API_KEY=your_glassnode_api_key

# ===== ALERTS =====
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
DISCORD_WEBHOOK_URL=your_discord_webhook_url

# ===== TRADITIONAL TRADING =====
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
NEWSAPI_API_KEY=your_newsapi_key
FRED_API_KEY=your_fred_api_key

# ===== TRADING SETTINGS =====
PAPER_TRADING=true
TRADING_MODE=paper  # paper | live
MAX_LEVERAGE=3
LOG_LEVEL=INFO
```

### 2. Configuration Files

**`config/crypto_settings.yaml`** - Crypto/Web3 configuration:
- Multi-chain RPC endpoints
- Token watchlists (BTC, ETH, SOL, etc.)
- Strategy parameters (funding rate, arbitrage, etc.)
- Risk management settings
- Alert configurations

**`config/settings.yaml`** - Traditional trading configuration:
- Stock ticker symbols
- ML model hyperparameters
- Risk management limits

## 🏃‍♂️ Running the System

### 🪙 Crypto/Web3 Demos 🆕

```bash
# Multi-chain portfolio operations
python demo_multi_chain.py

# DeFi trading on BSC/PancakeSwap
python defi_trading_demo.py

# More crypto demos coming soon:
# - demo_dex_aggregator.py
# - demo_arbitrage.py
# - demo_whale_tracking.py
# - demo_funding_arb.py
```

### 🎯 Quick Start - Dashboard

```bash
# Launch interactive dashboard
./run_dashboard.sh

# Opens at http://localhost:8501
# View portfolio, signals, backtests, and more!
```

### 🚀 Live Trading Demo

```bash
# Run automated trading demo (paper trading)
python demo_live_trading.py

# Demonstrates:
# - Data fetching
# - Signal generation
# - Trade execution
# - Portfolio tracking
# - Risk management
```

### Full Pipeline

```bash
# Run complete daily pipeline
python src/execution/daily_retrain.py
```

### Individual Components

```bash
# Fetch market data
python src/data_ingestion/fetch_data.py

# Generate features
python src/feature_engineering/feature_generator.py

# Train ML model
python src/modeling/train_model.py

# Generate trading signals
python src/strategy/simple_strategy.py

# Run backtest
python test_backtest.py
```

### Makefile Commands

```bash
make install       # Install dependencies
make install-dev   # Install dev dependencies + pre-commit
make test          # Run test suite
make test-cov      # Run tests with coverage
make pipeline      # Run daily pipeline
make docker-build  # Build Docker images
make docker-up     # Start services
make format        # Format code (black + isort)
make lint          # Lint code (ruff)
make clean         # Remove generated files
```

### Testing

```bash
# Run test suite
pytest tests/ -v

# Test with coverage
pytest tests/ --cov=src --cov-report=html

# Test specific module
python tests/test_trading_ai.py
```

## 📊 Key Features

### 🌟 Crypto/Web3 Platform 🆕

**Multi-Chain Infrastructure:**
- Unified chain manager supporting 7+ blockchains
- Ethereum mainnet + L2s (Arbitrum, Optimism, Base)
- Polygon, BSC, Avalanche, Solana support
- RPC fallback and automatic switching
- Gas price optimization across chains
- Multi-chain balance queries
- Cross-chain transaction tracking

**Crypto Data Integration:**
- Binance API: Spot + futures, funding rates, liquidations, open interest
- CoinGecko API: 15,000+ tokens, market data, trending coins
- Fear & Greed Index for market sentiment
- 24h statistics and order book depth
- Long/short ratio and trader positions
- Historical OHLCV data (1s to 1d timeframes)

**DEX Aggregation:**
- Multi-DEX price comparison (Uniswap, PancakeSwap, 1inch)
- Cross-DEX arbitrage detection
- Optimal routing for best execution
- Price impact calculation
- Gas cost optimization
- Split order routing

**Advanced Crypto Strategies:**
- Funding rate arbitrage (perpetual futures)
- Cross-exchange arbitrage detection
- Position sizing with leverage management
- Signal generation with confidence scores
- Risk-adjusted returns calculation
- Expected profit estimation

**On-Chain Analytics:**
- Whale wallet tracking
- Large transaction alerts
- Wallet behavior analysis
- Smart money detection
- Exchange flow monitoring
- Transaction pattern analysis

**Crypto-Specific ML Features:**
- NVT Ratio (Network Value to Transactions)
- MVRV (Market Value to Realized Value)
- SOPR (Spent Output Profit Ratio)
- Funding rate momentum
- Exchange netflow indicators
- Whale activity scoring
- BTC dominance trends
- Altcoin season index

**Multi-Channel Alerting:**
- Telegram bot integration
- Discord webhook notifications
- Slack webhook support
- Trade execution alerts
- Whale activity alerts
- Arbitrage opportunity alerts
- Funding rate alerts
- Error and system alerts

**🆕 Paper Trading & Backtesting Infrastructure:**
- **Paper Trading Engine:** Realistic simulation with gas costs, slippage, and multi-chain support
- **Historical Data:** Fetch from Binance, CoinGecko, or use simulated data for testing
- **Backtesting Framework:** Complete strategy testing with bar-by-bar execution
- **Performance Metrics:** Sharpe ratio, drawdown, win rate, volatility, returns
- **Multi-Asset Portfolios:** Track positions across multiple chains simultaneously
- **Technical Indicators:** Auto-generate 15+ indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- **Strategy Comparison:** Side-by-side strategy evaluation and benchmarking
- **Visualization:** Portfolio value plots, drawdown charts, returns distribution
- **Comprehensive Reporting:** Detailed performance reports with trade history

### ✅ Core Trading System

- **Data Pipeline:** Real-time market data via yfinance + macro indicators (FRED)
- **Technical Analysis:** 15+ indicators (SMA, EMA, RSI, MACD, Bollinger, ATR, etc.)
- **Machine Learning:** RandomForest + LSTM ensembles with validation
- **Signal Generation:** Multi-strategy BUY/SELL/HOLD with confidence scores
- **Backtesting:** Comprehensive performance analysis with 476-line engine
- **Risk Management:** Position sizing, drawdown limits, exposure controls
- **Logging:** Daily rotation with severity levels

### 🎯 Advanced Strategies

- **Portfolio Optimization:** Kelly Criterion, mean reversion, MPT Sharpe maximization
- **Sentiment Analysis:** Real NewsAPI + Reddit PRAW integration with fallback simulation
- **Options Trading:** Black-Scholes pricing, Greeks, spreads, straddles, iron condors
- **Multi-Timeframe:** 1-min, 5-min, hourly, daily analysis with cross-validation
- **Ensemble Models:** RandomForest + GradientBoosting + LSTM voting/stacking

### 📈 Real-Time Dashboard

- **Portfolio View:** Live equity, PnL, positions, drawdown tracking
- **Signal Visualization:** Interactive candlestick charts with buy/sell markers
- **Backtest Reports:** Equity curves, performance metrics, trade history
- **Advanced Analytics:** Strategy breakdown, risk assessment, confidence scoring
- **System Status:** Data availability, API connections, environment checks

### 🔌 Broker Integration

- **Alpaca API:** Paper trading + live trading support
- **Order Execution:** Market/limit orders with retry logic
- **Portfolio Tracking:** Real-time PnL, unrealized/realized gains, exposure
- **Risk Controls:** Max drawdown breakers, position limits, correlation checks
- **Trade Logging:** Complete audit trail with timestamps

### 🌐 Data Intelligence

- **News:** NewsAPI integration (100 req/day free) - real financial headlines
- **Social:** Reddit PRAW (60 req/min) - wallstreetbets, stocks, investing sentiment
- **Macro:** FRED API (unlimited) - CPI, Fed rates, unemployment, GDP, yield curve
- **Economic Regime:** Expansion/contraction/transition detection with confidence scores
- **Fallback:** Graceful degradation to simulated data when APIs unavailable

### ⛓️ DeFi/Blockchain

- **BSC Integration:** Web3.py connection to Binance Smart Chain
- **DEX Trading:** PancakeSwap router for token swaps
- **Token Analysis:** Price queries, liquidity pool info
- **Gas Estimation:** Transaction cost calculation
- **Multi-timeframe Analysis:** 1min/5min/1h/1d cross-validation with weighted signals
- **Signal Aggregation:** Weighted voting across 5+ strategies for robust decision-making

### 🔄 Current Capabilities

- Fetches OHLCV data for configurable tickers (multi-symbol support)
- Engineers 15+ technical features + 45+ advanced features
- Trains ML models with automated feature selection
- Generates trading signals with multi-strategy validation
- Provides portfolio-level recommendations with risk assessment
- Calculates optimal position sizes using Kelly Criterion
- Identifies options trading opportunities
- Tracks sentiment across multiple data sources
- Logs all operations with proper error handling
- Runs complete pipeline end-to-end with graceful degradation

## 📈 Sample Output

### Daily Pipeline Log

```
2025-06-13 09:00:00 - daily_pipeline - INFO - Starting daily pipeline for tickers: ['AAPL', 'MSFT', 'SPY']
2025-06-13 09:00:15 - daily_pipeline - INFO - Generated 8 features for AAPL
2025-06-13 09:00:30 - daily_pipeline - INFO - Model accuracy: 0.6250
2025-06-13 09:00:45 - daily_pipeline - INFO - Generated 250 signals
2025-06-13 09:00:45 - daily_pipeline - INFO - Signal distribution: {'BUY': 145, 'SELL': 105}
```

### Generated Signals

```csv
Date,Signal,Confidence,Price,Signal_Strength
2025-06-13,BUY,0.85,150.25,STRONG
2025-06-12,SELL,0.72,149.80,MEDIUM
2025-06-11,BUY,0.91,148.95,STRONG
```

## 🛠️ Development

### Development Workflow

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/

# Run linting
ruff check src/ tests/

# Run type checking
mypy src/
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_trading_ai.py -v
```

### Code Quality Tools

- **Black:** Code formatting (line length: 100)
- **isort:** Import sorting
- **Ruff:** Fast linting (replaces flake8, pylint)
- **mypy:** Static type checking
- **pre-commit:** Automated checks before commits

### Adding New Features

1. Create feature in appropriate module
2. Add configuration to `settings.yaml`
3. Update feature list in model training
4. Add tests in `tests/`
5. Run pre-commit checks
6. Update documentation

### Adding New Data Sources

1. Implement in `data_ingestion/`
2. Follow existing error handling patterns
3. Add API key to `.env.template`
4. Update configuration
5. Add integration tests

## 🚀 Roadmap

### ✅ Completed Phases

**Phase 1: Base Trading System**
- ✅ Data ingestion pipeline
- ✅ Feature engineering (15+ indicators)
- ✅ ML model training & validation
- ✅ Signal generation
- ✅ Advanced strategies suite

### 🎯 Phase 2: Broker Integration (Next)

- [ ] Alpaca API integration
- [ ] Paper trading implementation
- [ ] Order management system (buy, sell, modify, cancel)
- [ ] Portfolio tracking (real-time PnL, exposure)
- [ ] Trade execution logs
- [ ] Risk controls & position limits

**Target:** Q1 2026 | See [Phase 2 Guide](docs/phase_guides/phase_2_trading_system.md)

### 🔮 Phase 3: Intelligence Network

- [ ] Macro data ingestion (Fed rates, CPI, unemployment)
- [ ] News scraping & API integration
- [ ] Reddit/Twitter sentiment analysis
- [ ] Multimodal feature integration
- [ ] Regime detection & anomaly response

**Target:** Q2 2026 | See [Phase 3 Guide](docs/phase_guides/phase_3_intelligence_network.md)

### 🤖 Phase 4: Advanced ML

- [ ] Transformer models (TimesNet, Autoformer)
- [ ] Ensemble methods (stacking, blending)
- [ ] Hyperparameter tuning (Optuna)
- [ ] Model versioning & registry

**Target:** Q3 2026 | See [Phase 4 Guide](docs/phase_guides/phase_4_ai_powerup.md)

### 🎯 Phase 5: RL Execution Agents

- [ ] Gym environment for trading execution
- [ ] PPO agent training
- [ ] Slippage optimization
- [ ] Adaptive execution tactics

**Target:** Q4 2026 | See [Phase 5 Guide](docs/phase_guides/phase_5_smart_execution.md)

### 📊 Phases 6-12

- **Phase 6:** Command Center Dashboard
- **Phase 7:** Infrastructure Mastery (Kubernetes, monitoring)
- **Phase 8:** Frontier Research (Quantum ML, Federated Learning)
- **Phase 9:** Enhanced Intelligence
- **Phase 10:** Supercharged AI
- **Phase 11:** User Experience & Trust
- **Phase 12:** Business Scaling

See [docs/phase_guides/](docs/phase_guides/) for detailed roadmaps.

## 🔍 Monitoring & Debugging

### Logs

```bash
# View latest logs
tail -f logs/$(date +%Y-%m-%d).log

# Search for errors
grep ERROR logs/*.log
```

### Signals Analysis

```bash
# View generated signals
cat signals/AAPL_signals.csv

# Count signal distribution
grep -c BUY signals/*.csv
```

### Model Performance

Check model metrics in logs after training:

- Accuracy scores
- Feature importance
- Training/test sample counts

## ⚠️ Known Issues & Limitations

1. **Market Hours:** Currently doesn't check market hours (runs regardless)
2. **Data Quality:** Limited validation of fetched data (basic null checks only)
3. **Model Persistence:** Models retrain completely each run (no incremental learning)
4. **Real-time:** No real-time data processing yet (daily batch only)
5. **Backtesting:** Basic backtesting implemented but not integrated into main pipeline
6. **Broker Integration:** Phase 2 not yet started (no live trading capability)
7. **API Keys:** Sentiment analysis uses simulated data (requires API keys for live data)

## 👨‍💻 Development Workflow

### Setting Up Dev Environment

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/

# Run linting
ruff check src/ tests/

# Run type checking
mypy src/
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_trading_ai.py -v
```

### Code Quality Tools

- **Black:** Code formatting (line length: 100)
- **isort:** Import sorting
- **Ruff:** Fast linting (replaces flake8, pylint)
- **mypy:** Static type checking
- **pre-commit:** Automated checks before commits

## 🚀 Advanced Strategies Usage

The system includes a comprehensive suite of advanced strategies. See [ADVANCED_STRATEGIES_SUMMARY.md](ADVANCED_STRATEGIES_SUMMARY.md) and [docs/advanced_strategies_guide.md](docs/advanced_strategies_guide.md) for full documentation.

### Quick Example

```python
from advanced_strategies import AdvancedTradingStrategies

# Initialize with symbols
strategies = AdvancedTradingStrategies(['AAPL', 'MSFT', 'GOOGL'])

# Get comprehensive signals for a symbol
signals = strategies.get_comprehensive_signals(
    'AAPL',
    market_data,
    current_price=150.0,
    market_outlook='bullish'
)

print(f"Signal: {signals['aggregated_signal']['signal']}")
print(f"Confidence: {signals['aggregated_signal']['confidence']}")
print(f"Position Size: {signals['final_recommendations']['position_sizing']}")
```

### Available Strategy Components

1. **Portfolio Optimizer** - Kelly Criterion, mean reversion, MPT optimization
2. **Sentiment Analyzer** - Multi-source sentiment (Twitter, Reddit, News)
3. **Options Strategies** - Black-Scholes, spreads, straddles, iron condors
4. **Enhanced ML Models** - Ensemble methods, Prophet, ARIMA-GARCH
5. **Multi-timeframe Analysis** - Cross-timeframe validation (1m, 5m, 1h, 1d)

### Strategy Weights (Configurable)

```python
strategies.strategy_weights = {
    'ml_models': 0.30,          # Data-driven predictions
    'multi_timeframe': 0.25,    # Cross-timeframe validation
    'sentiment': 0.20,          # Market psychology
    'portfolio_optimization': 0.15,  # Risk management
    'options': 0.10             # Derivatives insights
}
```

## 🤝 Contributing

We welcome contributions! Please see our detailed contribution guidelines:

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Complete contribution guide
- **[BRANCH_MANAGEMENT.md](BRANCH_MANAGEMENT.md)** - Branching strategy & workflow

### Quick Start for Contributors

1. Fork the repository
2. Create feature branch: `git checkout -b feature/phase<N>-description`
3. Follow code style guidelines (Black, Ruff, mypy)
4. Add tests for new functionality
5. Update documentation
6. Submit pull request to `develop` branch

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on coding standards, testing, and PR process.

## 📚 Documentation

> **📖 Documentation Index**: See [BRANCH_DOCS_INDEX.md](BRANCH_DOCS_INDEX.md) for a complete navigation guide to all documentation.

### Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - Quick setup guide
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Detailed setup walkthrough
- **[CODESPACES.md](CODESPACES.md)** - GitHub Codespaces guide

### Development & Contributing
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[BRANCH_MANAGEMENT.md](BRANCH_MANAGEMENT.md)** - Branching strategy & workflow
- **[BRANCH_QUICK_REF.md](BRANCH_QUICK_REF.md)** - Quick reference for branch commands
- **[BRANCH_VISUAL_GUIDE.md](BRANCH_VISUAL_GUIDE.md)** - Visual branch flow diagrams
- **[BRANCH_STRATEGY_RECOMMENDATIONS.md](BRANCH_STRATEGY_RECOMMENDATIONS.md)** - Implementation recommendations

### Project Reports
- **[AUDIT_REPORT.md](AUDIT_REPORT.md)** - Code quality audit results
- **[SECURITY_REPORT.md](SECURITY_REPORT.md)** - Security analysis & scorecard
- **[FIXES.md](FIXES.md)** - Recent improvements & fixes

### Advanced Features
- **[ADVANCED_STRATEGIES_SUMMARY.md](ADVANCED_STRATEGIES_SUMMARY.md)** - Strategies overview
- **[VIRTUALS_INTEGRATION.md](VIRTUALS_INTEGRATION.md)** - Virtuals Protocol integration roadmap

**Phase Guides:** See `/docs/phase_guides/` for detailed Phase Execution Guides.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Status:** Phase 1 Complete ✅ | Next: Phase 2 Broker Integration 🎯
