# Trading-AI

Welcome to the Trading-AI Project — an autonomous, adaptive, next-generation AI-driven trading empire.

## 📜 Project Vision

Build a scalable, modular, fully autonomous AI trading system capable of evolving over time — from daily retraining models to cutting-edge innovations like Quantum ML, Federated Learning, and Neurosymbolic AI.

## 🚀 Quick Start

### Option 1: GitHub Codespaces (Easiest)

**Running in Codespaces right now?**

```bash
# One-command setup
bash START_HERE.sh

# Or manual steps
make install
make test
make pipeline
```

📖 **See [CODESPACES.md](CODESPACES.md) for Codespaces-specific guide**

### Option 2: Docker (Recommended for Local)

```bash
# Clone and setup
git clone <your-repo>
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

### Makefile (Developer Convenience)

A `Makefile` with common development targets is provided to simplify local workflows. Recommended commands:

```bash
# Install Python dependencies
make install

# Run the test suite
make test

# Run the daily pipeline locally
make pipeline

# Build docker images (no cache)
make docker-build

# Start services via docker-compose
make docker-up
```

## 📊 Current Status: Phase 1 Complete + Advanced Strategies ✅

### ✅ What's Working Now

**Phase 1: Base Trading System** (Complete)
- ✅ Daily data ingestion via yfinance (OHLCV)
- ✅ 15+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- ✅ RandomForest ML model training with validation
- ✅ Trading signal generation (BUY/SELL/HOLD)
- ✅ Comprehensive logging & error handling
- ✅ Docker containerization
- ✅ Test suite (9 passing tests)
- ✅ CI/CD pipeline (GitHub Actions)

**Advanced Strategies Suite** (Implemented)
- ✅ Portfolio optimization with Kelly Criterion
- ✅ Multi-source sentiment analysis (Twitter, Reddit, News)
- ✅ Options strategies (Black-Scholes, spreads, straddles, iron condors)
- ✅ Enhanced ML models (ensemble methods, Prophet, ARIMA-GARCH)
- ✅ Multi-timeframe analysis (1min, 5min, 1h, 1d)
- ✅ Signal aggregation with weighted voting

### 🎯 Next Phase: Phase 2 - Broker Integration

**What's Coming Next:**
- [ ] Alpaca API integration
- [ ] Paper trading mode
- [ ] Order management system (buy, sell, modify, cancel)
- [ ] Real-time portfolio tracking
- [ ] Live trading with risk controls
- [ ] Trade execution logs

See [Phase 2 Guide](docs/phase_guides/phase_2_trading_system.md) for details.

## 🗺️ Evolution Framework

- **Phase 0:** Command Center Setup ✅
- **Phase 1:** Base Trading System ✅ (Complete)
- **Phase 2:** Broker Connectivity & Paper Trading 🎯 (Next)
- **Phase 3:** Intelligence Network Expansion (Macro data, News, Sentiment)
- **Phase 4:** Advanced ML (Transformers, Ensembles)
- **Phase 5:** RL Execution Agents
- **Phase 6:** Command Center Dashboard
- **Phase 7–12:** Advanced Research (Quantum ML, Federated Learning, Neurosymbolic AI)

## 🛠️ Technology Stack

- **Core:** Python 3.11+, pandas, numpy, scikit-learn
- **Data:** yfinance, TA-Lib for technical indicators
- **ML:** RandomForest (Phase 1), PyTorch (Future)
- **Infrastructure:** Docker, Docker Compose
- **APIs:** Alpaca/IBKR (Future), Alpha Vantage (Future)
- **Advanced:** TensorFlow Federated, Qiskit, PennyLane (Future)

Note on native dependencies:
- `TA-Lib` requires system-level C library dependencies on many Linux distributions. If you encounter installation failures when running `pip install -r requirements.txt`, install the platform packages first (for Debian/Ubuntu: `apt-get install -y build-essential libtool libffi-dev libssl-dev libatlas-base-dev && apt-get install -y libta-lib0 libta-lib-dev` or build `ta-lib` from source). Using the provided Docker container can avoid local environment issues.

## 🗂️ Project Structure

```
trading-ai/
├── src/                          # Source code
│   ├── data_ingestion/          # Data fetching (yfinance, APIs)
│   ├── feature_engineering/     # Technical indicators (15+ features)
│   ├── modeling/                # ML training (RandomForest, ensembles)
│   ├── strategy/                # Signal generation & analysis
│   ├── execution/               # Pipeline orchestration & broker interface
│   ├── advanced_strategies/     # ✨ New: Portfolio opt, sentiment, options, multi-timeframe
│   ├── backtesting/            # Strategy backtesting
│   ├── utils/                  # Logging, config, helpers
│   └── monitoring/             # Performance tracking (future)
├── data/                        # Data storage
│   ├── raw/                    # Raw OHLCV data from yfinance
│   └── processed/              # Feature-engineered datasets
├── models/                      # Trained ML models (.joblib files)
├── signals/                     # Generated trading signals (.csv)
├── logs/                        # Application logs (daily rotation)
├── tests/                       # Test suite (pytest)
├── config/                      # Configuration (settings.yaml, .env)
├── docs/                        # Documentation
│   ├── phase_guides/           # Detailed phase implementation guides
│   ├── advanced_strategies_guide.md  # Advanced strategies docs
│   └── evolution_plan.md       # Long-term vision
└── research/                    # Experimental features
    ├── quantum_ml/             # Quantum machine learning (Phase 8+)
    ├── federated_learning/     # Privacy-preserving ML (Phase 8+)
    └── neurosymbolic_ai/       # Hybrid reasoning (Phase 8+)
```

## 🔧 Configuration

### Environment Variables

Copy `.env.template` to `.env` and configure:

```bash
# API Keys
ALPHA_VANTAGE_API_KEY=your_key_here
ALPACA_API_KEY=your_key_here

# Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
PAPER_TRADING=true
```

### Settings File

Edit `config/settings.yaml` for:

- Ticker symbols to trade
- Model parameters
- Feature engineering settings
- Risk management rules

## 🏃‍♂️ Running the System

### Full Pipeline

```bash
# Run complete daily pipeline
python src/execution/daily_retrain.py
```

### Individual Components

```bash
# Fetch data only
python src/data_ingestion/fetch_data.py

# Generate features
python src/feature_engineering/feature_generator.py

# Train model
python src/modeling/train_model.py

# Generate signals
python src/strategy/simple_strategy.py
```

### Testing

```bash
# Run test suite
python -m pytest tests/ -v

# Test specific component
python tests/test_trading_ai.py
```

## 📊 Current Features

### ✅ Core System (Phase 1)

- **Data Ingestion:** yfinance integration with error handling & retry logic
- **Feature Engineering:** 15+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.)
- **ML Pipeline:** RandomForest classifier with train/test split & cross-validation
- **Signal Generation:** BUY/SELL/HOLD signals with confidence levels & strength ratings
- **Logging:** Comprehensive daily log rotation with multiple severity levels
- **Configuration:** YAML-based settings + environment variable management
- **Containerization:** Docker + docker-compose with volume mounts
- **Testing:** 9 passing tests with pytest framework
- **CI/CD:** GitHub Actions workflow for automated testing & Docker builds

### 🚀 Advanced Strategies (New)

- **Portfolio Optimization:** Kelly Criterion position sizing, mean reversion detection, MPT optimization
- **Sentiment Analysis:** Multi-source aggregation (Twitter, Reddit, News) with consensus scoring
- **Options Strategies:** Black-Scholes pricing, Greeks, bull/bear spreads, straddles, iron condors
- **Enhanced ML Models:** Ensemble methods (RF, GBM, SVR), Prophet forecasting, ARIMA-GARCH
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

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Follow code style in existing modules
4. Add tests for new functionality
5. Update documentation
6. Submit pull request

## 📚 Documentation

See `/docs/phase_guides/` for detailed Phase Execution Guides.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Status:** Phase 1 Complete ✅ | Next: Phase 2 Broker Integration 🎯
