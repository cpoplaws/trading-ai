# Trading-AI

Welcome to the Trading-AI Project — an autonomous, adaptive, next-generation AI-driven trading empire.

## 📜 Project Vision

Build a scalable, modular, fully autonomous AI trading system capable of evolving over time — from daily retraining models to cutting-edge innovations like Quantum ML, Federated Learning, and Neurosymbolic AI.

## � Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone and setup
git clone <your-repo>
cd trading-ai

# Build and run
docker-compose up --build
```

### Option 2: Local Development

```bash
# Setup environment
./setup.sh

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python src/execution/daily_retrain.py
```

## �🛤️ Evolution Framework

- **Phase 0:** Command Center Setup ✅
- **Phase 1:** Base Trading System ✅ (Current)
- **Phase 2:** Broker Connectivity & Paper Trading
- **Phase 3:** Intelligence Network Expansion
- **Phase 4:** Transformer and Ensemble Model Deployment
- **Phase 5:** Reinforcement Learning Execution Agents
- **Phase 6:** Full Command Center Deployment
- **Phase 7–12:** Advanced Enhancements

## 🛠️ Technology Stack

- **Core:** Python 3.11+, pandas, numpy, scikit-learn
- **Data:** yfinance, TA-Lib for technical indicators
- **ML:** RandomForest (Phase 1), PyTorch (Future)
- **Infrastructure:** Docker, Docker Compose
- **APIs:** Alpaca/IBKR (Future), Alpha Vantage (Future)
- **Advanced:** TensorFlow Federated, Qiskit, PennyLane (Future)

## 🗂️ Project Structure

```
trading-ai/
├── src/                    # Source code
│   ├── data_ingestion/     # Data fetching and loading
│   ├── feature_engineering/# Technical indicator generation
│   ├── modeling/          # ML model training and evaluation
│   ├── strategy/          # Signal generation and analysis
│   ├── execution/         # Pipeline orchestration
│   ├── utils/             # Logging and utilities
│   └── backtesting/       # Strategy backtesting (TODO)
├── data/                  # Data storage
│   ├── raw/              # Raw OHLCV data
│   └── processed/        # Feature-engineered data
├── models/               # Trained ML models
├── signals/              # Generated trading signals
├── logs/                 # Application logs
├── tests/                # Test suite
├── config/               # Configuration files
└── docs/                 # Documentation
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

## 📊 Current Features (Phase 1)

### ✅ Implemented

- **Data Ingestion:** yfinance integration with error handling
- **Feature Engineering:** 10+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- **ML Pipeline:** RandomForest classifier with proper validation
- **Signal Generation:** Buy/Sell signals with confidence levels
- **Logging:** Comprehensive logging system
- **Configuration:** YAML-based configuration management
- **Containerization:** Docker setup for consistent deployment
- **Testing:** Basic test suite structure

### 🔄 Current Capabilities

- Fetches OHLCV data for configurable tickers
- Engineers 10+ technical features
- Trains ML models with train/test split
- Generates trading signals with confidence levels
- Logs all operations with proper error handling
- Runs complete pipeline end-to-end

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

### Adding New Features

1. Create feature in appropriate module
2. Add configuration to `settings.yaml`
3. Update feature list in model training
4. Add tests in `tests/`
5. Update documentation

### Adding New Data Sources

1. Implement in `data_ingestion/`
2. Follow existing error handling patterns
3. Add API key to `.env.template`
4. Update configuration

## 🚀 Roadmap

### Phase 2: Broker Integration

- [ ] Alpaca API integration
- [ ] Paper trading implementation
- [ ] Order management system
- [ ] Portfolio tracking

### Phase 3: Advanced Features

- [ ] Real-time data streaming
- [ ] Multiple timeframe analysis
- [ ] Portfolio optimization
- [ ] Risk management system

### Phase 4: Advanced ML

- [ ] LSTM/Transformer models
- [ ] Ensemble methods
- [ ] Feature selection optimization
- [ ] Model performance monitoring

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

1. **Market Hours:** Currently doesn't check market hours
2. **Data Quality:** Limited validation of fetched data
3. **Model Persistence:** Models retrain completely each run
4. **Real-time:** No real-time data processing yet
5. **Backtesting:** Basic backtesting not yet implemented

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
