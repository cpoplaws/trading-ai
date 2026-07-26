# Quick Setup Guide - Trading AI

## Prerequisites

- Docker & Docker Compose (recommended)
- OR Python 3.11+ with pip

## Option 1: Docker Setup (Recommended) 🐳

```bash
# Clone the repository
git clone https://github.com/cpoplaws/trading-ai.git
cd trading-ai

# Build and start services
docker-compose up --build

# The system will:
# - Fetch market data for configured tickers
# - Generate technical features
# - Train ML models
# - Generate trading signals
```

## Option 2: Local Development Setup 💻

### Step 1: Environment Setup

```bash
# Run the setup script
chmod +x setup.sh
./setup.sh

# Or manually:
mkdir -p data/{raw,processed} models signals logs

# Copy environment template
cp .env.template .env
```

### Step 2: Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies (optional)
pip install -r requirements-dev.txt

# Install pre-commit hooks (optional)
pre-commit install
```

### Step 3: Configuration

Edit `.env` file with your API keys (optional for Phase 1):

```bash
ALPHA_VANTAGE_API_KEY=your_key_here
ALPACA_API_KEY=your_key_here
ENVIRONMENT=development
LOG_LEVEL=INFO
```

Edit `config/settings.yaml` to configure:
- Ticker symbols
- Model parameters
- Feature engineering settings

### Step 4: Run the Pipeline

```bash
# Using Makefile (recommended)
make pipeline

# Or directly
python src/execution/daily_retrain.py
```

## Verify Installation ✅

```bash
# Run tests
make test

# Check logs
tail -f logs/$(date +%Y-%m-%d).log

# View generated signals
ls -la signals/

# View trained models
ls -la models/
```

## Common Issues & Solutions

### TA-Lib Installation Issues

If you encounter TA-Lib installation errors:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
pip install TA-Lib
```

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Windows:**
Use the Docker setup or download pre-built wheels from:
https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

### Python Version Issues

Ensure you're using Python 3.11+:
```bash
python --version  # Should be 3.11.x or higher
```

### Docker Issues

```bash
# Rebuild without cache
docker-compose build --no-cache

# Check logs
docker-compose logs -f trading-ai

# Restart services
docker-compose restart
```

## Next Steps

1. **Verify data:** Check `data/raw/` for downloaded market data
2. **Review models:** Inspect `models/` for trained model files
3. **Analyze signals:** Open `signals/*.csv` to view trading signals
4. **Check logs:** Review `logs/` for pipeline execution details
5. **Customize:** Edit `config/settings.yaml` for your use case
6. **Run tests:** Execute `make test` to verify functionality
7. **Phase 2:** Follow [Phase 2 Guide](docs/phase_guides/phase_2_trading_system.md) to add broker integration

## Quick Commands Reference

```bash
# Development
make install          # Install dependencies
make test            # Run test suite
make pipeline        # Run daily pipeline

# Docker
make docker-build    # Build images
make docker-up       # Start services

# Code quality
black src/ tests/    # Format code
ruff check src/      # Lint code
pytest tests/ --cov  # Test with coverage
```

## Getting Help

- **Documentation:** See [docs/](docs/) directory
- **Phase Guides:** See [docs/phase_guides/](docs/phase_guides/)
- **Advanced Strategies:** See [ADVANCED_STRATEGIES_SUMMARY.md](ADVANCED_STRATEGIES_SUMMARY.md)
- **Issues:** Open an issue on GitHub

## What to Expect

### First Run Output

```
2025-12-05 10:00:00 - daily_pipeline - INFO - Starting daily pipeline for tickers: ['AAPL', 'MSFT', 'SPY']
2025-12-05 10:00:15 - daily_pipeline - INFO - Fetching market data...
2025-12-05 10:00:30 - daily_pipeline - INFO - Generated 15 features for AAPL
2025-12-05 10:00:45 - daily_pipeline - INFO - Model accuracy: 0.6250
2025-12-05 10:01:00 - daily_pipeline - INFO - Generated 250 signals
2025-12-05 10:01:00 - daily_pipeline - INFO - Signal distribution: {'BUY': 145, 'SELL': 105}
2025-12-05 10:01:00 - daily_pipeline - INFO - Pipeline completed successfully!
```

### Generated Files

- `data/raw/AAPL.csv` - Raw OHLCV data
- `data/processed/AAPL.csv` - Feature-engineered data
- `models/model_AAPL.joblib` - Trained ML model
- `models/features_AAPL.joblib` - Feature list
- `signals/AAPL_signals.csv` - Trading signals

---

**Ready to trade? Start with Phase 1, then progress to Phase 2 for broker integration! 🚀**
