# 🚀 Trading AI - Codespaces Quick Start

You're running Trading AI in **GitHub Codespaces**! Here's how to get started.

## Step 1: Install Dependencies

```bash
# Install all Python dependencies
pip install -r requirements.txt

# Or use the Makefile
make install
```

## Step 2: Test Your Setup

```bash
# Quick environment test
chmod +x test-env.sh
./test-env.sh

# Or run the test suite
make test
```

## Step 3: Run Your First Pipeline

```bash
# Run the complete trading pipeline
make pipeline

# This will:
# - Fetch market data for AAPL, MSFT, SPY
# - Generate 15+ technical indicators
# - Train ML models
# - Generate trading signals
```

## Step 4: View Results

```bash
# Check generated signals
cat signals/AAPL_signals.csv

# View model files
ls -lh models/

# Check logs
tail -f logs/$(date +%Y-%m-%d).log
```

## Common Codespaces Commands

```bash
# Development workflow
make install-dev      # Install dev dependencies
make test             # Run tests
make test-cov         # Run tests with coverage
make format           # Format code
make lint             # Lint code

# Docker (if needed)
make docker-build     # Build Docker images
make docker-up        # Start services
```

## File Structure

```
/workspaces/trading-ai/
├── src/              # Source code
│   ├── data_ingestion/
│   ├── feature_engineering/
│   ├── modeling/
│   ├── strategy/
│   ├── execution/
│   └── advanced_strategies/
├── data/             # Data storage (gitignored)
│   ├── raw/         # Raw market data
│   └── processed/   # Processed features
├── models/          # Trained models (gitignored)
├── signals/         # Trading signals (gitignored)
├── logs/            # Logs (gitignored)
└── tests/           # Test suite
```

## Development Tips

### 1. Install Dev Tools

```bash
make install-dev      # Installs ruff, black, pytest, etc.
```

### 2. Use Pre-commit Hooks

```bash
pre-commit install    # Auto-format before commits
```

### 3. Run Tests Frequently

```bash
make test             # Quick test
make test-cov         # With coverage
```

### 4. View Logs in Real-time

```bash
tail -f logs/$(date +%Y-%m-%d).log
```

## Codespaces Features

### Port Forwarding
- Port 8888: Jupyter (if you start it)
- Port 8000: Future web interface

### Terminal
- Use the integrated terminal (Ctrl+`)
- Multiple terminals supported

### Extensions Pre-installed
- Python
- Pylance (Python language server)
- Jupyter
- Docker
- Ruff (linter)
- GitHub Copilot

## Troubleshooting

### TA-Lib Issues

TA-Lib is in requirements but may need system libraries:

```bash
# Install TA-Lib dependencies
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

Or just remove it from requirements.txt if not needed yet.

### File System Issues

If you get "No file system provider" errors:
1. Make sure you're in the correct directory: `/workspaces/trading-ai`
2. Try: `cd /workspaces/trading-ai`
3. Reload the window (Cmd/Ctrl+Shift+P → "Developer: Reload Window")

### Python Not Found

```bash
which python3
which pip3
python3 --version
```

If python3 exists but python doesn't:
```bash
alias python=python3
alias pip=pip3
```

## What to Do Next

### Immediate Actions:
1. ✅ Run `make install` 
2. ✅ Run `make test`
3. ✅ Run `make pipeline`
4. ✅ View results in `signals/` directory

### Configuration (Optional):
1. Edit `config/settings.yaml` - Configure tickers, model params
2. Create `.env` - Add API keys (not needed for Phase 1)

### Development:
1. Follow Phase 2 guide: `docs/phase_guides/phase_2_trading_system.md`
2. Implement broker integration (Alpaca/IBKR)
3. Add paper trading

## Quick Commands Cheatsheet

```bash
# Install & Setup
make install          # Install dependencies
make install-dev      # Install dev dependencies

# Testing
make test             # Run tests
make test-cov         # With coverage

# Running
make pipeline         # Run trading pipeline

# Code Quality
make format           # Format code (black + isort)
make lint             # Lint (ruff)
make clean            # Clean generated files

# Help
make help             # Show all commands
```

## Need Help?

- 📖 Main README: [README.md](../README.md)
- 📖 Quick Start: [QUICKSTART.md](../QUICKSTART.md)
- 📖 Phase Guides: [docs/phase_guides/](../docs/phase_guides/)
- 📖 Advanced Strategies: [ADVANCED_STRATEGIES_SUMMARY.md](../ADVANCED_STRATEGIES_SUMMARY.md)

---

**You're all set! Run `make install && make test && make pipeline` to get started! 🎯**
