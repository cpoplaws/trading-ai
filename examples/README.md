# Examples

This directory contains runnable examples demonstrating different features of the Trading AI system.

## Directory Structure

```
examples/
├── strategies/          # Trading strategy examples
├── defi/               # DeFi strategy examples
├── integration/        # System integration examples
└── system/            # System-level examples
```

## Quick Start

### Run Strategy Examples
```bash
# Paper trading demo
python examples/strategies/demo_crypto_paper_trading.py

# Live trading demo (requires API keys)
python examples/strategies/demo_live_trading.py

# Simple backtest
python examples/strategies/simple_backtest_demo.py
```

### Run DeFi Examples
```bash
# Simple DeFi demo
python examples/defi/defi_simple_demo.py

# Full DeFi trading demo
python examples/defi/defi_trading_demo.py

# Multi-chain arbitrage
python examples/defi/demo_multi_chain.py
```

### Run Integration Examples
```bash
# Phase 2 & 3 integration demo
python examples/integration/phase2_phase3_demo.py
```

## Or Use the Unified Entry Point

Instead of running examples individually, use the unified `start.py`:

```bash
# See all available options
python start.py --help

# Run specific strategy
python start.py --strategy momentum

# Start agent swarm
python start.py --agents

# Open unified dashboard
python start.py
```

---

**Note**: All examples still work exactly as before - they're just better organized now!
