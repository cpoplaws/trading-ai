# DeFi Strategy Examples

Examples demonstrating DeFi (Decentralized Finance) trading strategies.

## Files

### `defi_simple_demo.py`
Simple introduction to DeFi strategies.
- Yield Optimizer basics
- Impermanent loss hedging intro
- Easy to understand

**Run**: `python examples/defi/defi_simple_demo.py`

---

### `defi_trading_demo.py`
Complete DeFi trading demo.
- All 3 DeFi strategies
- Yield Optimizer
- Impermanent Loss Hedging
- Multi-Chain Arbitrage basics

**Run**: `python examples/defi/defi_trading_demo.py`

---

### `demo_multi_chain.py`
Multi-chain arbitrage demonstration.
- Cross-chain price differences
- Bridge protocols (Hop, Across, Stargate, Synapse)
- Arbitrage opportunities across 6+ blockchains

**Run**: `python examples/defi/demo_multi_chain.py`

---

## DeFi Strategies Included

### 1. Yield Optimizer
Automatically finds and rotates capital to highest-yield DeFi protocols.
- Compound, Aave, Curve, Yearn
- Auto-compounding
- Gas optimization

### 2. Impermanent Loss Hedging
Hedges liquidity provision risk in AMMs.
- Detects IL exposure
- Hedges with perpetuals
- Protects LP positions

### 3. Multi-Chain Arbitrage
Exploits price differences across blockchains.
- Ethereum, Polygon, Arbitrum, Optimism, BSC, Avalanche
- Bridge routing optimization
- Flash loan support

---

## Quick Start

```bash
# Start simple
python examples/defi/defi_simple_demo.py

# Try full DeFi suite
python examples/defi/defi_trading_demo.py

# Advanced: Multi-chain arbitrage
python examples/defi/demo_multi_chain.py
```

## Requirements

DeFi strategies require:
- Web3 provider (Infura, Alchemy, or local node)
- Gas tokens for transactions
- Sufficient capital for meaningful arbitrage

**Note**: DeFi demos use simulated prices for safety. Real deployment requires mainnet connection.

---

**Documentation**: See `docs/MULTICHAIN_ARBITRAGE.md` for detailed strategy info.
