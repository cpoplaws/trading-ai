# Base Blockchain Setup Guide

**Status**: ✅ Base is fully supported and ready to use!
**Chain ID**: 8453
**Native Token**: ETH
**Explorer**: https://basescan.org

---

## 🌐 What is Base?

Base is Coinbase's Layer 2 blockchain built on Ethereum using Optimism's OP Stack. It offers:
- ⚡ **Fast transactions** (~2 seconds)
- 💰 **Low gas fees** (~$0.01 per transaction)
- 🔒 **Ethereum security** (inherits from L1)
- 🏦 **Easy Coinbase integration**
- 🎯 **Growing DeFi ecosystem**

---

## ✅ Base is Already Configured

Your trading AI system has Base fully integrated:

```python
Chain.BASE = "base"

ChainConfig(
    chain_id=8453,
    name="Base",
    rpc_urls=[
        "https://mainnet.base.org",        # Primary RPC
        "https://base.llamarpc.com",       # Fallback #1
        "https://base.meowrpc.com"         # Fallback #2
    ],
    native_token="ETH",
    explorer_url="https://basescan.org"
)
```

---

## 🔑 Step 1: Configure Your Wallet

### Option A: Use Existing Ethereum Wallet

Base uses the same addresses as Ethereum. If you have an Ethereum wallet, you can use it on Base!

Edit your `.env` file:

```bash
# Your Ethereum private key works on Base too!
ETH_PRIVATE_KEY=your_ethereum_wallet_private_key_here

# Base RPC (already configured)
BASE_RPC_URL=https://mainnet.base.org
```

### Option B: Create New Wallet (Recommended for Security)

```python
from eth_account import Account

# Generate new wallet
account = Account.create()
print(f"Address: {account.address}")
print(f"Private Key: {account.key.hex()}")

# Add to .env file
```

⚠️ **Security Warning**: Never commit private keys to git! Keep them in `.env` only.

---

## 🚀 Step 2: Connect to Base

### Quick Test Connection

```python
from src.blockchain.chain_manager import ChainManager, Chain

# Initialize manager
manager = ChainManager()

# Connect to Base
success = manager.connect(Chain.BASE)

if success:
    print("✅ Connected to Base!")

    # Get Base configuration
    config = manager.get_chain_config(Chain.BASE)
    print(f"Chain: {config.name}")
    print(f"Chain ID: {config.chain_id}")
    print(f"Native Token: {config.native_token}")
else:
    print("❌ Failed to connect to Base")
```

### Check Your Balance

```python
# Your wallet address
my_address = "0xYourAddressHere"

# Get ETH balance on Base
base_connection = manager.get_connection(Chain.BASE)
balance = base_connection.get_balance(my_address)

print(f"Base Balance: {balance} ETH")
```

---

## 💰 Step 3: Fund Your Base Wallet

You need ETH on Base to trade. Here's how to get it:

### Method 1: Bridge from Ethereum (Official Bridge)

1. Visit: https://bridge.base.org
2. Connect your wallet (MetaMask, Coinbase Wallet, etc.)
3. Bridge ETH from Ethereum L1 → Base L2
4. **Time**: ~10-15 minutes
5. **Cost**: Ethereum gas fee (~$5-20 depending on network)

### Method 2: Coinbase (Fastest & Cheapest)

1. Buy ETH on Coinbase
2. Select "Send to Base" when withdrawing
3. Paste your wallet address
4. **Time**: ~2 minutes
5. **Cost**: Nearly free

### Method 3: Third-Party Bridges

- **Hop Protocol**: https://hop.exchange
- **Across Protocol**: https://across.to
- **Stargate**: https://stargate.finance

---

## 🎯 Step 4: Start Trading on Base

### Example 1: Check Token Balances

```python
from src.blockchain.chain_manager import ChainManager, Chain

manager = ChainManager()
manager.connect(Chain.BASE)

# Your wallet
my_address = "0xYourAddressHere"

# Check ETH balance
eth_balance = manager.get_balance(Chain.BASE, my_address)
print(f"ETH: {eth_balance}")

# Check USDC balance (Base USDC contract: 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913)
usdc_address = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
usdc_balance = manager.get_balance(Chain.BASE, my_address, usdc_address)
print(f"USDC: {usdc_balance}")
```

### Example 2: Monitor Base DEXs

```python
from src.defi.dex_aggregator import DEXAggregator

# Initialize DEX aggregator
dex = DEXAggregator(chain=Chain.BASE)

# Get best price for swapping 1 ETH to USDC
best_quote = dex.get_best_quote(
    token_in="ETH",
    token_out="USDC",
    amount_in=1.0
)

print(f"Best DEX: {best_quote.dex}")
print(f"Expected output: {best_quote.amount_out} USDC")
print(f"Price impact: {best_quote.price_impact}%")
```

### Example 3: Execute Swap on Base

```python
from src.defi.uniswap_v3 import UniswapV3Trader

# Initialize Uniswap V3 on Base
trader = UniswapV3Trader(chain=Chain.BASE)

# Swap 0.1 ETH for USDC
result = trader.swap(
    token_in="ETH",
    token_out="USDC",
    amount_in=0.1,
    slippage_tolerance=0.5  # 0.5% slippage
)

if result.success:
    print(f"✅ Swap successful!")
    print(f"Tx hash: {result.tx_hash}")
    print(f"Received: {result.amount_out} USDC")
else:
    print(f"❌ Swap failed: {result.error}")
```

---

## 🏦 Popular Protocols on Base

### DEXs (Decentralized Exchanges)
- **Uniswap V3**: https://app.uniswap.org
- **Aerodrome**: Native Base DEX
- **BaseSwap**: Community DEX
- **SushiSwap**: Multi-chain DEX

### Stablecoins
- **USDC**: `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913`
- **USDbC**: Bridged USDC
- **DAI**: Available via bridges

### Popular Tokens
- **WETH**: `0x4200000000000000000000000000000000000006`
- **cbETH**: Coinbase Wrapped Staked ETH

---

## 📊 Run Multi-Chain Demo (Includes Base)

Your system has a built-in demo:

```bash
cd /Users/silasmarkowicz/trading-ai-working

# Run multi-chain demo (includes Base)
python3 examples/defi/demo_multi_chain.py
```

This will:
1. Connect to Base and other chains
2. Show your balances across all chains
3. Display gas prices on each chain
4. Query token information

---

## 🎯 Quick Start: Trade on Base Now

### 1. **Add Private Key** (`.env` file):
```bash
ETH_PRIVATE_KEY=your_private_key_here
BASE_RPC_URL=https://mainnet.base.org
```

### 2. **Test Connection**:
```bash
python3 -c "
from src.blockchain.chain_manager import ChainManager, Chain
m = ChainManager()
success = m.connect(Chain.BASE)
print('✅ Connected to Base!' if success else '❌ Connection failed')
"
```

### 3. **Check Balance**:
```bash
python3 -c "
from src.blockchain.chain_manager import ChainManager, Chain
m = ChainManager()
m.connect(Chain.BASE)
balance = m.get_balance(Chain.BASE, 'YOUR_ADDRESS')
print(f'Base Balance: {balance} ETH')
"
```

### 4. **Run Multi-Chain Demo**:
```bash
python3 examples/defi/demo_multi_chain.py
```

---

## 🔧 Advanced: Cross-Chain Arbitrage

Trade between Base and other chains automatically:

```python
from src.defi.multichain_arbitrage import MultiChainArbitrage

# Initialize arbitrage bot
arb = MultiChainArbitrage(
    chains=[Chain.BASE, Chain.ARBITRUM, Chain.OPTIMISM],
    min_profit_usd=10.0  # Minimum $10 profit
)

# Find arbitrage opportunities
opportunities = arb.find_opportunities(
    token_pair=("ETH", "USDC")
)

for opp in opportunities:
    print(f"Buy on {opp.buy_chain} at {opp.buy_price}")
    print(f"Sell on {opp.sell_chain} at {opp.sell_price}")
    print(f"Profit: ${opp.profit_usd}")
```

---

## 🛡️ Security Best Practices

### 1. **Use Separate Wallets**
```bash
# Development wallet (small amounts)
ETH_PRIVATE_KEY_DEV=0x...

# Production wallet (larger amounts)
ETH_PRIVATE_KEY_PROD=0x...
```

### 2. **Set Transaction Limits**
```python
# In your trading config
MAX_TRADE_SIZE_ETH = 0.1  # Max 0.1 ETH per trade
MAX_DAILY_VOLUME_USD = 1000  # Max $1000 per day
```

### 3. **Test on Base Sepolia First**
```bash
# Base testnet
BASE_SEPOLIA_RPC_URL=https://sepolia.base.org
```

### 4. **Monitor Gas Prices**
```python
# Check gas before trading
base_connection = manager.get_connection(Chain.BASE)
gas_price = base_connection.get_gas_price()

if gas_price['standard'] > 0.1:  # More than 0.1 Gwei
    print("⚠️ Gas prices high, waiting...")
```

---

## 📈 Base Trading Strategies

### 1. **Low-Cost Arbitrage**
Base's low fees make micro-arbitrage profitable:
- DEX arbitrage (Uniswap vs Aerodrome)
- Cross-chain arbitrage (Base ↔ Optimism)

### 2. **High-Frequency Trading**
Fast block times enable HFT strategies:
- Market making on Base DEXs
- MEV opportunities (less competition than Ethereum)

### 3. **Yield Farming**
Deploy capital in Base protocols:
- Aerodrome liquidity pools
- Moonwell lending markets

---

## 🐛 Troubleshooting

### Issue: "Failed to connect to Base"

**Solution**:
```bash
# Try alternate RPC
BASE_RPC_URL=https://base.llamarpc.com

# Or use Alchemy/Infura (requires API key)
BASE_RPC_URL=https://base-mainnet.g.alchemy.com/v2/YOUR_KEY
```

### Issue: "Insufficient funds for gas"

**Solution**:
- Bridge more ETH to Base
- Minimum: ~0.001 ETH for gas

### Issue: "Transaction reverted"

**Solution**:
- Increase slippage tolerance (try 1-2%)
- Check token approvals
- Verify sufficient balance

---

## 📚 Resources

### Official Links
- **Base Website**: https://base.org
- **Base Bridge**: https://bridge.base.org
- **Base Docs**: https://docs.base.org
- **BaseScan Explorer**: https://basescan.org

### Developer Resources
- **Base RPC**: https://mainnet.base.org
- **Chain ID**: 8453
- **Block Explorer API**: https://api.basescan.org/api

### Community
- **Base Discord**: https://discord.gg/buildonbase
- **Base Twitter**: @BuildOnBase

---

## 🎉 You're Ready!

Your trading AI is now connected to Base and ready to trade!

**Next steps**:
1. ✅ Fund your Base wallet (bridge ETH)
2. ✅ Test connection: `python3 examples/defi/demo_multi_chain.py`
3. ✅ Start trading on Base DEXs
4. ✅ Explore arbitrage opportunities

**Need help?** Check the docs or run the demos!

---

**Created**: 2026-02-17
**System**: Trading AI Multi-Chain Platform
**Supported Chains**: Ethereum, Base, Arbitrum, Optimism, Polygon, BSC, Avalanche
