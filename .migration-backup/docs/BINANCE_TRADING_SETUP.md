# Binance Trading Setup Guide

## Overview

This guide covers setting up the Binance trading client for live trading. **Read this entire guide before connecting to live trading.**

## ⚠️ CRITICAL SAFETY WARNINGS

1. **ALWAYS start with TESTNET** - Never test with real money
2. **API Keys** - Enable only necessary permissions (NO withdrawal)
3. **Small Amounts** - Start with tiny positions
4. **Kill Switch** - Know how to stop the bot immediately
5. **Monitor Closely** - Watch the first few hours constantly
6. **Risk Limits** - Set conservative daily loss limits
7. **Paper Trading First** - Test strategies for weeks before live trading

## Prerequisites

- Binance account (or Binance.US if in USA)
- Verified identity (KYC complete)
- 2FA enabled
- Understanding of trading risks

## Step 1: Binance Testnet Setup

### Create Testnet Account

1. **Go to Binance Testnet**
   - URL: https://testnet.binance.vision/
   - This is a separate system from mainnet
   - Uses fake money (no real risk)

2. **Generate API Keys**
   - Login to testnet
   - Go to API Management
   - Create new API key
   - Save API Key and Secret Key (shown only once!)
   - **IMPORTANT**: These are TESTNET keys (won't work on mainnet)

### Configure Testnet Keys

```bash
# Create .env file in project root
cat > .env << EOF
# Binance Testnet Credentials
BINANCE_API_KEY=your_testnet_api_key_here
BINANCE_API_SECRET=your_testnet_secret_key_here
EOF

# Secure the file
chmod 600 .env
```

### Test Testnet Connection

```bash
cd src/exchanges
python binance_trading_client.py
```

Expected output:
```
✅ Connected to Binance
📅 Server Time: 2025-01-15 12:00:00
💰 BTC Price: $45,000.00
✅ Account accessible
Non-zero balances: 2
  BTC: 1.00000000
  USDT: 10000.00000000
```

## Step 2: Testnet Trading

### Basic Trading Example

```python
from exchanges.binance_trading_client import (
    BinanceTradingClient, OrderSide
)

# Initialize client (testnet=True by default)
client = BinanceTradingClient(testnet=True)

# Get account balance
balance = client.get_balance('USDT')
print(f"USDT Balance: {balance['free']}")

# Get current BTC price
btc_price = client.get_ticker_price('BTCUSDT')
print(f"BTC Price: ${btc_price:,.2f}")

# Place a small market buy order
try:
    order = client.place_market_order(
        symbol='BTCUSDT',
        side=OrderSide.BUY,
        quantity=0.001  # 0.001 BTC
    )
    print(f"Order placed: {order['orderId']}")
    print(f"Status: {order['status']}")

except Exception as e:
    print(f"Order failed: {e}")
```

### Test Your Strategies

Run your trading strategies on testnet for at least 2-4 weeks:

```python
from autonomous_agent.trading_agent import AutonomousTradingAgent, AgentConfig

# Configure agent for testnet
config = AgentConfig(
    initial_capital=10000.0,
    paper_trading=False,  # Use real (testnet) orders
    check_interval_seconds=60,
    max_daily_loss=100.0,
    enabled_strategies=['dca_bot']
)

agent = AutonomousTradingAgent(config)

# Connect to Binance testnet
agent.exchange_client = BinanceTradingClient(testnet=True)

# Run agent
await agent.start()
```

## Step 3: Mainnet Setup (LIVE TRADING)

### ⚠️ Pre-Flight Checklist

Before proceeding, ensure:
- [ ] Tested on testnet for 2-4 weeks minimum
- [ ] Strategies are profitable on testnet
- [ ] Risk limits are configured
- [ ] You understand all strategy behavior
- [ ] You can afford to lose 100% of trading capital
- [ ] You have a kill switch plan
- [ ] You will monitor constantly at first

### Create Mainnet API Keys

1. **Go to Binance.com** (or Binance.US)
2. **Profile → API Management**
3. **Create New API Key**
   - Label: "Trading Bot"
   - API restrictions:
     - ✅ Enable Reading
     - ✅ Enable Spot & Margin Trading
     - ❌ **DISABLE** Withdrawal
     - ❌ **DISABLE** Universal Transfer
     - ❌ **DISABLE** Futures Trading (unless needed)

4. **IP Whitelist** (HIGHLY RECOMMENDED)
   - Add your server's IP address
   - Prevents unauthorized access even if keys leak

5. **Save Keys Securely**
   - Store in password manager
   - Never commit to git
   - Use environment variables only

### Configure Mainnet Keys

```bash
# Create separate .env.production file
cat > .env.production << EOF
# Binance MAINNET Credentials - LIVE TRADING
BINANCE_API_KEY=your_mainnet_api_key_here
BINANCE_API_SECRET=your_mainnet_secret_key_here

# Safety limits
MAX_DAILY_LOSS=50.0
MAX_POSITION_SIZE=0.05
INITIAL_CAPITAL=100.0
EOF

# Secure the file
chmod 600 .env.production

# Never commit this file
echo ".env.production" >> .gitignore
```

### Test Mainnet Connection (READ-ONLY)

```python
from exchanges.binance_trading_client import BinanceTradingClient

# Initialize with MAINNET (testnet=False)
client = BinanceTradingClient(testnet=False)

# Test connectivity
if client.test_connectivity():
    print("✅ Connected to Binance MAINNET")

# Get balance (read-only, safe)
balances = client.get_balances()
for balance in balances:
    print(f"{balance['asset']}: {balance['total']}")
```

## Step 4: First Live Trade

### Start with Minimal Capital

```python
# VERY conservative first trade
config = AgentConfig(
    initial_capital=50.0,  # Start with $50 ONLY
    paper_trading=False,   # Live trading
    max_daily_loss=5.0,    # Max $5 loss per day
    max_position_size=0.1, # Max 10% per position
    enabled_strategies=['dca_bot']  # Simple strategy only
)

agent = AutonomousTradingAgent(config)
agent.exchange_client = BinanceTradingClient(testnet=False)

# Start with close monitoring
await agent.start()
```

### Monitoring Your First Trades

1. **Watch the Dashboard**
   ```bash
   streamlit run src/dashboard/streamlit_app.py
   ```

2. **Check Logs**
   ```bash
   tail -f logs/trading_agent.log
   ```

3. **Monitor Binance App**
   - Keep Binance mobile app open
   - Watch for order notifications
   - Check positions regularly

## Step 5: Scaling Up

After 1-2 weeks of successful live trading with minimal capital:

### Gradual Scaling Plan

| Week | Capital | Daily Loss Limit | Notes |
|------|---------|------------------|-------|
| 1-2  | $50     | $5               | Learning phase |
| 3-4  | $200    | $20              | Build confidence |
| 5-6  | $500    | $50              | Monitor closely |
| 7-8  | $1,000  | $100             | Normal operation |
| 9+   | Custom  | 5% of capital    | Scale as comfortable |

### Never:
- ❌ Go all-in immediately
- ❌ Trade more than you can afford to lose
- ❌ Ignore stop losses
- ❌ Disable risk limits
- ❌ Leave bot unmonitored for days
- ❌ Add funds after large losses (take a break)

## Emergency Procedures

### Kill Switch (IMMEDIATE STOP)

#### Method 1: Stop Agent
```python
# In Python console
agent.stop()

# Or via API
curl -X POST http://localhost:8000/api/v1/agents/{agent_id}/stop \
  -H "X-API-Key: your_key"
```

#### Method 2: Cancel All Orders
```python
client = BinanceTradingClient(testnet=False)

# Cancel all orders for a symbol
client.cancel_all_orders('BTCUSDT')

# Or cancel all orders (all symbols)
for symbol in ['BTCUSDT', 'ETHUSDT']:
    try:
        client.cancel_all_orders(symbol)
    except:
        pass
```

#### Method 3: Disable API Key
1. Log into Binance
2. API Management
3. Delete or disable the API key
4. Takes effect immediately

### Recovery Procedures

If something goes wrong:

1. **STOP IMMEDIATELY** - Use kill switch
2. **Review Trades** - Check what happened
3. **Calculate Loss** - Understand damage
4. **Take a Break** - Don't revenge trade
5. **Analyze Root Cause** - What went wrong?
6. **Fix the Issue** - Update code/config
7. **Test on Testnet** - Verify fix
8. **Restart Gradually** - Don't jump back in

## Security Best Practices

### API Key Security

1. **Never Share API Keys**
   - Don't post in forums
   - Don't commit to git
   - Don't send in email

2. **Rotate Keys Regularly**
   - Change keys every 90 days
   - Change immediately if compromised

3. **Use IP Whitelist**
   - Restrict to known IPs
   - Update when IP changes

4. **Monitor API Usage**
   - Check Binance API logs
   - Look for unauthorized access
   - Set up alerts for suspicious activity

### System Security

1. **Secure the Server**
   ```bash
   # Firewall
   ufw allow 22/tcp   # SSH only
   ufw enable

   # SSH key auth only
   # Disable password authentication
   ```

2. **Encrypt Sensitive Files**
   ```bash
   # Encrypt .env files
   gpg -c .env.production

   # Decrypt when needed
   gpg .env.production.gpg
   ```

3. **Audit Logs**
   - Review trading logs daily
   - Check for anomalies
   - Monitor performance metrics

## Troubleshooting

### Common Issues

#### 1. "Signature verification failed"
**Cause**: Time synchronization issue

**Solution**:
```bash
# Sync system time
sudo ntpdate pool.ntp.org

# Or install NTP
sudo apt-get install ntp
```

#### 2. "API key not authorized"
**Cause**: API permissions not enabled

**Solution**:
- Go to Binance API Management
- Edit API key
- Enable "Spot & Margin Trading"
- Save changes

#### 3. "Rate limit exceeded"
**Cause**: Too many requests

**Solution**:
- Reduce check_interval_seconds
- Implement request caching
- Use WebSocket for real-time data

#### 4. "Insufficient balance"
**Cause**: Not enough funds

**Solution**:
- Check balance: `client.get_balance('USDT')`
- Reduce position size
- Add funds to account

#### 5. "Invalid symbol"
**Cause**: Wrong trading pair format

**Solution**:
- Use correct format: 'BTCUSDT' (not 'BTC-USDT')
- Check available symbols: `client.get_exchange_info()`

## Performance Optimization

### Reduce API Calls

```python
# Use WebSocket for real-time data
from realtime import BinanceWebSocket, BinanceConfig

ws_config = BinanceConfig(
    symbols=['BTCUSDT', 'ETHUSDT'],
    streams=[BinanceStream.TICKER],
    testnet=False
)

ws = BinanceWebSocket(ws_config)
await ws.connect()

# Use cached prices instead of API calls
```

### Batch Operations

```python
# Get multiple balances at once
balances = client.get_balances()

# Instead of multiple calls
# balance_btc = client.get_balance('BTC')
# balance_eth = client.get_balance('ETH')
# ...
```

## Compliance and Regulations

### Know Your Obligations

1. **Tax Reporting**
   - Keep records of all trades
   - Report capital gains/losses
   - Consult tax professional

2. **Regulatory Compliance**
   - Know your local regulations
   - Binance.US for USA customers
   - Some countries restrict trading

3. **Risk Disclosure**
   - Trading involves substantial risk
   - Past performance ≠ future results
   - Can lose more than invested (with leverage)

## Support Resources

### Documentation
- Binance API Docs: https://binance-docs.github.io/apidocs/spot/en/
- Project Docs: `/docs`

### Getting Help
- Binance Support: https://www.binance.com/en/support
- Community: Discord/Telegram
- GitHub Issues: Report bugs

### Important Links
- Testnet: https://testnet.binance.vision/
- Mainnet: https://www.binance.com/
- API Status: https://www.binance.com/en/support/announcement
- Fee Structure: https://www.binance.com/en/fee/schedule

## Final Checklist

Before going live:
- [ ] Tested on testnet for 2+ weeks
- [ ] Profitable on testnet
- [ ] API keys created with minimal permissions
- [ ] IP whitelist configured
- [ ] Risk limits set conservatively
- [ ] Kill switch tested
- [ ] Dashboard running and monitored
- [ ] Logs being collected
- [ ] Starting with minimal capital
- [ ] Prepared for losses
- [ ] Emergency procedures documented
- [ ] Team/family aware (if applicable)

## Remember

- **Start small**
- **Monitor constantly**
- **Respect risk limits**
- **Can always scale up later**
- **Can't undo large losses**

**TRADE AT YOUR OWN RISK**

Good luck, and trade safely! 🚀
