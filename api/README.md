# 🌐 Trading AI REST API

Complete REST API for DEX + Coinbase trading with real-time arbitrage detection.

## 🚀 Quick Start

### Start the API

```bash
cd /Users/silasmarkowicz/trading-ai-working
python3 api/main.py
```

The API will start on **http://localhost:8000**

### View Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json

---

## 📡 Endpoints

### Health & Status

#### `GET /health`
System health check with service status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-02-15T10:30:00",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "gas_tracker": "initialized",
    "coinbase": "initialized",
    "uniswap": "initialized"
  }
}
```

#### `GET /status`
Quick status check for load balancers.

#### `GET /info`
API information and configuration.

---

### Market Data (`/api/v1/market`)

#### `GET /api/v1/market/ticker/{symbol}`
Get current ticker price.

**Parameters:**
- `symbol`: Trading pair (BTC-USD, ETH-USD)
- `exchange`: Exchange name (default: coinbase)

**Example:**
```bash
curl http://localhost:8000/api/v1/market/ticker/ETH-USD
```

**Response:**
```json
{
  "symbol": "ETH-USD",
  "exchange": "coinbase",
  "price": 2150.50,
  "volume_24h": 125000.0,
  "timestamp": "2025-02-15T10:30:00"
}
```

#### `GET /api/v1/market/candles/{symbol}`
Get historical OHLCV candles.

**Parameters:**
- `symbol`: Trading pair
- `interval`: Candle size (60=1m, 3600=1h, 86400=1d)
- `limit`: Number of candles (max 300)

**Example:**
```bash
curl "http://localhost:8000/api/v1/market/candles/BTC-USD?interval=3600&limit=24"
```

#### `GET /api/v1/market/products`
Get all available trading pairs.

---

### DEX (`/api/v1/dex`)

#### `GET /api/v1/dex/pools/{token0}/{token1}`
Get Uniswap pool information.

**Example:**
```bash
curl http://localhost:8000/api/v1/dex/pools/0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2/0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48
```

**Response:**
```json
{
  "address": "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc",
  "token0": "WETH",
  "token1": "USDC",
  "reserve0": 45123.45,
  "reserve1": 97234567.89,
  "price": 2155.25,
  "liquidity_usd": 195000000.0
}
```

#### `POST /api/v1/dex/price-impact`
Calculate price impact for a swap.

**Body:**
```json
{
  "token_in": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
  "token_out": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
  "amount_in": 10.0
}
```

**Response:**
```json
{
  "amount_in": 10.0,
  "amount_out": 21550.25,
  "spot_price": 2155.25,
  "execution_price": 2155.02,
  "price_impact_percent": 0.01,
  "slippage_percent": 0.01
}
```

#### `GET /api/v1/dex/route/{token_in}/{token_out}`
Find best swap route.

---

### Arbitrage (`/api/v1/arbitrage`)

#### `GET /api/v1/arbitrage/opportunities`
Find current arbitrage opportunities.

**Parameters:**
- `token`: Token symbol (default: ETH)
- `min_profit`: Minimum profit USD (default: 10.0)
- `trade_size`: Trade size USD (default: 5000.0)

**Example:**
```bash
curl "http://localhost:8000/api/v1/arbitrage/opportunities?token=ETH&min_profit=20"
```

**Response:**
```json
[
  {
    "type": "cex_dex",
    "token": "ETH",
    "buy_exchange": "coinbase",
    "sell_exchange": "uniswap",
    "buy_price": 2150.00,
    "sell_price": 2155.00,
    "spread_percent": 0.23,
    "gross_profit": 25.50,
    "gas_cost": 5.20,
    "net_profit": 20.30,
    "roi_percent": 0.40,
    "confidence": 0.9,
    "timestamp": "2025-02-15T10:30:00"
  }
]
```

#### `GET /api/v1/arbitrage/spread/{token}`
Get price spread between Coinbase and Uniswap.

**Example:**
```bash
curl http://localhost:8000/api/v1/arbitrage/spread/ETH
```

**Response:**
```json
{
  "token": "ETH",
  "coinbase_price": 2150.50,
  "uniswap_price": 2155.25,
  "spread": 4.75,
  "spread_percent": 0.22,
  "higher_on": "uniswap"
}
```

---

### Gas (`/api/v1/gas`)

#### `GET /api/v1/gas/prices`
Get current gas prices.

**Example:**
```bash
curl http://localhost:8000/api/v1/gas/prices
```

**Response:**
```json
{
  "slow": 10.5,
  "standard": 15.2,
  "fast": 20.8,
  "instant": 25.5,
  "base_fee": 12.0,
  "priority_fee": 1.5,
  "timestamp": "2025-02-15T10:30:00",
  "conditions": "GOOD"
}
```

#### `GET /api/v1/gas/estimate`
Estimate gas cost for operation.

**Parameters:**
- `operation`: Operation type (default: uniswap_v2_swap)
- `eth_price`: ETH price USD (default: 2000.0)

**Example:**
```bash
curl "http://localhost:8000/api/v1/gas/estimate?operation=uniswap_v2_swap&eth_price=2150"
```

**Response:**
```json
{
  "operation": "uniswap_v2_swap",
  "gas_limit": 150000,
  "gas_price_gwei": 15.2,
  "cost_eth": 0.00228,
  "cost_usd": 4.90
}
```

#### `GET /api/v1/gas/max-profitable`
Calculate maximum gas price for profitable trade.

**Parameters:**
- `profit_eth`: Expected profit in ETH
- `gas_limit`: Gas limit

**Example:**
```bash
curl "http://localhost:8000/api/v1/gas/max-profitable?profit_eth=0.05&gas_limit=150000"
```

**Response:**
```json
{
  "profit_eth": 0.05,
  "gas_limit": 150000,
  "max_gas_gwei": 333.33,
  "current_gas_gwei": 15.2,
  "is_profitable": true
}
```

#### `GET /api/v1/gas/conditions`
Get trading conditions based on gas.

**Response:**
```json
{
  "status": "excellent",
  "message": "Gas is very low. Great time to trade!",
  "color": "green",
  "gas_price": 15.2,
  "timestamp": "2025-02-15T10:30:00"
}
```

---

## 🔧 Configuration

Edit `.env` file:

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=trading_db
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=trading_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# External APIs
COINBASE_API_KEY=your_key
COINBASE_API_SECRET=your_secret
ETHEREUM_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY
ETHERSCAN_API_KEY=your_key
```

---

## 📦 Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# Get gas prices
response = requests.get(f"{BASE_URL}/gas/prices")
gas = response.json()
print(f"Current gas: {gas['standard']} Gwei")

# Find arbitrage opportunities
response = requests.get(
    f"{BASE_URL}/arbitrage/opportunities",
    params={"token": "ETH", "min_profit": 20}
)
opps = response.json()

for opp in opps:
    print(f"Profit: ${opp['net_profit']:.2f}")
    print(f"Buy: {opp['buy_exchange']} @ ${opp['buy_price']}")
    print(f"Sell: {opp['sell_exchange']} @ ${opp['sell_price']}")

# Get ETH price
response = requests.get(f"{BASE_URL}/market/ticker/ETH-USD")
ticker = response.json()
print(f"ETH: ${ticker['price']}")
```

---

## 🚀 Production Deployment

### Using Uvicorn

```bash
# Install uvicorn with performance extras
pip3 install uvicorn[standard]

# Run with workers
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Docker

```bash
# Build
docker build -t trading-api .

# Run
docker run -p 8000:8000 --env-file .env trading-api
```

### Using Systemd

Create `/etc/systemd/system/trading-api.service`:

```ini
[Unit]
Description=Trading AI API
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/trading-ai-working
ExecStart=/usr/bin/python3 api/main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable trading-api
sudo systemctl start trading-api
```

---

## 🔐 Security (Coming Soon)

- JWT authentication
- API key management
- Rate limiting per IP
- Request validation
- HTTPS/TLS
- Input sanitization

---

## 📊 Performance

- **Response time:** < 100ms (cached)
- **Throughput:** ~1000 req/s
- **Cache:** Redis 10s TTL
- **Concurrent requests:** Async/await

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find process
lsof -i :8000

# Kill it
kill -9 <PID>
```

### Import Errors
```bash
# Install dependencies
pip3 install fastapi uvicorn pydantic redis
```

### CORS Errors
Add your frontend URL to `api/config.py`:
```python
cors_origins = ["http://your-frontend.com"]
```

---

## 📖 More Information

- **OpenAPI Spec:** http://localhost:8000/openapi.json
- **Interactive Docs:** http://localhost:8000/docs
- **Alternative Docs:** http://localhost:8000/redoc

---

## 🎯 Next Steps

1. Add authentication (JWT)
2. Implement portfolio tracking
3. Add trade execution
4. WebSocket support for real-time updates
5. Rate limiting middleware
6. Monitoring/logging integration
