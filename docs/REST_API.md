# Trading AI REST API Documentation

## Overview

The Trading AI REST API provides programmatic access to all trading system functionality including agent control, portfolio management, market data, risk analysis, and trading signals. Built with FastAPI, it features automatic OpenAPI documentation, request validation, and comprehensive error handling.

## Base URL

```
Production: https://api.trading-ai.example.com
Development: http://localhost:8000
```

## Authentication

All endpoints (except health checks) require an API key in the request header:

```http
X-API-Key: sk_your_api_key_here
```

API keys can be generated in the dashboard or via the CLI.

## Rate Limiting

| Tier | Rate Limit | Burst |
|------|-----------|--------|
| Free | 60 req/min | 10 req/sec |
| Pro | 600 req/min | 100 req/sec |
| Enterprise | Unlimited | Unlimited |

Rate limit information is returned in response headers:
- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Unix timestamp when limit resets

## Endpoints

### Health & Status

#### GET /health/
Basic health check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T12:00:00Z",
  "service": "trading-ai-api"
}
```

#### GET /health/detailed
Detailed health check with system metrics.

#### GET /health/ready
Readiness check for Kubernetes (checks dependencies).

#### GET /health/live
Liveness check for Kubernetes.

---

### Agent Control

#### POST /api/v1/agents/
Create a new trading agent.

**Request Body:**
```json
{
  "initial_capital": 10000.0,
  "paper_trading": true,
  "check_interval_seconds": 5,
  "max_daily_loss": 500.0,
  "max_position_size": 0.2,
  "enabled_strategies": ["dca_bot", "momentum"],
  "send_alerts": false
}
```

**Response:**
```json
{
  "agent_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "created",
  "message": "Agent created successfully"
}
```

#### GET /api/v1/agents/
List all agents.

**Response:**
```json
[
  {
    "agent_id": "550e8400-e29b-41d4-a716-446655440000",
    "state": "running",
    "portfolio_value": 11250.50,
    "total_pnl": 1250.50,
    "created_at": "2025-01-15T10:00:00Z"
  }
]
```

#### GET /api/v1/agents/{agent_id}
Get agent status.

**Response:**
```json
{
  "agent_id": "550e8400-e29b-41d4-a716-446655440000",
  "state": "running",
  "uptime_seconds": 3600.0,
  "portfolio_value": 11250.50,
  "total_pnl": 1250.50,
  "daily_pnl": 150.25,
  "total_trades": 125,
  "active_positions": 3,
  "enabled_strategies": ["dca_bot", "momentum"],
  "last_update": "2025-01-15T12:00:00Z"
}
```

#### POST /api/v1/agents/{agent_id}/start
Start an agent.

#### POST /api/v1/agents/{agent_id}/stop
Stop an agent.

#### POST /api/v1/agents/{agent_id}/pause
Pause an agent temporarily.

#### POST /api/v1/agents/{agent_id}/resume
Resume a paused agent.

#### DELETE /api/v1/agents/{agent_id}
Delete an agent (must be stopped first).

#### GET /api/v1/agents/{agent_id}/metrics
Get agent performance metrics.

**Response:**
```json
{
  "sharpe_ratio": 1.85,
  "max_drawdown": 0.08,
  "win_rate": 0.656,
  "total_return_percent": 12.51,
  "total_trades": 125,
  "winning_trades": 82,
  "losing_trades": 43,
  "avg_win": 58.32,
  "avg_loss": -32.15,
  "profit_factor": 2.15
}
```

#### POST /api/v1/agents/{agent_id}/strategies
Enable or disable a strategy.

**Request Body:**
```json
{
  "strategy_name": "market_making",
  "enabled": true
}
```

#### PUT /api/v1/agents/{agent_id}/risk-limits
Update risk limits.

**Request Body:**
```json
{
  "max_daily_loss": 600.0,
  "max_position_size": 0.25,
  "max_drawdown": 0.15
}
```

#### GET /api/v1/agents/{agent_id}/strategies/performance
Get performance by strategy.

**Response:**
```json
[
  {
    "strategy": "dca_bot",
    "total_trades": 45,
    "total_pnl": 550.25,
    "win_rate": 0.78,
    "avg_pnl_per_trade": 12.23
  },
  {
    "strategy": "momentum",
    "total_trades": 80,
    "total_pnl": 700.25,
    "win_rate": 0.60,
    "avg_pnl_per_trade": 8.75
  }
]
```

---

### Portfolio Management

#### GET /api/v1/portfolio/summary
Get portfolio summary.

**Query Parameters:**
- `agent_id` (optional): Filter by agent

**Response:**
```json
{
  "total_value_usd": 11250.50,
  "cash_balance_usd": 5000.00,
  "positions_value_usd": 6250.50,
  "total_pnl": 1250.50,
  "total_pnl_percent": 12.51,
  "daily_pnl": 150.25,
  "num_positions": 3,
  "sharpe_ratio": 1.85,
  "max_drawdown": 0.08,
  "win_rate": 0.65
}
```

#### GET /api/v1/portfolio/positions
Get open positions.

**Query Parameters:**
- `agent_id` (optional)
- `symbol` (optional): Filter by symbol

**Response:**
```json
[
  {
    "symbol": "BTC-USD",
    "quantity": 0.15,
    "entry_price": 42000.00,
    "current_price": 45000.00,
    "unrealized_pnl": 450.00,
    "unrealized_pnl_percent": 7.14,
    "value_usd": 6750.00
  }
]
```

#### GET /api/v1/portfolio/trades
Get trade history.

**Query Parameters:**
- `agent_id` (optional)
- `symbol` (optional)
- `start_date` (optional): ISO format
- `end_date` (optional): ISO format
- `limit` (default: 100, max: 1000)

**Response:**
```json
[
  {
    "id": "trade_001",
    "symbol": "BTC-USD",
    "side": "BUY",
    "quantity": 0.1,
    "price": 42000.00,
    "value": 4200.00,
    "fee": 4.20,
    "pnl": null,
    "strategy": "dca_bot",
    "executed_at": "2025-01-15T10:30:00Z"
  }
]
```

#### GET /api/v1/portfolio/performance
Get portfolio performance over time.

**Query Parameters:**
- `agent_id` (optional)
- `period`: `1d`, `7d`, `30d`, `90d`, `1y`, `all` (default: `7d`)

**Response:**
```json
{
  "period": "7d",
  "equity_curve": [
    {"timestamp": "2025-01-08T00:00:00Z", "value": 10000.00},
    {"timestamp": "2025-01-09T00:00:00Z", "value": 10150.00},
    ...
  ],
  "metrics": {
    "start_value": 10000.00,
    "end_value": 11250.50,
    "total_return": 12.51,
    "volatility": 0.15,
    "sharpe_ratio": 1.85,
    "max_drawdown": 0.08,
    "calmar_ratio": 2.31
  }
}
```

#### GET /api/v1/portfolio/statistics
Get detailed portfolio statistics.

#### GET /api/v1/portfolio/allocation
Get asset allocation.

#### GET /api/v1/portfolio/risk-metrics
Get portfolio risk metrics including VaR, CVaR, and exposure.

---

### Market Data

#### GET /api/v1/market/ticker/{symbol}
Get current ticker data.

**Query Parameters:**
- `exchange` (default: `binance`)

**Response:**
```json
{
  "symbol": "BTC-USD",
  "price": 45000.50,
  "volume_24h": 5000.25,
  "high_24h": 46000.00,
  "low_24h": 44000.00,
  "price_change_24h": 500.50,
  "price_change_percent_24h": 1.12,
  "timestamp": "2025-01-15T12:00:00Z"
}
```

#### GET /api/v1/market/tickers
Get multiple tickers.

**Query Parameters:**
- `symbols`: Comma-separated (e.g., `BTC-USD,ETH-USD,SOL-USD`)
- `exchange` (default: `binance`)

#### GET /api/v1/market/orderbook/{symbol}
Get order book.

**Query Parameters:**
- `exchange` (default: `binance`)
- `depth` (default: 10, max: 100)

**Response:**
```json
{
  "symbol": "BTC-USD",
  "bids": [[45000.00, 1.5], [44999.50, 2.0]],
  "asks": [[45001.00, 1.2], [45002.00, 3.5]],
  "timestamp": "2025-01-15T12:00:00Z"
}
```

#### GET /api/v1/market/candles/{symbol}
Get candlestick/OHLCV data.

**Query Parameters:**
- `interval`: `1m`, `5m`, `15m`, `1h`, `4h`, `1d` (default: `1h`)
- `exchange` (default: `binance`)
- `limit` (default: 100, max: 1000)

**Response:**
```json
[
  {
    "timestamp": "2025-01-15T11:00:00Z",
    "open": 44500.00,
    "high": 45200.00,
    "low": 44300.00,
    "close": 45000.00,
    "volume": 125.50
  }
]
```

#### GET /api/v1/market/trades/{symbol}
Get recent trades.

#### GET /api/v1/market/exchanges
List supported exchanges.

#### GET /api/v1/market/markets
List available trading pairs.

---

### Trading Signals

#### GET /api/v1/signals/active
Get active trading signals.

**Query Parameters:**
- `symbol` (optional)
- `strategy` (optional)
- `min_confidence` (default: 0.0)

**Response:**
```json
[
  {
    "id": "signal_001",
    "symbol": "BTC-USD",
    "strategy": "momentum",
    "action": "BUY",
    "confidence": 0.85,
    "price": 45000.00,
    "target_price": 47000.00,
    "stop_loss": 43500.00,
    "reason": "Strong upward momentum, RSI not overbought",
    "timestamp": "2025-01-15T12:00:00Z"
  }
]
```

#### GET /api/v1/signals/history
Get historical signals.

#### GET /api/v1/signals/strategies
List available strategies.

**Response:**
```json
[
  {
    "name": "dca_bot",
    "display_name": "Dollar Cost Averaging",
    "description": "Systematic periodic buying regardless of price",
    "type": "accumulation",
    "parameters": ["frequency", "amount", "dip_threshold"],
    "active": true
  }
]
```

#### GET /api/v1/signals/performance/{strategy}
Get strategy performance metrics.

#### POST /api/v1/signals/backtest
Run strategy backtest.

**Request Body:**
```json
{
  "strategy": "momentum",
  "symbol": "BTC-USD",
  "start_date": "2025-01-01",
  "end_date": "2025-01-15",
  "parameters": {
    "lookback_period": 50,
    "adx_threshold": 25.0
  }
}
```

---

### Risk Management

#### GET /api/v1/risk/var
Calculate Value at Risk.

**Query Parameters:**
- `portfolio_id` (required)
- `confidence` (default: 0.95)
- `horizon_days` (default: 1)

**Response:**
```json
{
  "var_95": 450.25,
  "var_99": 725.50,
  "cvar_95": 625.50,
  "cvar_99": 950.75,
  "method": "historical_simulation",
  "confidence": 0.95,
  "horizon_days": 1
}
```

#### GET /api/v1/risk/position-sizing
Calculate optimal position size.

**Query Parameters:**
- `symbol` (required)
- `risk_per_trade` (default: 0.02)
- `stop_loss_pct` (default: 0.05)

**Response:**
```json
{
  "symbol": "BTC-USD",
  "portfolio_value": 10000.0,
  "risk_per_trade_pct": 2.0,
  "risk_amount_usd": 200.0,
  "stop_loss_pct": 5.0,
  "recommended_position_size_usd": 4000.0,
  "max_position_size_usd": 2000.0,
  "kelly_criterion": 0.15
}
```

#### GET /api/v1/risk/correlation-matrix
Get asset correlation matrix.

#### GET /api/v1/risk/exposure
Get portfolio exposure analysis.

#### GET /api/v1/risk/stress-test
Run portfolio stress test.

#### GET /api/v1/risk/drawdown
Analyze portfolio drawdowns.

---

## Error Responses

All errors follow a consistent format:

```json
{
  "error": "Error message description",
  "status_code": 400,
  "path": "/api/v1/agents/invalid-id"
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized (missing API key) |
| 403 | Forbidden (invalid API key) |
| 404 | Not Found |
| 429 | Too Many Requests (rate limited) |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

---

## Examples

### Python

```python
import requests

API_BASE = "http://localhost:8000"
API_KEY = "sk_your_api_key"

headers = {"X-API-Key": API_KEY}

# Create agent
response = requests.post(
    f"{API_BASE}/api/v1/agents/",
    headers=headers,
    json={
        "initial_capital": 10000.0,
        "paper_trading": True,
        "enabled_strategies": ["dca_bot"]
    }
)

agent_id = response.json()["agent_id"]

# Start agent
requests.post(
    f"{API_BASE}/api/v1/agents/{agent_id}/start",
    headers=headers
)

# Get status
status = requests.get(
    f"{API_BASE}/api/v1/agents/{agent_id}",
    headers=headers
).json()

print(f"Portfolio Value: ${status['portfolio_value']}")
```

### cURL

```bash
# Create agent
curl -X POST http://localhost:8000/api/v1/agents/ \
  -H "X-API-Key: sk_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_capital": 10000.0,
    "paper_trading": true,
    "enabled_strategies": ["dca_bot"]
  }'

# Get ticker
curl -X GET "http://localhost:8000/api/v1/market/ticker/BTC-USD" \
  -H "X-API-Key: sk_your_api_key"
```

### JavaScript

```javascript
const API_BASE = "http://localhost:8000";
const API_KEY = "sk_your_api_key";

// Create agent
const response = await fetch(`${API_BASE}/api/v1/agents/`, {
  method: "POST",
  headers: {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    initial_capital: 10000.0,
    paper_trading: true,
    enabled_strategies: ["dca_bot"]
  })
});

const { agent_id } = await response.json();

// Start agent
await fetch(`${API_BASE}/api/v1/agents/${agent_id}/start`, {
  method: "POST",
  headers: { "X-API-Key": API_KEY }
});
```

---

## Interactive Documentation

The API provides interactive documentation via Swagger UI and ReDoc:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

These interfaces allow you to:
- Browse all available endpoints
- View request/response schemas
- Test API calls directly in the browser
- Generate client code

---

## Running the API

### Development

```bash
# Start with uvicorn
cd src/api
python main.py

# Or with hot reload
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
# Using Docker
docker build -t trading-ai-api .
docker run -p 8000:8000 trading-ai-api

# Using docker-compose
docker-compose up api
```

---

## WebSocket Support

For real-time updates, connect to the WebSocket endpoint:

```
ws://localhost:8000/ws
```

Supported message types:
- `ticker_update`: Real-time price updates
- `signal_generated`: New trading signals
- `trade_executed`: Trade executions
- `agent_status`: Agent status changes

---

## SDKs and Client Libraries

Official client libraries:
- Python: `pip install trading-ai-client`
- JavaScript/TypeScript: `npm install @trading-ai/client`
- Go: `go get github.com/trading-ai/client-go`

---

## Support

- Documentation: https://docs.trading-ai.example.com
- API Status: https://status.trading-ai.example.com
- Support: support@trading-ai.example.com
- GitHub: https://github.com/trading-ai/api
