# API Reference

**Version**: 2.0
**Base URL**: `http://localhost:8000`
**Authentication**: Bearer Token (JWT) or API Key

---

## Table of Contents

1. [Authentication](#authentication)
2. [Agent Management](#agent-management)
3. [Portfolio Operations](#portfolio-operations)
4. [Trading Signals](#trading-signals)
5. [Market Data](#market-data)
6. [Risk Management](#risk-management)
7. [Health & Monitoring](#health--monitoring)
8. [Error Handling](#error-handling)

---

## Authentication

### Generate Token

**Endpoint**: `POST /auth/token`

**Request**:
```json
{
  "username": "trader",
  "password": "secure_password"
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

**Using the Token**:
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/api/v1/agents/
```

### API Key Authentication

**Alternative**: Use API key in header
```bash
curl -H "X-API-Key: YOUR_API_KEY" \
     http://localhost:8000/api/v1/agents/
```

---

## Agent Management

### List All Agents

**Endpoint**: `GET /api/v1/agents/`

**Response**:
```json
{
  "agents": [
    {
      "agent_id": "agent-001",
      "name": "DCA Trading Bot",
      "status": "running",
      "portfolio_value": 12500.50,
      "strategies": ["dca_bot", "momentum"],
      "created_at": "2026-02-15T10:00:00Z"
    }
  ],
  "total": 1
}
```

---

### Get Agent Details

**Endpoint**: `GET /api/v1/agents/{agent_id}`

**Response**:
```json
{
  "agent_id": "agent-001",
  "name": "DCA Trading Bot",
  "status": "running",
  "config": {
    "initial_capital": 10000.0,
    "max_daily_loss": 500.0,
    "max_position_size": 0.2,
    "check_interval_seconds": 60
  },
  "portfolio": {
    "value": 12500.50,
    "cash": 5000.00,
    "positions": 3
  },
  "performance": {
    "total_pnl": 2500.50,
    "total_pnl_pct": 25.00,
    "sharpe_ratio": 1.85,
    "win_rate": 0.68,
    "max_drawdown": -0.05
  },
  "created_at": "2026-02-15T10:00:00Z",
  "updated_at": "2026-02-16T12:30:00Z"
}
```

---

### Create Agent

**Endpoint**: `POST /api/v1/agents/`

**Request**:
```json
{
  "name": "New Trading Bot",
  "initial_capital": 5000.0,
  "paper_trading": true,
  "max_daily_loss": 250.0,
  "max_position_size": 0.15,
  "strategies": ["dca_bot", "mean_reversion"],
  "check_interval_seconds": 30
}
```

**Response**:
```json
{
  "agent_id": "agent-002",
  "name": "New Trading Bot",
  "status": "idle",
  "created_at": "2026-02-16T13:00:00Z"
}
```

---

### Start Agent

**Endpoint**: `POST /api/v1/agents/{agent_id}/start`

**Response**:
```json
{
  "agent_id": "agent-001",
  "status": "starting",
  "message": "Agent is starting"
}
```

---

### Pause Agent

**Endpoint**: `POST /api/v1/agents/{agent_id}/pause`

**Response**:
```json
{
  "agent_id": "agent-001",
  "status": "paused",
  "message": "Agent paused, positions maintained"
}
```

---

### Resume Agent

**Endpoint**: `POST /api/v1/agents/{agent_id}/resume`

**Response**:
```json
{
  "agent_id": "agent-001",
  "status": "running",
  "message": "Agent resumed"
}
```

---

### Stop Agent

**Endpoint**: `POST /api/v1/agents/{agent_id}/stop`

**Request** (optional):
```json
{
  "close_positions": true
}
```

**Response**:
```json
{
  "agent_id": "agent-001",
  "status": "stopped",
  "message": "Agent stopped, positions closed"
}
```

---

### Delete Agent

**Endpoint**: `DELETE /api/v1/agents/{agent_id}`

**Response**:
```json
{
  "message": "Agent deleted successfully"
}
```

---

### Update Agent Configuration

**Endpoint**: `PATCH /api/v1/agents/{agent_id}/config`

**Request**:
```json
{
  "max_daily_loss": 300.0,
  "check_interval_seconds": 45,
  "strategies": ["dca_bot", "momentum", "grid_trading"]
}
```

**Response**:
```json
{
  "agent_id": "agent-001",
  "config": {
    "max_daily_loss": 300.0,
    "check_interval_seconds": 45,
    "strategies": ["dca_bot", "momentum", "grid_trading"]
  },
  "message": "Configuration updated"
}
```

---

## Portfolio Operations

### Get Portfolio Summary

**Endpoint**: `GET /api/v1/portfolio/{agent_id}`

**Response**:
```json
{
  "agent_id": "agent-001",
  "portfolio_value": 12500.50,
  "cash": 5000.00,
  "positions_value": 7500.50,
  "total_pnl": 2500.50,
  "total_pnl_pct": 25.00,
  "timestamp": "2026-02-16T12:30:00Z"
}
```

---

### Get Positions

**Endpoint**: `GET /api/v1/portfolio/{agent_id}/positions`

**Response**:
```json
{
  "positions": [
    {
      "symbol": "BTCUSDT",
      "quantity": 0.15,
      "avg_price": 45000.00,
      "current_price": 47000.00,
      "value": 7050.00,
      "unrealized_pnl": 300.00,
      "unrealized_pnl_pct": 4.44,
      "opened_at": "2026-02-15T14:30:00Z"
    },
    {
      "symbol": "ETHUSDT",
      "quantity": 2.0,
      "avg_price": 2500.00,
      "current_price": 2525.00,
      "value": 5050.00,
      "unrealized_pnl": 50.00,
      "unrealized_pnl_pct": 1.00,
      "opened_at": "2026-02-16T09:15:00Z"
    }
  ],
  "total": 2
}
```

---

### Get Trade History

**Endpoint**: `GET /api/v1/portfolio/{agent_id}/trades`

**Query Parameters**:
- `limit` (optional): Number of trades (default: 50)
- `offset` (optional): Pagination offset (default: 0)
- `symbol` (optional): Filter by symbol
- `start_date` (optional): ISO format date
- `end_date` (optional): ISO format date

**Example**: `GET /api/v1/portfolio/agent-001/trades?limit=10&symbol=BTCUSDT`

**Response**:
```json
{
  "trades": [
    {
      "trade_id": "trade-12345",
      "symbol": "BTCUSDT",
      "side": "BUY",
      "quantity": 0.05,
      "price": 46500.00,
      "value": 2325.00,
      "fee": 2.325,
      "strategy": "dca_bot",
      "timestamp": "2026-02-16T11:00:00Z"
    },
    {
      "trade_id": "trade-12344",
      "symbol": "ETHUSDT",
      "side": "SELL",
      "quantity": 1.0,
      "price": 2550.00,
      "value": 2550.00,
      "fee": 2.55,
      "pnl": 100.00,
      "strategy": "momentum",
      "timestamp": "2026-02-16T10:30:00Z"
    }
  ],
  "total": 2,
  "limit": 10,
  "offset": 0
}
```

---

### Get Performance Metrics

**Endpoint**: `GET /api/v1/portfolio/{agent_id}/performance`

**Query Parameters**:
- `period` (optional): `1d`, `7d`, `30d`, `90d`, `1y`, `all` (default: `30d`)

**Response**:
```json
{
  "period": "30d",
  "total_pnl": 2500.50,
  "total_pnl_pct": 25.00,
  "sharpe_ratio": 1.85,
  "sortino_ratio": 2.10,
  "max_drawdown": -0.05,
  "max_drawdown_duration_days": 3,
  "win_rate": 0.68,
  "profit_factor": 2.5,
  "avg_win": 150.00,
  "avg_loss": -60.00,
  "total_trades": 45,
  "winning_trades": 31,
  "losing_trades": 14,
  "by_strategy": {
    "dca_bot": {
      "pnl": 1500.00,
      "win_rate": 0.75,
      "trades": 20
    },
    "momentum": {
      "pnl": 1000.50,
      "win_rate": 0.60,
      "trades": 25
    }
  }
}
```

---

## Trading Signals

### Get Latest Signals

**Endpoint**: `GET /api/v1/signals/{agent_id}`

**Response**:
```json
{
  "signals": [
    {
      "strategy": "dca_bot",
      "symbol": "BTCUSDT",
      "action": "BUY",
      "size": 0.01,
      "confidence": 0.85,
      "reason": "Price below moving average",
      "timestamp": "2026-02-16T12:30:00Z"
    },
    {
      "strategy": "momentum",
      "symbol": "ETHUSDT",
      "action": "SELL",
      "size": 0.5,
      "confidence": 0.72,
      "reason": "Overbought RSI",
      "timestamp": "2026-02-16T12:29:00Z"
    }
  ],
  "total": 2
}
```

---

### Generate Signal (Manual)

**Endpoint**: `POST /api/v1/signals/{agent_id}/generate`

**Request**:
```json
{
  "strategy": "dca_bot",
  "symbol": "BTCUSDT"
}
```

**Response**:
```json
{
  "signal": {
    "strategy": "dca_bot",
    "symbol": "BTCUSDT",
    "action": "BUY",
    "size": 0.01,
    "confidence": 0.85,
    "reason": "DCA interval reached",
    "timestamp": "2026-02-16T12:35:00Z"
  }
}
```

---

## Market Data

### Get Current Price

**Endpoint**: `GET /api/v1/market/price/{symbol}`

**Example**: `GET /api/v1/market/price/BTCUSDT`

**Response**:
```json
{
  "symbol": "BTCUSDT",
  "price": 47250.50,
  "timestamp": "2026-02-16T12:35:00Z",
  "source": "binance"
}
```

---

### Get Order Book

**Endpoint**: `GET /api/v1/market/orderbook/{symbol}`

**Query Parameters**:
- `limit` (optional): Depth (default: 10)

**Response**:
```json
{
  "symbol": "BTCUSDT",
  "bids": [
    [47250.00, 2.5],
    [47249.50, 1.2],
    [47249.00, 3.0]
  ],
  "asks": [
    [47251.00, 1.8],
    [47251.50, 2.1],
    [47252.00, 1.5]
  ],
  "timestamp": "2026-02-16T12:35:00Z"
}
```

---

### Get Historical Data

**Endpoint**: `GET /api/v1/market/history/{symbol}`

**Query Parameters**:
- `interval`: `1m`, `5m`, `15m`, `1h`, `4h`, `1d`
- `start_time`: ISO format
- `end_time`: ISO format
- `limit` (optional): Max data points (default: 500)

**Example**: `GET /api/v1/market/history/BTCUSDT?interval=1h&limit=24`

**Response**:
```json
{
  "symbol": "BTCUSDT",
  "interval": "1h",
  "data": [
    {
      "timestamp": "2026-02-16T00:00:00Z",
      "open": 46500.00,
      "high": 46800.00,
      "low": 46400.00,
      "close": 46750.00,
      "volume": 1250.5
    }
  ],
  "count": 24
}
```

---

## Risk Management

### Get Risk Metrics

**Endpoint**: `GET /api/v1/risk/{agent_id}`

**Response**:
```json
{
  "agent_id": "agent-001",
  "risk_metrics": {
    "var_95": -450.00,
    "cvar_95": -620.00,
    "volatility": 0.15,
    "beta": 0.85,
    "max_drawdown": -0.05,
    "daily_var": -250.00
  },
  "exposure": {
    "long": 7500.00,
    "short": 0.00,
    "net": 7500.00,
    "gross": 7500.00,
    "leverage": 0.60
  },
  "limits": {
    "max_daily_loss": 500.00,
    "current_daily_loss": -120.00,
    "remaining": 380.00,
    "max_position_size": 0.20,
    "current_max_position": 0.15
  },
  "timestamp": "2026-02-16T12:35:00Z"
}
```

---

### Update Risk Limits

**Endpoint**: `PATCH /api/v1/risk/{agent_id}/limits`

**Request**:
```json
{
  "max_daily_loss": 600.00,
  "max_position_size": 0.25
}
```

**Response**:
```json
{
  "agent_id": "agent-001",
  "limits": {
    "max_daily_loss": 600.00,
    "max_position_size": 0.25
  },
  "message": "Risk limits updated"
}
```

---

## Health & Monitoring

### Health Check

**Endpoint**: `GET /health/`

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-02-16T12:35:00Z",
  "version": "2.0.0",
  "uptime_seconds": 86400
}
```

---

### Database Health

**Endpoint**: `GET /health/database`

**Response**:
```json
{
  "status": "healthy",
  "latency_ms": 15,
  "connections": {
    "active": 5,
    "idle": 10,
    "total": 15
  }
}
```

---

### Exchange Health

**Endpoint**: `GET /health/exchanges`

**Response**:
```json
{
  "exchanges": [
    {
      "name": "binance",
      "status": "healthy",
      "latency_ms": 120,
      "last_update": "2026-02-16T12:35:00Z"
    },
    {
      "name": "coinbase",
      "status": "healthy",
      "latency_ms": 180,
      "last_update": "2026-02-16T12:34:58Z"
    }
  ]
}
```

---

### Metrics

**Endpoint**: `GET /metrics`

**Format**: Prometheus format

**Response**:
```prometheus
# HELP trading_agent_portfolio_value Portfolio value in USD
# TYPE trading_agent_portfolio_value gauge
trading_agent_portfolio_value{agent_id="agent-001"} 12500.50

# HELP trading_agent_trades_total Total number of trades
# TYPE trading_agent_trades_total counter
trading_agent_trades_total{agent_id="agent-001",strategy="dca_bot"} 45

# HELP api_request_duration_seconds API request duration
# TYPE api_request_duration_seconds histogram
api_request_duration_seconds_bucket{endpoint="/api/v1/agents",le="0.1"} 950
api_request_duration_seconds_bucket{endpoint="/api/v1/agents",le="0.5"} 990
api_request_duration_seconds_bucket{endpoint="/api/v1/agents",le="+Inf"} 1000
```

---

## Error Handling

### Error Response Format

All errors follow this format:

```json
{
  "error": {
    "code": "INVALID_AGENT_ID",
    "message": "Agent not found",
    "details": "Agent with ID 'agent-999' does not exist",
    "timestamp": "2026-02-16T12:35:00Z",
    "request_id": "req-12345"
  }
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created |
| 400 | Bad Request | Invalid request data |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Common Error Codes

| Code | Description |
|------|-------------|
| `INVALID_REQUEST` | Request validation failed |
| `AGENT_NOT_FOUND` | Agent ID doesn't exist |
| `INVALID_STATE` | Invalid agent state transition |
| `INSUFFICIENT_BALANCE` | Not enough funds |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `EXCHANGE_ERROR` | Exchange API error |
| `DATABASE_ERROR` | Database operation failed |
| `AUTHENTICATION_FAILED` | Invalid credentials |

---

## Rate Limiting

**Limits**:
- **Public endpoints**: 100 requests/minute
- **Authenticated endpoints**: 1000 requests/minute
- **Trading endpoints**: 50 requests/minute

**Headers**:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1645876800
```

**Rate Limit Exceeded Response**:
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": "Limit: 1000 req/min. Try again in 30 seconds.",
    "retry_after": 30
  }
}
```

---

## Webhooks

### Configure Webhook

**Endpoint**: `POST /api/v1/webhooks`

**Request**:
```json
{
  "url": "https://your-server.com/webhook",
  "events": ["trade.executed", "agent.stopped", "alert.triggered"],
  "secret": "webhook_secret_key"
}
```

**Response**:
```json
{
  "webhook_id": "webhook-001",
  "url": "https://your-server.com/webhook",
  "events": ["trade.executed", "agent.stopped", "alert.triggered"],
  "created_at": "2026-02-16T12:35:00Z"
}
```

### Webhook Events

**trade.executed**:
```json
{
  "event": "trade.executed",
  "timestamp": "2026-02-16T12:35:00Z",
  "data": {
    "trade_id": "trade-12345",
    "agent_id": "agent-001",
    "symbol": "BTCUSDT",
    "side": "BUY",
    "quantity": 0.01,
    "price": 47250.00
  }
}
```

---

## SDKs & Libraries

### Python
```python
from trading_ai_sdk import TradingAIClient

client = TradingAIClient(
    api_key="YOUR_API_KEY",
    base_url="http://localhost:8000"
)

# List agents
agents = client.agents.list()

# Get portfolio
portfolio = client.portfolio.get("agent-001")

# Start agent
client.agents.start("agent-001")
```

### JavaScript
```javascript
const TradingAI = require('trading-ai-sdk');

const client = new TradingAI({
  apiKey: 'YOUR_API_KEY',
  baseURL: 'http://localhost:8000'
});

// List agents
const agents = await client.agents.list();

// Get portfolio
const portfolio = await client.portfolio.get('agent-001');
```

---

## Interactive Documentation

Access interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json

---

**Document Version**: 2.0
**Last Updated**: 2026-02-16
**Base URL**: http://localhost:8000
