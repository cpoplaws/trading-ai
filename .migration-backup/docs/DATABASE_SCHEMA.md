# Database Schema Documentation

**Last Updated**: 2026-02-16
**Version**: 1.0
**Database**: PostgreSQL with TimescaleDB extension
**ORM**: SQLAlchemy 2.0

## Table of Contents

1. [Overview](#overview)
2. [Entity-Relationship Diagram](#entity-relationship-diagram)
3. [Tables](#tables)
4. [Relationships](#relationships)
5. [Indexes](#indexes)
6. [Enumerations](#enumerations)
7. [Migrations](#migrations)
8. [Query Examples](#query-examples)

---

## Overview

The Trading AI system uses PostgreSQL with the TimescaleDB extension for time-series data. The schema consists of 10 main tables organized around users, portfolios, trading, and analytics.

### Database Statistics

| Metric | Value |
|--------|-------|
| Total Tables | 10 |
| Relationships | 15 |
| Indexes | 25 |
| Enumerations | 6 |

### Technology Stack

- **Database**: PostgreSQL 15+
- **Time-Series**: TimescaleDB 2.x
- **ORM**: SQLAlchemy 2.0+
- **Migrations**: Alembic
- **Connection Pool**: pgbouncer (recommended)

---

## Entity-Relationship Diagram

```
┌─────────────────┐
│     Users       │
├─────────────────┤
│ id (PK)         │
│ username        │◄────┐
│ email           │     │
│ password_hash   │     │
│ role            │     │
│ api_key         │     │
│ created_at      │     │
└─────────────────┘     │
         │              │
         │ 1:N          │
         ▼              │
┌─────────────────┐     │
│   Portfolios    │     │
├─────────────────┤     │
│ id (PK)         │     │
│ user_id (FK)    │─────┘
│ name            │
│ total_value_usd │
│ cash_balance    │
│ total_pnl       │
│ sharpe_ratio    │
│ is_paper        │
│ created_at      │
└─────────────────┘
         │
         │ 1:N
         ├─────────────────────┐
         │                     │
         ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│   Positions     │   │     Trades      │
├─────────────────┤   ├─────────────────┤
│ id (PK)         │   │ id (PK)         │
│ portfolio_id(FK)│   │ portfolio_id(FK)│
│ symbol          │   │ order_id (FK)   │
│ quantity        │   │ symbol          │
│ entry_price     │   │ side            │
│ current_price   │   │ quantity        │
│ unrealized_pnl  │   │ price           │
│ opened_at       │   │ value           │
└─────────────────┘   │ pnl             │
                      │ executed_at     │
                      └─────────────────┘
                               ▲
                               │ N:1
                               │
┌─────────────────┐            │
│     Orders      │────────────┘
├─────────────────┤
│ id (PK)         │
│ user_id (FK)    │─────────┐
│ portfolio_id(FK)│         │
│ strategy_id (FK)│         │
│ exchange_order  │         │
│ symbol          │         │
│ side            │         │
│ order_type      │         │
│ status          │         │
│ quantity        │         │
│ price           │         │
│ created_at      │         │
└─────────────────┘         │
         ▲                  │
         │                  │
         │ 1:N              │
         │                  │
┌─────────────────┐         │
│   Strategies    │         │
├─────────────────┤         │
│ id (PK)         │         │
│ user_id (FK)    │─────────┤
│ name            │         │
│ strategy_type   │         │
│ config (JSON)   │         │
│ is_active       │         │
│ total_pnl       │         │
│ win_rate        │         │
│ created_at      │         │
└─────────────────┘         │
                            │
┌─────────────────┐         │
│     Alerts      │         │
├─────────────────┤         │
│ id (PK)         │         │
│ user_id (FK)    │─────────┤
│ order_id (FK)   │         │
│ strategy_id (FK)│         │
│ title           │         │
│ message         │         │
│ severity        │         │
│ is_read         │         │
│ created_at      │         │
└─────────────────┘         │
                            │
┌─────────────────┐         │
│    API Keys     │         │
├─────────────────┤         │
│ id (PK)         │         │
│ user_id (FK)    │─────────┘
│ key             │
│ key_hash        │
│ permissions     │
│ rate_limit      │
│ is_active       │
│ created_at      │
│ expires_at      │
└─────────────────┘

┌──────────────────────────────────────────────┐
│         Time-Series Data (TimescaleDB)       │
├──────────────────────────────────────────────┤
│             Price Data (Hypertable)          │
├──────────────────────────────────────────────┤
│ id (PK)                                      │
│ symbol                                       │
│ exchange                                     │
│ timestamp                                    │
│ open, high, low, close, volume               │
│ vwap, num_trades                             │
└──────────────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│              ML Predictions                  │
├──────────────────────────────────────────────┤
│ id (PK)                                      │
│ model_name, model_version                   │
│ symbol                                       │
│ predicted_price, predicted_direction         │
│ confidence                                   │
│ feature_importance (JSON)                    │
│ predicted_at, prediction_horizon             │
│ actual_price, prediction_error, was_correct  │
└──────────────────────────────────────────────┘
```

---

## Tables

### 1. Users

**Purpose**: User accounts, authentication, and API credentials.

**Table**: `users`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTO_INCREMENT | Unique user identifier |
| username | VARCHAR(50) | UNIQUE, NOT NULL, INDEX | Login username |
| email | VARCHAR(100) | UNIQUE, NOT NULL, INDEX | Email address |
| password_hash | VARCHAR(255) | NOT NULL | Hashed password (bcrypt) |
| role | ENUM(UserRole) | NOT NULL, DEFAULT 'trader' | User role: admin, trader, viewer |
| api_key | VARCHAR(255) | UNIQUE | Exchange API key (encrypted) |
| api_secret | TEXT | | Exchange API secret (encrypted) |
| two_factor_enabled | BOOLEAN | DEFAULT FALSE | 2FA status |
| email_verified | BOOLEAN | DEFAULT FALSE | Email verification status |
| is_active | BOOLEAN | DEFAULT TRUE | Account active status |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW() | Account creation timestamp |
| updated_at | TIMESTAMP | DEFAULT NOW(), ON UPDATE NOW() | Last update timestamp |
| last_login | TIMESTAMP | | Last login timestamp |

**Relationships**:
- One-to-Many: portfolios, orders, strategies, alerts

**Indexes**:
- `idx_users_username` on `username`
- `idx_users_email` on `email`

**Example Row**:
```json
{
  "id": 1,
  "username": "trader_01",
  "email": "trader@example.com",
  "role": "trader",
  "two_factor_enabled": true,
  "email_verified": true,
  "is_active": true,
  "created_at": "2026-01-01T00:00:00Z"
}
```

---

### 2. Portfolios

**Purpose**: User portfolios with balances, positions, and performance metrics.

**Table**: `portfolios`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTO_INCREMENT | Unique portfolio identifier |
| user_id | INTEGER | FOREIGN KEY(users.id), NOT NULL | Owner user ID |
| name | VARCHAR(100) | NOT NULL | Portfolio name |
| total_value_usd | FLOAT | DEFAULT 0.0 | Total portfolio value (USD) |
| cash_balance_usd | FLOAT | DEFAULT 0.0 | Available cash balance |
| total_pnl | FLOAT | DEFAULT 0.0 | Total profit/loss |
| total_pnl_percent | FLOAT | DEFAULT 0.0 | Total PnL percentage |
| daily_pnl | FLOAT | DEFAULT 0.0 | Daily PnL |
| sharpe_ratio | FLOAT | | Risk-adjusted return metric |
| max_drawdown | FLOAT | | Maximum drawdown percentage |
| win_rate | FLOAT | | Percentage of profitable trades |
| is_paper | BOOLEAN | DEFAULT TRUE | Paper trading mode |
| is_active | BOOLEAN | DEFAULT TRUE | Active status |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW() | Creation timestamp |
| updated_at | TIMESTAMP | DEFAULT NOW(), ON UPDATE NOW() | Last update timestamp |

**Relationships**:
- Many-to-One: user
- One-to-Many: positions, trades

**Indexes**:
- `idx_portfolio_user` on `user_id`

**Example Row**:
```json
{
  "id": 1,
  "user_id": 1,
  "name": "Main Portfolio",
  "total_value_usd": 12500.50,
  "cash_balance_usd": 5000.00,
  "total_pnl": 2500.50,
  "total_pnl_percent": 25.01,
  "sharpe_ratio": 1.85,
  "max_drawdown": -12.5,
  "win_rate": 62.5,
  "is_paper": false,
  "is_active": true
}
```

---

### 3. Positions

**Purpose**: Current open positions in assets.

**Table**: `positions`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTO_INCREMENT | Unique position identifier |
| portfolio_id | INTEGER | FOREIGN KEY(portfolios.id), NOT NULL | Portfolio ID |
| symbol | VARCHAR(20) | NOT NULL | Asset symbol (e.g., BTC/USDT) |
| exchange | VARCHAR(50) | | Exchange name |
| quantity | FLOAT | NOT NULL | Position size |
| entry_price | FLOAT | NOT NULL | Average entry price |
| current_price | FLOAT | | Current market price |
| unrealized_pnl | FLOAT | DEFAULT 0.0 | Unrealized profit/loss |
| unrealized_pnl_percent | FLOAT | DEFAULT 0.0 | Unrealized PnL percentage |
| opened_at | TIMESTAMP | NOT NULL, DEFAULT NOW() | Position open timestamp |
| updated_at | TIMESTAMP | DEFAULT NOW(), ON UPDATE NOW() | Last update timestamp |

**Relationships**:
- Many-to-One: portfolio

**Indexes**:
- `idx_position_portfolio` on `portfolio_id`
- `idx_position_symbol` on `symbol`

**Example Row**:
```json
{
  "id": 1,
  "portfolio_id": 1,
  "symbol": "BTC/USDT",
  "exchange": "binance",
  "quantity": 0.5,
  "entry_price": 45000.00,
  "current_price": 47500.00,
  "unrealized_pnl": 1250.00,
  "unrealized_pnl_percent": 5.56,
  "opened_at": "2026-02-10T10:00:00Z"
}
```

---

### 4. Orders

**Purpose**: All orders (pending, filled, cancelled).

**Table**: `orders`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTO_INCREMENT | Unique order identifier |
| user_id | INTEGER | FOREIGN KEY(users.id), NOT NULL | User ID |
| portfolio_id | INTEGER | FOREIGN KEY(portfolios.id) | Portfolio ID |
| exchange_order_id | VARCHAR(100) | UNIQUE | Exchange-assigned order ID |
| client_order_id | VARCHAR(100) | UNIQUE | Client-assigned order ID |
| symbol | VARCHAR(20) | NOT NULL | Trading pair |
| exchange | VARCHAR(50) | NOT NULL | Exchange name |
| side | ENUM(OrderSide) | NOT NULL | Order side: buy, sell |
| order_type | ENUM(OrderType) | NOT NULL | Order type: market, limit, stop_loss, etc. |
| status | ENUM(OrderStatus) | NOT NULL, DEFAULT 'pending' | Order status |
| quantity | FLOAT | NOT NULL | Order quantity |
| filled_quantity | FLOAT | DEFAULT 0.0 | Filled quantity |
| price | FLOAT | | Limit price (null for market) |
| average_fill_price | FLOAT | | Average execution price |
| fee | FLOAT | DEFAULT 0.0 | Trading fee |
| fee_currency | VARCHAR(10) | | Fee currency |
| strategy_id | INTEGER | FOREIGN KEY(strategies.id) | Associated strategy |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW(), INDEX | Order creation time |
| updated_at | TIMESTAMP | DEFAULT NOW(), ON UPDATE NOW() | Last update time |
| filled_at | TIMESTAMP | | Fill timestamp |
| cancelled_at | TIMESTAMP | | Cancellation timestamp |

**Relationships**:
- Many-to-One: user, strategy
- One-to-Many: trades

**Indexes**:
- `idx_order_user` on `user_id`
- `idx_order_symbol` on `symbol`
- `idx_order_status` on `status`
- `idx_order_created` on `created_at`

**Example Row**:
```json
{
  "id": 1,
  "user_id": 1,
  "portfolio_id": 1,
  "exchange_order_id": "123456789",
  "symbol": "BTC/USDT",
  "exchange": "binance",
  "side": "buy",
  "order_type": "limit",
  "status": "filled",
  "quantity": 0.1,
  "filled_quantity": 0.1,
  "price": 46000.00,
  "average_fill_price": 46005.50,
  "fee": 4.60055,
  "fee_currency": "USDT",
  "created_at": "2026-02-15T10:00:00Z",
  "filled_at": "2026-02-15T10:00:15Z"
}
```

---

### 5. Trades

**Purpose**: Executed trades (fills).

**Table**: `trades`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTO_INCREMENT | Unique trade identifier |
| portfolio_id | INTEGER | FOREIGN KEY(portfolios.id), NOT NULL | Portfolio ID |
| order_id | INTEGER | FOREIGN KEY(orders.id) | Associated order |
| exchange_trade_id | VARCHAR(100) | UNIQUE | Exchange-assigned trade ID |
| symbol | VARCHAR(20) | NOT NULL | Trading pair |
| exchange | VARCHAR(50) | NOT NULL | Exchange name |
| side | ENUM(OrderSide) | NOT NULL | Trade side: buy, sell |
| quantity | FLOAT | NOT NULL | Trade quantity |
| price | FLOAT | NOT NULL | Execution price |
| value | FLOAT | NOT NULL | Trade value (price × quantity) |
| fee | FLOAT | DEFAULT 0.0 | Trading fee |
| fee_currency | VARCHAR(10) | | Fee currency |
| pnl | FLOAT | | Realized profit/loss (for closed) |
| pnl_percent | FLOAT | | PnL percentage |
| executed_at | TIMESTAMP | NOT NULL, DEFAULT NOW(), INDEX | Execution timestamp |

**Relationships**:
- Many-to-One: portfolio, order

**Indexes**:
- `idx_trade_portfolio` on `portfolio_id`
- `idx_trade_symbol` on `symbol`
- `idx_trade_executed` on `executed_at`

**Example Row**:
```json
{
  "id": 1,
  "portfolio_id": 1,
  "order_id": 1,
  "exchange_trade_id": "987654321",
  "symbol": "BTC/USDT",
  "exchange": "binance",
  "side": "buy",
  "quantity": 0.1,
  "price": 46005.50,
  "value": 4600.55,
  "fee": 4.60,
  "fee_currency": "USDT",
  "executed_at": "2026-02-15T10:00:15Z"
}
```

---

### 6. PriceData (TimescaleDB Hypertable)

**Purpose**: Historical OHLCV price data (time-series).

**Table**: `price_data`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTO_INCREMENT | Unique identifier |
| symbol | VARCHAR(20) | NOT NULL | Trading pair |
| exchange | VARCHAR(50) | NOT NULL | Exchange name |
| timestamp | TIMESTAMP | NOT NULL, INDEX | Candle timestamp |
| open | FLOAT | NOT NULL | Opening price |
| high | FLOAT | NOT NULL | Highest price |
| low | FLOAT | NOT NULL | Lowest price |
| close | FLOAT | NOT NULL | Closing price |
| volume | FLOAT | NOT NULL | Trading volume |
| vwap | FLOAT | | Volume-weighted average price |
| num_trades | INTEGER | | Number of trades |

**Indexes**:
- `idx_price_symbol_time` on `(symbol, timestamp)`
- `idx_price_exchange_time` on `(exchange, timestamp)`

**TimescaleDB Configuration**:
```sql
SELECT create_hypertable('price_data', 'timestamp',
    chunk_time_interval => INTERVAL '1 day');

-- Retention policy: Keep 2 years of data
SELECT add_retention_policy('price_data', INTERVAL '2 years');

-- Compression: Compress data older than 7 days
SELECT add_compression_policy('price_data', INTERVAL '7 days');
```

**Example Row**:
```json
{
  "symbol": "BTC/USDT",
  "exchange": "binance",
  "timestamp": "2026-02-15T10:00:00Z",
  "open": 46000.00,
  "high": 46150.00,
  "low": 45900.00,
  "close": 46050.00,
  "volume": 125.5,
  "vwap": 46025.00,
  "num_trades": 1542
}
```

---

### 7. MLPredictions

**Purpose**: Machine learning model predictions and validation.

**Table**: `ml_predictions`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTO_INCREMENT | Unique prediction identifier |
| model_name | VARCHAR(100) | NOT NULL | ML model name |
| model_version | VARCHAR(50) | | Model version |
| symbol | VARCHAR(20) | NOT NULL | Predicted asset |
| predicted_price | FLOAT | | Predicted price |
| predicted_direction | VARCHAR(10) | | Direction: 'up', 'down', 'neutral' |
| confidence | FLOAT | | Prediction confidence (0-1) |
| feature_importance | JSON | | Feature importance scores |
| predicted_at | TIMESTAMP | NOT NULL, DEFAULT NOW(), INDEX | Prediction timestamp |
| prediction_horizon | INTEGER | | Minutes ahead |
| actual_price | FLOAT | | Actual price (backfilled) |
| prediction_error | FLOAT | | Absolute error |
| was_correct | BOOLEAN | | Direction correctness |

**Indexes**:
- `idx_prediction_symbol_time` on `(symbol, predicted_at)`
- `idx_prediction_model` on `model_name`

**Example Row**:
```json
{
  "id": 1,
  "model_name": "lstm_v2",
  "model_version": "2.1.0",
  "symbol": "BTC/USDT",
  "predicted_price": 46500.00,
  "predicted_direction": "up",
  "confidence": 0.78,
  "feature_importance": {
    "rsi_14": 0.25,
    "macd": 0.20,
    "volume_ratio": 0.15
  },
  "predicted_at": "2026-02-15T10:00:00Z",
  "prediction_horizon": 60,
  "actual_price": 46450.00,
  "prediction_error": 50.00,
  "was_correct": true
}
```

---

### 8. Strategies

**Purpose**: Strategy configurations and performance.

**Table**: `strategies`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTO_INCREMENT | Unique strategy identifier |
| user_id | INTEGER | FOREIGN KEY(users.id), NOT NULL | Owner user ID |
| name | VARCHAR(100) | NOT NULL | Strategy name |
| strategy_type | ENUM(StrategyType) | NOT NULL | Strategy type enum |
| description | TEXT | | Strategy description |
| config | JSON | NOT NULL | Strategy configuration |
| is_active | BOOLEAN | DEFAULT TRUE | Active status |
| is_backtest | BOOLEAN | DEFAULT FALSE | Backtest mode |
| total_pnl | FLOAT | DEFAULT 0.0 | Total profit/loss |
| win_rate | FLOAT | | Win rate percentage |
| sharpe_ratio | FLOAT | | Sharpe ratio |
| num_trades | INTEGER | DEFAULT 0 | Total number of trades |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW() | Creation timestamp |
| updated_at | TIMESTAMP | DEFAULT NOW(), ON UPDATE NOW() | Last update |
| last_signal_at | TIMESTAMP | | Last signal generated |

**Relationships**:
- Many-to-One: user
- One-to-Many: orders

**Indexes**:
- `idx_strategy_user` on `user_id`
- `idx_strategy_type` on `strategy_type`

**Example Row**:
```json
{
  "id": 1,
  "user_id": 1,
  "name": "BTC Momentum Strategy",
  "strategy_type": "momentum",
  "config": {
    "symbol": "BTC/USDT",
    "lookback_period": 14,
    "rsi_period": 14,
    "momentum_threshold": 0.05
  },
  "is_active": true,
  "is_backtest": false,
  "total_pnl": 1250.50,
  "win_rate": 65.5,
  "sharpe_ratio": 1.85,
  "num_trades": 47,
  "created_at": "2026-01-01T00:00:00Z"
}
```

---

### 9. Alerts

**Purpose**: System alerts and notifications.

**Table**: `alerts`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTO_INCREMENT | Unique alert identifier |
| user_id | INTEGER | FOREIGN KEY(users.id) | User ID |
| title | VARCHAR(200) | NOT NULL | Alert title |
| message | TEXT | NOT NULL | Alert message |
| severity | ENUM(AlertSeverity) | NOT NULL, DEFAULT 'info' | Severity level |
| category | VARCHAR(50) | | Category: price, trade, system, ml |
| symbol | VARCHAR(20) | | Related symbol |
| order_id | INTEGER | FOREIGN KEY(orders.id) | Related order |
| strategy_id | INTEGER | FOREIGN KEY(strategies.id) | Related strategy |
| is_read | BOOLEAN | DEFAULT FALSE | Read status |
| is_sent | BOOLEAN | DEFAULT FALSE | Notification sent status |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW(), INDEX | Creation timestamp |

**Relationships**:
- Many-to-One: user

**Indexes**:
- `idx_alert_user_created` on `(user_id, created_at)`
- `idx_alert_severity` on `severity`

**Example Row**:
```json
{
  "id": 1,
  "user_id": 1,
  "title": "Price Alert Triggered",
  "message": "BTC/USDT reached target price of $46,000",
  "severity": "info",
  "category": "price",
  "symbol": "BTC/USDT",
  "is_read": false,
  "is_sent": true,
  "created_at": "2026-02-15T10:00:00Z"
}
```

---

### 10. APIKeys

**Purpose**: API keys for programmatic access.

**Table**: `api_keys`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTO_INCREMENT | Unique key identifier |
| user_id | INTEGER | FOREIGN KEY(users.id), NOT NULL | Owner user ID |
| key | VARCHAR(100) | UNIQUE, NOT NULL, INDEX | API key |
| key_hash | VARCHAR(255) | NOT NULL | Hashed key for verification |
| name | VARCHAR(100) | | Key name/label |
| description | TEXT | | Key description |
| permissions | JSON | | Allowed operations list |
| rate_limit_per_minute | INTEGER | DEFAULT 60 | Rate limit |
| is_active | BOOLEAN | DEFAULT TRUE | Active status |
| last_used_at | TIMESTAMP | | Last usage timestamp |
| total_requests | INTEGER | DEFAULT 0 | Total request count |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW() | Creation timestamp |
| expires_at | TIMESTAMP | | Expiration timestamp |

**Example Row**:
```json
{
  "id": 1,
  "user_id": 1,
  "key": "ak_prod_abc123...",
  "name": "Production API Key",
  "permissions": ["read:portfolio", "write:orders"],
  "rate_limit_per_minute": 100,
  "is_active": true,
  "total_requests": 15420,
  "created_at": "2026-01-01T00:00:00Z",
  "expires_at": "2027-01-01T00:00:00Z"
}
```

---

## Relationships

### User Relationships

```
User (1) ──< (N) Portfolios
User (1) ──< (N) Orders
User (1) ──< (N) Strategies
User (1) ──< (N) Alerts
User (1) ──< (N) API Keys
```

### Portfolio Relationships

```
Portfolio (1) ──< (N) Positions
Portfolio (1) ──< (N) Trades
```

### Order Relationships

```
Order (1) ──< (N) Trades
Order (N) ──> (1) Strategy
```

### Complete Relationship Graph

```
User
 ├── Portfolios
 │    ├── Positions
 │    └── Trades
 ├── Orders
 │    └── Trades
 ├── Strategies
 │    └── Orders
 ├── Alerts
 └── API Keys

Independent Tables:
 ├── PriceData (time-series)
 └── MLPredictions
```

---

## Indexes

### Primary Indexes (Automatic)

All tables have primary key indexes on `id` column.

### Foreign Key Indexes

```sql
-- Portfolio relationships
CREATE INDEX idx_portfolio_user ON portfolios(user_id);
CREATE INDEX idx_position_portfolio ON positions(portfolio_id);
CREATE INDEX idx_trade_portfolio ON trades(portfolio_id);

-- Order relationships
CREATE INDEX idx_order_user ON orders(user_id);
CREATE INDEX idx_order_portfolio ON orders(portfolio_id);

-- Strategy relationships
CREATE INDEX idx_strategy_user ON strategies(user_id);
```

### Unique Indexes

```sql
-- User authentication
CREATE UNIQUE INDEX idx_users_username ON users(username);
CREATE UNIQUE INDEX idx_users_email ON users(email);

-- Order tracking
CREATE UNIQUE INDEX idx_orders_exchange_id ON orders(exchange_order_id);
CREATE UNIQUE INDEX idx_orders_client_id ON orders(client_order_id);

-- Trade tracking
CREATE UNIQUE INDEX idx_trades_exchange_id ON trades(exchange_trade_id);
```

### Query Optimization Indexes

```sql
-- Order queries
CREATE INDEX idx_order_symbol ON orders(symbol);
CREATE INDEX idx_order_status ON orders(status);
CREATE INDEX idx_order_created ON orders(created_at);

-- Position queries
CREATE INDEX idx_position_symbol ON positions(symbol);

-- Trade queries
CREATE INDEX idx_trade_symbol ON trades(symbol);
CREATE INDEX idx_trade_executed ON trades(executed_at);

-- Alert queries
CREATE INDEX idx_alert_user_created ON alerts(user_id, created_at);
CREATE INDEX idx_alert_severity ON alerts(severity);

-- Strategy queries
CREATE INDEX idx_strategy_type ON strategies(strategy_type);

-- Time-series queries
CREATE INDEX idx_price_symbol_time ON price_data(symbol, timestamp);
CREATE INDEX idx_price_exchange_time ON price_data(exchange, timestamp);

-- ML prediction queries
CREATE INDEX idx_prediction_symbol_time ON ml_predictions(symbol, predicted_at);
CREATE INDEX idx_prediction_model ON ml_predictions(model_name);
```

---

## Enumerations

### UserRole

```python
class UserRole(str, Enum):
    ADMIN = "admin"      # Full system access
    TRADER = "trader"    # Trading operations
    VIEWER = "viewer"    # Read-only access
```

### OrderStatus

```python
class OrderStatus(str, Enum):
    PENDING = "pending"              # Created, not submitted
    OPEN = "open"                    # Submitted to exchange
    FILLED = "filled"                # Completely filled
    PARTIALLY_FILLED = "partially_filled"  # Partially filled
    CANCELLED = "cancelled"          # Cancelled
    REJECTED = "rejected"            # Rejected by exchange
```

### OrderType

```python
class OrderType(str, Enum):
    MARKET = "market"          # Market order
    LIMIT = "limit"            # Limit order
    STOP_LOSS = "stop_loss"    # Stop-loss order
    STOP_LIMIT = "stop_limit"  # Stop-limit order
    TWAP = "twap"              # Time-weighted average price
    VWAP = "vwap"              # Volume-weighted average price
```

### OrderSide

```python
class OrderSide(str, Enum):
    BUY = "buy"      # Buy order
    SELL = "sell"    # Sell order
```

### StrategyType

```python
class StrategyType(str, Enum):
    GRID_TRADING = "grid_trading"
    DCA = "dca"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MARKET_MAKING = "market_making"
    WHALE_FOLLOWING = "whale_following"
    LIQUIDATION_HUNTING = "liquidation_hunting"
    YIELD_OPTIMIZATION = "yield_optimization"
```

### AlertSeverity

```python
class AlertSeverity(str, Enum):
    INFO = "info"          # Informational
    WARNING = "warning"    # Warning
    ERROR = "error"        # Error
    CRITICAL = "critical"  # Critical issue
```

---

## Migrations

### Using Alembic

```bash
# Initialize Alembic
alembic init alembic

# Create a new migration
alembic revision --autogenerate -m "Add new column"

# Apply migrations
alembic upgrade head

# Rollback one version
alembic downgrade -1

# View migration history
alembic history
```

### Example Migration

```python
"""Add sharpe_ratio to portfolios

Revision ID: abc123def456
Revises: previous_revision
Create Date: 2026-02-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'abc123def456'
down_revision = 'previous_revision'

def upgrade():
    op.add_column('portfolios',
        sa.Column('sharpe_ratio', sa.Float(), nullable=True)
    )

def downgrade():
    op.drop_column('portfolios', 'sharpe_ratio')
```

---

## Query Examples

### User and Portfolio Queries

```python
from sqlalchemy.orm import Session
from src.database.models import User, Portfolio, Position

# Get user with all portfolios
user = session.query(User)\
    .filter(User.username == "trader_01")\
    .first()

portfolios = user.portfolios

# Get portfolio with positions
portfolio = session.query(Portfolio)\
    .filter(Portfolio.id == 1)\
    .first()

positions = portfolio.positions
total_value = sum(p.quantity * p.current_price for p in positions)

# Get user's total PnL across all portfolios
total_pnl = session.query(func.sum(Portfolio.total_pnl))\
    .filter(Portfolio.user_id == user_id)\
    .scalar()
```

### Order and Trade Queries

```python
from src.database.models import Order, Trade, OrderStatus

# Get all open orders for a user
open_orders = session.query(Order)\
    .filter(
        Order.user_id == user_id,
        Order.status.in_([OrderStatus.PENDING, OrderStatus.OPEN])
    )\
    .all()

# Get filled orders with trades
filled_orders = session.query(Order)\
    .join(Trade)\
    .filter(Order.status == OrderStatus.FILLED)\
    .all()

# Get trade history for a portfolio
trades = session.query(Trade)\
    .filter(Trade.portfolio_id == portfolio_id)\
    .order_by(Trade.executed_at.desc())\
    .limit(100)\
    .all()

# Calculate realized PnL
realized_pnl = session.query(func.sum(Trade.pnl))\
    .filter(Trade.portfolio_id == portfolio_id)\
    .scalar()
```

### Price Data Queries (TimescaleDB)

```python
from src.database.models import PriceData

# Get latest price for a symbol
latest_price = session.query(PriceData)\
    .filter(PriceData.symbol == "BTC/USDT")\
    .order_by(PriceData.timestamp.desc())\
    .first()

# Get OHLCV data for last 24 hours
from datetime import datetime, timedelta

start_time = datetime.utcnow() - timedelta(hours=24)
candles = session.query(PriceData)\
    .filter(
        PriceData.symbol == "BTC/USDT",
        PriceData.timestamp >= start_time
    )\
    .order_by(PriceData.timestamp.asc())\
    .all()

# Time-series aggregation (hourly average)
hourly_avg = session.query(
    func.time_bucket('1 hour', PriceData.timestamp).label('hour'),
    func.avg(PriceData.close).label('avg_close'),
    func.sum(PriceData.volume).label('total_volume')
)\
    .filter(PriceData.symbol == "BTC/USDT")\
    .group_by('hour')\
    .order_by('hour')\
    .all()
```

### ML Prediction Queries

```python
from src.database.models import MLPrediction

# Get latest predictions for a model
predictions = session.query(MLPrediction)\
    .filter(MLPrediction.model_name == "lstm_v2")\
    .order_by(MLPrediction.predicted_at.desc())\
    .limit(10)\
    .all()

# Calculate model accuracy
model_accuracy = session.query(
    func.avg(case((MLPrediction.was_correct == True, 1.0), else_=0.0))
)\
    .filter(
        MLPrediction.model_name == "lstm_v2",
        MLPrediction.was_correct.isnot(None)
    )\
    .scalar()

# Get prediction errors
avg_error = session.query(func.avg(MLPrediction.prediction_error))\
    .filter(MLPrediction.model_name == "lstm_v2")\
    .scalar()
```

### Strategy Performance Queries

```python
from src.database.models import Strategy, StrategyType

# Get all active strategies
active_strategies = session.query(Strategy)\
    .filter(Strategy.is_active == True)\
    .all()

# Get top performing strategies
top_strategies = session.query(Strategy)\
    .filter(Strategy.num_trades >= 10)\
    .order_by(Strategy.sharpe_ratio.desc())\
    .limit(5)\
    .all()

# Get strategy performance by type
strategy_stats = session.query(
    Strategy.strategy_type,
    func.avg(Strategy.total_pnl).label('avg_pnl'),
    func.avg(Strategy.win_rate).label('avg_win_rate'),
    func.count(Strategy.id).label('count')
)\
    .group_by(Strategy.strategy_type)\
    .all()
```

---

## Database Maintenance

### Backup

```bash
# Full database backup
pg_dump -U postgres trading_ai > backup_$(date +%Y%m%d).sql

# Backup specific table
pg_dump -U postgres -t price_data trading_ai > price_data_backup.sql

# Compressed backup
pg_dump -U postgres trading_ai | gzip > backup_$(date +%Y%m%d).sql.gz
```

### Restore

```bash
# Restore from backup
psql -U postgres trading_ai < backup_20260216.sql

# Restore compressed backup
gunzip -c backup_20260216.sql.gz | psql -U postgres trading_ai
```

### Vacuum and Analyze

```sql
-- Vacuum tables
VACUUM ANALYZE users;
VACUUM ANALYZE portfolios;
VACUUM ANALYZE orders;
VACUUM ANALYZE trades;

-- Full vacuum (reclaim space)
VACUUM FULL price_data;
```

---

## Performance Optimization

### Connection Pooling

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    "postgresql://user:pass@localhost/trading_ai",
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)
```

### Query Optimization Tips

1. **Use Indexes**: Ensure frequently queried columns are indexed
2. **Limit Results**: Always use `.limit()` for large result sets
3. **Eager Loading**: Use `.joinedload()` to prevent N+1 queries
4. **Batch Operations**: Use `bulk_insert_mappings()` for large inserts
5. **Connection Pooling**: Reuse database connections

---

## Conclusion

This database schema provides a robust foundation for the Trading AI system. For more information:

- **ORM Models**: `src/database/models.py`
- **Migrations**: `alembic/versions/`
- **API Documentation**: `docs/API_REFERENCE.md`
- **Architecture**: `docs/ARCHITECTURE.md`

---

**Document Version**: 1.0
**Last Updated**: 2026-02-16
**Maintainer**: Trading AI Team
