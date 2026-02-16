-- Trading AI Database Initialization
-- Creates necessary tables and extensions for TimescaleDB

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create trades table
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    strategy VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity DECIMAL(18, 8) NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    pnl DECIMAL(18, 8),
    fees DECIMAL(18, 8) DEFAULT 0,
    exchange VARCHAR(50),
    order_id VARCHAR(100),
    notes TEXT
);

-- Convert trades to hypertable (TimescaleDB)
SELECT create_hypertable('trades', 'timestamp', if_not_exists => TRUE);

-- Create index on timestamp for faster queries
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades (strategy);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol);

-- Create portfolio_snapshots table
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    total_value DECIMAL(18, 8) NOT NULL,
    cash_balance DECIMAL(18, 8) NOT NULL,
    positions_value DECIMAL(18, 8) NOT NULL,
    pnl_day DECIMAL(18, 8),
    pnl_total DECIMAL(18, 8),
    num_positions INTEGER DEFAULT 0
);

-- Convert portfolio_snapshots to hypertable
SELECT create_hypertable('portfolio_snapshots', 'timestamp', if_not_exists => TRUE);

-- Create positions table
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    avg_entry_price DECIMAL(18, 8) NOT NULL,
    current_price DECIMAL(18, 8),
    market_value DECIMAL(18, 8),
    pnl DECIMAL(18, 8),
    pnl_pct DECIMAL(10, 4),
    strategy VARCHAR(100),
    UNIQUE(symbol)
);

-- Create strategy_performance table
CREATE TABLE IF NOT EXISTS strategy_performance (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    strategy VARCHAR(100) NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(18, 8) DEFAULT 0,
    win_rate DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    avg_trade_duration_minutes INTEGER
);

-- Convert strategy_performance to hypertable
SELECT create_hypertable('strategy_performance', 'timestamp', if_not_exists => TRUE);

-- Create market_data table
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    volume DECIMAL(18, 8),
    bid DECIMAL(18, 8),
    ask DECIMAL(18, 8),
    high_24h DECIMAL(18, 8),
    low_24h DECIMAL(18, 8)
);

-- Convert market_data to hypertable
SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data (symbol, timestamp DESC);

-- Create agent_logs table
CREATE TABLE IF NOT EXISTS agent_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    agent_name VARCHAR(100) NOT NULL,
    log_level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB
);

-- Convert agent_logs to hypertable
SELECT create_hypertable('agent_logs', 'timestamp', if_not_exists => TRUE);

-- Create continuous aggregates for performance metrics
CREATE MATERIALIZED VIEW IF NOT EXISTS trades_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS hour,
    strategy,
    symbol,
    COUNT(*) as trade_count,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(pnl) as total_pnl,
    AVG(pnl) as avg_pnl,
    MAX(pnl) as max_pnl,
    MIN(pnl) as min_pnl
FROM trades
GROUP BY hour, strategy, symbol
WITH NO DATA;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('trades_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Create retention policy (keep raw data for 1 year, aggregates for 3 years)
SELECT add_retention_policy('trades', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('market_data', INTERVAL '6 months', if_not_exists => TRUE);
SELECT add_retention_policy('agent_logs', INTERVAL '3 months', if_not_exists => TRUE);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trader;

-- Insert initial test data (optional)
-- INSERT INTO trades (strategy, symbol, side, quantity, price, pnl)
-- VALUES ('momentum', 'BTC/USD', 'BUY', 0.1, 45000.00, 150.00);

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Trading AI database initialized successfully!';
    RAISE NOTICE 'Tables created: trades, portfolio_snapshots, positions, strategy_performance, market_data, agent_logs';
    RAISE NOTICE 'Continuous aggregates: trades_hourly';
    RAISE NOTICE 'Retention policies applied';
END $$;
