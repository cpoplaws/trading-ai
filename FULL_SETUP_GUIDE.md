# 🚀 Trading AI - Full Stack Setup Guide

## Quick Start (Automated)

The easiest way to get everything running:

```bash
cd /Users/silasmarkowicz/trading-ai-working
./setup-full-stack.sh
```

This script will:
- ✅ Check all prerequisites
- ✅ Create .env configuration
- ✅ Install Python dependencies
- ✅ Start all Docker services
- ✅ Initialize database
- ✅ Run tests
- ✅ Verify everything is working

**Time required**: 5-10 minutes

---

## Manual Setup (Step-by-Step)

If you prefer to do it manually or the script fails:

### 1. Install Python Dependencies

```bash
pip3 install -r requirements-full.txt
```

### 2. Configure Environment

Copy and edit `.env`:

```bash
cp .env.example .env
nano .env  # Edit with your API keys
```

**Required API keys**:
- Binance API (for crypto trading)
- Etherscan API (for blockchain data)
- Alchemy or Infura (for Web3)

### 3. Start Docker Services

```bash
# Start all core services
docker compose -f docker-compose.full.yml up -d

# Or start specific services
docker compose -f docker-compose.full.yml up -d timescaledb redis trading-api
```

### 4. Initialize Database

```bash
# Wait for database to be ready
docker exec trading_timescaledb pg_isready -U trading_user -d trading_db

# Run migrations (if using alembic)
alembic upgrade head
```

### 5. Verify Services

```bash
# Check running containers
docker ps

# Test API
curl http://localhost:8000/health

# View logs
docker compose -f docker-compose.full.yml logs -f
```

---

## Services Overview

### Core Services (Always Running)

| Service | Port | URL | Purpose |
|---------|------|-----|---------|
| **Trading API** | 8000 | http://localhost:8000 | FastAPI backend |
| **TimescaleDB** | 5432 | localhost:5432 | Time-series database |
| **Redis** | 6379 | localhost:6379 | Cache & pub/sub |
| **Prometheus** | 9090 | http://localhost:9090 | Metrics collection |
| **Grafana** | 3001 | http://localhost:3001 | Monitoring dashboards |

### Optional Services

Start with profiles:

```bash
# Start with web dashboard
docker compose -f docker-compose.full.yml --profile dashboard up -d

# Start with database tools
docker compose -f docker-compose.full.yml --profile tools up -d
```

| Service | Port | URL | Credentials |
|---------|------|-----|-------------|
| **Web Dashboard** | 3000 | http://localhost:3000 | API key required |
| **pgAdmin** | 5050 | http://localhost:5050 | admin@trading.ai / admin |

---

## Common Tasks

### View Logs

```bash
# All services
docker compose -f docker-compose.full.yml logs -f

# Specific service
docker compose -f docker-compose.full.yml logs -f trading-api

# Last 100 lines
docker compose -f docker-compose.full.yml logs --tail=100 trading-api
```

### Restart Services

```bash
# Restart all
docker compose -f docker-compose.full.yml restart

# Restart specific service
docker compose -f docker-compose.full.yml restart trading-api
```

### Stop Services

```bash
# Stop all
docker compose -f docker-compose.full.yml down

# Stop and remove volumes (⚠️ deletes data!)
docker compose -f docker-compose.full.yml down -v
```

### Access Containers

```bash
# API container
docker exec -it trading_api bash

# Database container
docker exec -it trading_timescaledb psql -U trading_user -d trading_db

# Redis container
docker exec -it trading_redis redis-cli
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# View coverage
open htmlcov/index.html
```

### Database Operations

```bash
# Connect to database
docker exec -it trading_timescaledb psql -U trading_user -d trading_db

# Backup database
docker exec trading_timescaledb pg_dump -U trading_user trading_db > backup.sql

# Restore database
cat backup.sql | docker exec -i trading_timescaledb psql -U trading_user -d trading_db
```

---

## Testing the System

### 1. Test API Endpoints

Visit http://localhost:8000/docs for interactive API documentation.

Or use curl:

```bash
# Health check
curl http://localhost:8000/health

# Get market data (requires API key)
curl -H "X-API-Key: sk_test_key" http://localhost:8000/api/v1/market/ticker/BTCUSDT
```

### 2. Test Database Connection

```python
from src.database.database_manager import DatabaseManager

db = DatabaseManager()
print("Database connection successful!")
```

### 3. Test Real-Time Data

```python
from src.realtime.binance_websocket import BinanceWebSocket

ws = BinanceWebSocket()
ws.connect()
print("WebSocket connected!")
```

### 4. Test Risk Management

```python
from src.risk_management.var_calculator import VaRCalculator
import numpy as np

calc = VaRCalculator()
returns = np.random.normal(0.001, 0.02, 252)
result = calc.calculate_var(returns, method='historical', portfolio_value=100000)

print(f"VaR (95%): ${result.var:,.2f}")
print(f"CVaR (95%): ${result.cvar:,.2f}")
```

---

## Monitoring

### Grafana Dashboards

1. Open http://localhost:3001
2. Login: admin / admin
3. Navigate to Dashboards
4. Import dashboards from `infrastructure/monitoring/grafana/dashboards/`

**Pre-configured dashboards**:
- Trading System Overview
- Risk Metrics
- API Performance
- Database Performance

### Prometheus Metrics

Visit http://localhost:9090 to query metrics:

```promql
# API request rate
rate(http_requests_total[5m])

# Database query time
histogram_quantile(0.95, rate(db_query_duration_bucket[5m]))

# VaR calculation time
histogram_quantile(0.99, rate(var_calculation_duration_bucket[5m]))
```

---

## Troubleshooting

### Port Already in Use

```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or change port in docker-compose.full.yml
```

### Database Connection Failed

```bash
# Check if database is running
docker ps | grep timescaledb

# Check database logs
docker logs trading_timescaledb

# Restart database
docker compose -f docker-compose.full.yml restart timescaledb
```

### API Not Starting

```bash
# Check API logs
docker logs trading_api

# Check if dependencies are installed
docker exec trading_api pip list

# Rebuild container
docker compose -f docker-compose.full.yml build trading-api
docker compose -f docker-compose.full.yml up -d trading-api
```

### Out of Memory

```bash
# Check Docker resources
docker stats

# Increase Docker memory limit in Docker Desktop settings
# Recommended: 8GB RAM, 4 CPUs
```

---

## Performance Tuning

### Database Optimization

```sql
-- Connect to database
docker exec -it trading_timescaledb psql -U trading_user -d trading_db

-- Enable query timing
\timing

-- Check table sizes
SELECT pg_size_pretty(pg_total_relation_size('ohlcv'));

-- Vacuum and analyze
VACUUM ANALYZE;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

### Redis Performance

```bash
# Connect to Redis
docker exec -it trading_redis redis-cli

# Check memory usage
INFO memory

# Check connected clients
CLIENT LIST

# Monitor commands in real-time
MONITOR
```

---

## Next Steps

After setup is complete:

1. **Configure API Keys**: Update `.env` with real exchange API keys
2. **Collect Data**: Run data collection scripts to populate database
3. **Train Models**: Train ML models with historical data
4. **Backtest**: Test strategies on historical data
5. **Paper Trade**: Start paper trading to validate system
6. **Monitor**: Watch Grafana dashboards for system health
7. **Go Live**: When confident, enable live trading

---

## Getting Help

- **Check logs**: `docker compose -f docker-compose.full.yml logs -f`
- **API docs**: http://localhost:8000/docs
- **Database**: http://localhost:5050
- **Monitoring**: http://localhost:3001

---

## Stopping the System

```bash
# Stop all services (data persists)
docker compose -f docker-compose.full.yml down

# Stop and remove all data (⚠️ destructive!)
docker compose -f docker-compose.full.yml down -v
```

---

**Status**: Setup complete! Your trading AI system is ready to use. 🚀
