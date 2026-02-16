# Troubleshooting Guide

**Version**: 1.0
**Last Updated**: February 16, 2026

## Quick Reference

| Issue | Likely Cause | Quick Fix |
|-------|--------------|-----------|
| Agent won't start | Missing API keys | Check `.env` file |
| Database connection failed | PostgreSQL not running | `docker-compose up -d database` |
| High memory usage | Memory leak | Restart agent, check logs |
| Trades not executing | Rate limit | Wait 1 minute, check limits |
| Dashboard blank | API not running | Start API: `uvicorn src.api.main:app` |
| WebSocket disconnected | Network issue | Auto-reconnects in 30s |

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Configuration Problems](#configuration-problems)
3. [Agent Issues](#agent-issues)
4. [API Problems](#api-problems)
5. [Database Issues](#database-issues)
6. [Exchange Connection Problems](#exchange-connection-problems)
7. [Performance Issues](#performance-issues)
8. [Security & Authentication](#security--authentication)
9. [Docker & Deployment](#docker--deployment)
10. [Logging & Monitoring](#logging--monitoring)

---

## Installation Issues

### Problem: `pip install` fails with compilation errors

**Symptoms**:
```
ERROR: Could not build wheels for TA-Lib
```

**Cause**: TA-Lib requires C library to be installed first

**Solution**:
```bash
# macOS
brew install ta-lib

# Ubuntu/Debian
sudo apt-get install ta-lib

# Then retry
pip install TA-Lib
```

---

### Problem: TensorFlow won't install

**Symptoms**:
```
ERROR: Could not find a version that satisfies the requirement tensorflow
```

**Cause**: Incompatible Python version or platform

**Solution**:
```bash
# Check Python version (need 3.9-3.12)
python --version

# For Apple Silicon Macs
pip install tensorflow-macos

# For GPU support (NVIDIA)
pip install tensorflow[and-cuda]

# Standard installation
pip install tensorflow>=2.20.0
```

---

### Problem: Module not found errors

**Symptoms**:
```python
ModuleNotFoundError: No module named 'src'
```

**Cause**: Python path not configured

**Solution**:
```bash
# Option 1: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/trading-ai"

# Option 2: Install in development mode
pip install -e .

# Option 3: Run from project root
cd /path/to/trading-ai
python -m src.autonomous_agent.trading_agent
```

---

## Configuration Problems

### Problem: Environment variables not loading

**Symptoms**:
```
KeyError: 'BINANCE_API_KEY'
```

**Cause**: `.env` file not found or not loaded

**Solution**:
```bash
# 1. Check .env file exists
ls -la .env

# 2. Copy from template if missing
cp .env.example .env

# 3. Edit with your API keys
nano .env

# 4. Verify loading in Python
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('BINANCE_API_KEY'))"
```

---

### Problem: Invalid API credentials

**Symptoms**:
```
APIError: Signature verification failed
```

**Cause**: Wrong API keys or time sync issue

**Solution**:
```bash
# 1. Check API keys are correct
cat .env | grep BINANCE

# 2. Verify no extra spaces
# BAD:  BINANCE_API_KEY= abc123
# GOOD: BINANCE_API_KEY=abc123

# 3. Check system time sync
date
sudo ntpdate pool.ntp.org  # Sync time

# 4. Test connectivity
python -m src.exchanges.binance_trading_client
```

---

##  Agent Issues

### Problem: Agent keeps crashing

**Symptoms**:
Agent starts but stops after a few minutes

**Diagnosis**:
```bash
# Check logs
tail -f logs/trading_agent.log

# Check for memory issues
top -p $(pgrep -f trading_agent)

# Check disk space
df -h
```

**Common Causes & Fixes**:

1. **Out of Memory**
   ```bash
   # Check memory usage
   free -h

   # Reduce batch size in config
   # config/agent.yml
   strategies:
     ml_prediction:
       batch_size: 32  # Reduce this
   ```

2. **Unhandled Exception**
   ```python
   # Check error recovery is working
   from utils.retry import retry
   from utils.dead_letter_queue import DeadLetterQueue

   # Review dead letter queue
   dlq = DeadLetterQueue()
   failed_ops = dlq.list()
   for op in failed_ops:
       print(f"{op['operation_type']}: {op['error']}")
   ```

3. **Database Connection Lost**
   ```bash
   # Check database is running
   docker ps | grep postgres

   # Restart database
   docker-compose restart database

   # Check connection pooling
   # src/database/models.py
   # engine = create_engine(..., pool_pre_ping=True)
   ```

---

### Problem: Agent not executing trades

**Symptoms**:
Agent running but no trades in database

**Diagnosis**:
```bash
# Check agent state
curl http://localhost:8000/api/v1/agents/agent-001/status

# Check strategy signals
python -c "from src.crypto_strategies.dca_bot import DCABot; bot = DCABot(); print(bot.generate_signal('BTCUSDT'))"

# Check risk limits
psql -U trading_user -d trading_db -c "SELECT * FROM agent_config WHERE agent_id='agent-001';"
```

**Common Causes**:

1. **Risk Limits Exceeded**
   - Max daily loss reached
   - Position size too large
   - Insufficient balance

   **Fix**: Adjust limits in agent config

2. **No Trading Signals**
   - Market conditions not met
   - Strategy parameters too strict

   **Fix**: Review strategy thresholds

3. **Paper Trading Mode**
   - Agent configured for paper trading only

   **Fix**: Set `paper_trading=False` in config

---

### Problem: Agent state recovery fails

**Symptoms**:
```
Error: Failed to load agent state
```

**Cause**: Corrupted state file

**Solution**:
```bash
# Check state file
cat /tmp/agent_state/agent-001.json

# Try backup restore
python -c "
from utils.state_recovery import StateRecoveryManager
manager = StateRecoveryManager('agent-001')
state = manager._restore_from_backup()
print(state)
"

# If all backups corrupt, delete state
rm -rf /tmp/agent_state/agent-001*

# Agent will start fresh
```

---

## API Problems

### Problem: API won't start

**Symptoms**:
```
Address already in use: 0.0.0.0:8000
```

**Cause**: Port 8000 already occupied

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 $(lsof -t -i :8000)

# Or use different port
uvicorn src.api.main:app --port 8001
```

---

### Problem: API returning 500 errors

**Symptoms**:
All API requests return Internal Server Error

**Diagnosis**:
```bash
# Check API logs
docker logs trading-api

# Test directly
curl -v http://localhost:8000/health/

# Check database connection
curl http://localhost:8000/health/database
```

**Common Causes**:

1. **Database Not Reachable**
   ```bash
   # Check connection string
   echo $DATABASE_URL

   # Test connection
   psql $DATABASE_URL -c "SELECT 1;"
   ```

2. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install -r requirements-secure.txt
   ```

3. **Configuration Error**
   ```bash
   # Check config file
   cat config/api.yml

   # Validate syntax
   python -c "import yaml; yaml.safe_load(open('config/api.yml'))"
   ```

---

### Problem: Slow API response

**Symptoms**:
API requests taking > 5 seconds

**Diagnosis**:
```bash
# Check response time
time curl http://localhost:8000/api/v1/agents/

# Profile endpoint
python -m cProfile -s cumtime src/api/main.py
```

**Solutions**:

1. **Enable Caching**
   ```python
   # Ensure Redis is running
   docker-compose up -d redis

   # Check cache hits
   redis-cli INFO stats | grep hits
   ```

2. **Database Query Optimization**
   ```sql
   -- Check slow queries
   SELECT query, mean_exec_time, calls
   FROM pg_stat_statements
   ORDER BY mean_exec_time DESC
   LIMIT 10;

   -- Add missing indexes
   CREATE INDEX idx_trades_agent_id ON trades(agent_id);
   CREATE INDEX idx_trades_timestamp ON trades(timestamp);
   ```

3. **Connection Pooling**
   ```python
   # src/database/models.py
   engine = create_engine(
       DATABASE_URL,
       poolclass=QueuePool,
       pool_size=10,      # Increase
       max_overflow=20,   # Increase
       pool_pre_ping=True
   )
   ```

---

## Database Issues

### Problem: Cannot connect to database

**Symptoms**:
```
OperationalError: could not connect to server
```

**Solution**:
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Start database
docker-compose up -d database

# Check logs
docker logs trading-database

# Test connection
psql -h localhost -U trading_user -d trading_db -c "SELECT version();"
```

---

### Problem: Database migrations fail

**Symptoms**:
```
alembic.util.exc.CommandError: Target database is not up to date
```

**Solution**:
```bash
# Check current version
alembic current

# Check pending migrations
alembic heads

# Run migrations
alembic upgrade head

# If errors, check migration files
ls -la alembic/versions/

# Rollback if needed
alembic downgrade -1
```

---

### Problem: Database is full

**Symptoms**:
```
ERROR: could not extend file: No space left on device
```

**Solution**:
```bash
# Check disk space
df -h

# Check database size
psql -U trading_user -d trading_db -c "
SELECT pg_size_pretty(pg_database_size('trading_db'));
"

# Clean old data
psql -U trading_user -d trading_db -c "
DELETE FROM trades WHERE timestamp < NOW() - INTERVAL '90 days';
VACUUM FULL trades;
"

# Set up automated cleanup
# Add to cron:
# 0 2 * * * psql -U trading_user -d trading_db -c "DELETE FROM trades WHERE timestamp < NOW() - INTERVAL '90 days';"
```

---

## Exchange Connection Problems

### Problem: Binance API rate limit exceeded

**Symptoms**:
```
APIError: Rate limit exceeded. Retry after 60 seconds.
```

**Solution**:
```python
# Check current rate limit usage
from src.exchanges.binance_trading_client import BinanceTradingClient

client = BinanceTradingClient()
print(f"Requests: {client.request_count}/{client.RATE_LIMIT}")

# Increase delays between requests
# config/agent.yml
check_interval_seconds: 60  # Increase from 30

# Use WebSocket for market data (doesn't count toward REST limit)
# Enable in config:
use_websocket: true
```

---

### Problem: WebSocket keeps disconnecting

**Symptoms**:
```
WARNING: WebSocket disconnected. Reconnecting...
```

**Cause**: Network instability or exchange maintenance

**Solution**:
```bash
# Check network connectivity
ping api.binance.com

# Check exchange status
curl https://api.binance.com/api/v3/ping

# WebSocket will auto-reconnect
# Verify in logs:
grep "WebSocket" logs/trading_agent.log | tail -20

# If persistent, use REST fallback
# config/agent.yml
websocket:
  enabled: false  # Temporarily disable
```

---

### Problem: Orders rejected by exchange

**Symptoms**:
```
Order failed: Insufficient balance
```

**Diagnosis**:
```python
from src.exchanges.binance_trading_client import BinanceTradingClient

client = BinanceTradingClient(testnet=True)

# Check balance
balance = client.get_balance('USDT')
print(f"Available: ${balance['free']}")

# Check minimum order size
symbol_info = client.get_symbol_info('BTCUSDT')
print(f"Min order: {symbol_info['filters']}")

# Check account status
account = client.get_account_info()
print(f"Can trade: {account['canTrade']}")
```

**Solutions**:
1. Add funds to exchange account
2. Reduce position size in agent config
3. Check exchange minimum order requirements
4. Verify API permissions enabled

---

## Performance Issues

### Problem: High CPU usage

**Symptoms**:
CPU constantly at 100%

**Diagnosis**:
```bash
# Identify hot functions
python -m cProfile -o profile.stats -m src.autonomous_agent.trading_agent

# Analyze
python -m pstats profile.stats
>>> sort cumtime
>>> stats 20

# Check for infinite loops
strace -p $(pgrep -f trading_agent)
```

**Solutions**:

1. **Optimize Data Processing**
   ```python
   # Use vectorized operations
   import numpy as np
   import pandas as pd

   # BAD: Loop over rows
   for row in df.iterrows():
       result = calculate(row)

   # GOOD: Vectorized
   df['result'] = df.apply(calculate, axis=1)
   # or
   result = np.array([calculate(x) for x in df.values])
   ```

2. **Reduce Polling Frequency**
   ```yaml
   # config/agent.yml
   check_interval_seconds: 60  # Increase from 10
   ```

3. **Use Async Operations**
   ```python
   # Use async for I/O bound operations
   import asyncio

   async def fetch_multiple_prices(symbols):
       tasks = [fetch_price(symbol) for symbol in symbols]
       return await asyncio.gather(*tasks)
   ```

---

### Problem: High memory usage

**Symptoms**:
Memory usage growing over time (memory leak)

**Diagnosis**:
```bash
# Monitor memory
python -m memory_profiler src/autonomous_agent/trading_agent.py

# Check for leaks
import objgraph
objgraph.show_most_common_types(limit=20)

# Track objects
import gc
gc.collect()
print(f"Objects: {len(gc.get_objects())}")
```

**Solutions**:

1. **Clear Large DataFrames**
   ```python
   # Limit DataFrame size
   df = df.tail(1000)  # Keep only recent data

   # Delete when done
   del df
   gc.collect()
   ```

2. **Close Database Connections**
   ```python
   # Use context managers
   with session_scope() as session:
       # Operations here
       pass
   # Connection automatically closed
   ```

3. **Limit Cache Size**
   ```python
   # Configure Redis max memory
   # redis.conf
   maxmemory 2gb
   maxmemory-policy allkeys-lru
   ```

---

## Security & Authentication

### Problem: JWT token expired

**Symptoms**:
```
401 Unauthorized: Token expired
```

**Solution**:
```bash
# Get new token
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"user","password":"pass"}'

# Increase token expiration
# config/api.yml
jwt:
  expiration_minutes: 1440  # 24 hours
```

---

### Problem: API key not working

**Symptoms**:
```
403 Forbidden: Invalid API key
```

**Solution**:
```bash
# Check API key format
echo $API_KEY | wc -c  # Should be 32+ characters

# Regenerate API key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Update in database
psql -U trading_user -d trading_db -c "
UPDATE users SET api_key='NEW_KEY' WHERE username='user';
"

# Test with new key
curl -H "X-API-Key: NEW_KEY" http://localhost:8000/api/v1/agents/
```

---

## Docker & Deployment

### Problem: Docker container won't start

**Symptoms**:
```
Error: Container exits immediately
```

**Diagnosis**:
```bash
# Check logs
docker logs trading-agent

# Run interactively
docker run -it --entrypoint /bin/bash trading-agent:latest

# Check health
docker inspect trading-agent | grep Health
```

**Solutions**:

1. **Missing Environment Variables**
   ```bash
   # Check required variables
   docker run --rm trading-agent env

   # Pass environment file
   docker run --env-file .env trading-agent
   ```

2. **Wrong Command**
   ```dockerfile
   # Dockerfile
   CMD ["python", "-m", "src.autonomous_agent.trading_agent"]
   # Not: CMD ["python", "trading_agent.py"]
   ```

3. **Permission Issues**
   ```dockerfile
   # Create non-root user
   RUN useradd -m -u 1000 trader
   USER trader
   ```

---

### Problem: Docker Compose services can't communicate

**Symptoms**:
```
Connection refused when accessing database
```

**Solution**:
```yaml
# docker-compose.yml
services:
  api:
    environment:
      - DATABASE_URL=postgresql://user:pass@database:5432/db
      # Use service name "database", not "localhost"

  database:
    networks:
      - trading-network

networks:
  trading-network:
    driver: bridge
```

---

## Logging & Monitoring

### Problem: No logs appearing

**Symptoms**:
Logs directory empty or no recent entries

**Solution**:
```bash
# Check logging configuration
python -c "
from utils.structured_logging import setup_structured_logging
setup_structured_logging(log_file='test.log')
import logging
logging.info('Test message')
"

# Check file permissions
ls -la logs/
chmod 755 logs/

# Check log rotation
# logrotate.conf
/var/log/trading-agent/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

---

### Problem: Grafana dashboard shows no data

**Symptoms**:
Dashboards empty despite agent running

**Solution**:
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check metrics endpoint
curl http://localhost:8000/metrics

# Verify Prometheus scraping
docker logs trading-prometheus | grep -i error

# Check Grafana datasource
curl http://admin:admin@localhost:3000/api/datasources
```

---

## Getting Help

### Collect Diagnostic Information

Before asking for help, collect:

```bash
#!/bin/bash
# diagnostic_info.sh

echo "=== System Info ==="
uname -a
python --version
docker --version

echo -e "\n=== Service Status ==="
docker-compose ps

echo -e "\n=== Recent Logs ==="
docker logs trading-agent --tail 50

echo -e "\n=== Database Status ==="
psql -U trading_user -d trading_db -c "SELECT version();"

echo -e "\n=== API Health ==="
curl http://localhost:8000/health/

echo -e "\n=== Disk Space ==="
df -h

echo -e "\n=== Memory ==="
free -h
```

### Support Channels

- **Documentation**: `docs/` directory
- **GitHub Issues**: [github.com/trading-ai/issues](https://github.com/trading-ai/issues)
- **Community**: Discord server
- **Email**: support@trading-ai.local

### Emergency Contacts

- **On-Call Engineer**: [phone/slack]
- **Database Admin**: dba@trading-ai.local
- **Security Team**: security@trading-ai.local

---

**Document Version**: 1.0
**Last Updated**: 2026-02-16
**Maintainer**: Engineering Team
