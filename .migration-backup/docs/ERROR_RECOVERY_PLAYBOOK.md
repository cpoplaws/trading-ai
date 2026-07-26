# Error Recovery Playbook

## Overview

This playbook provides step-by-step procedures for handling common errors and failures in the trading system. Follow these procedures to quickly diagnose and recover from various failure scenarios.

## Quick Reference

| Error Type | Severity | First Action | Recovery Time |
|------------|----------|--------------|---------------|
| API Timeout | Transient | Automatic retry | 1-10 seconds |
| Rate Limit | Recoverable | Exponential backoff | 30-60 seconds |
| Authentication | Permanent | Check credentials | Manual fix |
| Database Connection | Recoverable | Reconnect | 5-30 seconds |
| WebSocket Disconnect | Transient | Auto-reconnect | 1-5 seconds |
| Agent Crash | Critical | State recovery | 10-60 seconds |
| Exchange Outage | Critical | Switch to fallback | 5-10 minutes |
| Data Corruption | Permanent | Restore from backup | 1-10 minutes |

## Error Classification

### Transient Errors
**Definition**: Temporary failures that resolve on their own.

**Examples**:
- Network timeouts
- Temporary connection loss
- Brief service interruptions

**Recovery Strategy**:
- Immediate retry (no delay)
- Max 3 quick retries
- If still failing, escalate to recoverable

**Implementation**:
```python
from utils.retry import retry, RetryStrategy

@retry(max_attempts=3, base_delay=0.5, strategy=RetryStrategy.FIXED)
def fetch_data():
    return api.get_data()
```

### Recoverable Errors
**Definition**: Failures that can be resolved with backoff and retry.

**Examples**:
- Rate limits
- Service temporarily unavailable (503)
- Connection refused
- Overload errors

**Recovery Strategy**:
- Exponential backoff retry
- Max 5-10 retries
- Gradually increasing delays
- Switch to fallback if exhausted

**Implementation**:
```python
@retry(
    max_attempts=5,
    base_delay=2.0,
    max_delay=60.0,
    strategy=RetryStrategy.EXPONENTIAL
)
def call_api():
    return exchange.place_order(...)
```

### Permanent Errors
**Definition**: Errors that won't be resolved by retrying.

**Examples**:
- Authentication failures (401, 403)
- Invalid requests (400)
- Not found (404)
- Invalid API keys

**Recovery Strategy**:
- Don't retry
- Log error
- Add to dead letter queue
- Alert administrator

**Implementation**:
```python
from utils.retry import classify_error, ErrorSeverity

try:
    result = exchange.place_order(...)
except Exception as e:
    severity = classify_error(e)
    if severity == ErrorSeverity.PERMANENT:
        logger.error(f"Permanent error, not retrying: {e}")
        dlq.add("order_placement", order_data, e)
        send_alert(e)
```

### Critical Errors
**Definition**: System-wide failures requiring immediate attention.

**Examples**:
- Out of memory
- Disk full
- Database corruption
- Multiple service failures

**Recovery Strategy**:
- Stop all operations
- Send critical alert
- Save state
- Manual intervention required

## Common Failure Scenarios

### 1. API Timeout

**Symptoms**:
- Requests hanging
- Timeout exceptions
- Slow response times

**Diagnosis**:
```bash
# Check network connectivity
ping api.binance.com

# Check DNS resolution
nslookup api.binance.com

# Test API endpoint
curl -w "Time: %{time_total}s\n" https://api.binance.com/api/v3/ping

# Check system time (important for signed requests)
date
```

**Recovery Steps**:
1. Automatic retry (3 attempts)
2. Increase timeout threshold
3. Switch to WebSocket for real-time data
4. Use cached data temporarily
5. If persistent, switch to backup exchange

**Prevention**:
- Use WebSocket connections for real-time data
- Implement request caching
- Set appropriate timeouts (10-30s for REST)
- Use multiple data sources

### 2. Rate Limit Exceeded

**Symptoms**:
- 429 HTTP status
- "Rate limit exceeded" errors
- Requests being rejected

**Diagnosis**:
```python
# Check current request rate
from exchanges.binance_trading_client import BinanceTradingClient

client = BinanceTradingClient()
print(f"Requests in window: {client.request_count}")
print(f"Window start: {client.request_window_start}")
```

**Recovery Steps**:
1. Stop sending new requests immediately
2. Wait for rate limit window to reset (usually 1 minute)
3. Resume with reduced request rate
4. Implement request batching
5. Use WebSocket for data (doesn't count against REST limit)

**Prevention**:
- Monitor request rate proactively
- Use WebSocket for market data
- Batch account queries
- Cache frequently accessed data
- Implement request queue with rate limiting

**Code Example**:
```python
from utils.retry import retry, RetryStrategy

@retry(
    max_attempts=5,
    base_delay=60.0,  # Wait 1 minute before retry
    strategy=RetryStrategy.FIXED
)
def rate_limited_call():
    return exchange.get_account()
```

### 3. Authentication Failure

**Symptoms**:
- 401 Unauthorized
- 403 Forbidden
- "Invalid signature" errors
- "API key not found"

**Diagnosis**:
```python
# Test API credentials
client = BinanceTradingClient(testnet=True)

# Test connectivity (no auth)
if not client.test_connectivity():
    print("❌ Cannot reach API")

# Test authentication
try:
    balance = client.get_balance('USDT')
    print(f"✅ Auth working: {balance}")
except Exception as e:
    print(f"❌ Auth failed: {e}")
```

**Recovery Steps**:
1. **Verify API Keys**
   ```bash
   # Check environment variables
   echo $BINANCE_API_KEY
   echo $BINANCE_API_SECRET

   # Check .env file
   cat .env | grep BINANCE
   ```

2. **Check API Permissions**
   - Log into exchange
   - Go to API Management
   - Verify permissions are enabled (Reading, Spot Trading)
   - Check IP whitelist

3. **Verify Time Synchronization** (critical for signed requests)
   ```bash
   # Check system time
   date

   # Sync time
   sudo ntpdate pool.ntp.org

   # Or install NTP daemon
   sudo apt-get install ntp
   ```

4. **Test API Key**
   ```python
   # Verify signature generation
   client = BinanceTradingClient()
   server_time = client.get_server_time()
   local_time = int(time.time() * 1000)
   time_diff = abs(server_time - local_time)

   if time_diff > 5000:  # More than 5 seconds
       print(f"⚠️ Time diff: {time_diff}ms - sync system time!")
   ```

**Prevention**:
- Use NTP for time synchronization
- Rotate API keys regularly (every 90 days)
- Store keys securely (environment variables, not code)
- Set up IP whitelisting
- Monitor API usage logs

### 4. Database Connection Lost

**Symptoms**:
- "Connection refused"
- "No route to host"
- "Too many connections"
- Query timeouts

**Diagnosis**:
```bash
# Check database status
pg_isready -h localhost -p 5432

# Check connections
psql -U trading_user -d trading_db -c "SELECT count(*) FROM pg_stat_activity;"

# Check logs
tail -f /var/log/postgresql/postgresql.log
```

**Recovery Steps**:
1. **Automatic Reconnection**
   ```python
   from utils.retry import retry
   from database.models import session_scope

   @retry(max_attempts=3, base_delay=5.0)
   def save_trade(trade_data):
       with session_scope() as session:
           trade = Trade(**trade_data)
           session.add(trade)
           session.commit()
   ```

2. **Manual Reconnection**
   ```bash
   # Restart PostgreSQL
   sudo systemctl restart postgresql

   # Or restart container
   docker restart trading-db
   ```

3. **Connection Pool Cleanup**
   ```python
   from database.models import engine

   # Dispose of connection pool
   engine.dispose()

   # New connections will be created
   ```

4. **Check Connection Limits**
   ```sql
   -- Show max connections
   SHOW max_connections;

   -- Show current connections
   SELECT count(*) FROM pg_stat_activity;

   -- Kill idle connections
   SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE state = 'idle' AND state_change < now() - interval '5 minutes';
   ```

**Prevention**:
- Use connection pooling
- Close connections properly
- Set reasonable pool size (10-20 connections)
- Monitor connection count
- Implement connection timeout
- Use read replicas for heavy queries

### 5. WebSocket Disconnect

**Symptoms**:
- No real-time updates
- "Connection closed" errors
- Stale market data

**Diagnosis**:
```python
from realtime.binance_websocket import BinanceWebSocket

ws = BinanceWebSocket(config)

# Check connection status
if ws.connected:
    print("✅ Connected")
else:
    print("❌ Disconnected")

# Check last message time
time_since_message = time.time() - ws.last_message_time
if time_since_message > 60:
    print(f"⚠️ No messages for {time_since_message}s")
```

**Recovery Steps**:
1. **Automatic Reconnection** (already implemented)
   - WebSocket client auto-reconnects on disconnect
   - Exponential backoff (1s, 2s, 4s, 8s...)
   - Max 10 retry attempts

2. **Manual Reconnection**
   ```python
   # Disconnect
   await ws.disconnect()

   # Reconnect
   await ws.connect()

   # Resubscribe to channels
   await ws.subscribe(['ticker@BTCUSDT', 'depth@ETHUSDT'])
   ```

3. **Fallback to REST API**
   ```python
   if not ws.connected:
       # Use REST API temporarily
       price = binance_client.get_ticker_price('BTCUSDT')
   ```

**Prevention**:
- Implement heartbeat/ping (30-60s interval)
- Monitor connection health
- Have REST API fallback ready
- Use multiple WebSocket connections for redundancy
- Handle reconnection gracefully

### 6. Agent Crash

**Symptoms**:
- Agent process terminated
- No heartbeat
- Unresponsive to commands

**Diagnosis**:
```bash
# Check if process is running
ps aux | grep trading_agent

# Check system resources
top
free -h
df -h

# Check logs
tail -100 /var/log/trading_agent.log

# Check for segmentation faults
dmesg | grep -i "segmentation fault"
```

**Recovery Steps**:
1. **Automatic State Recovery**
   ```python
   from utils.state_recovery import StateRecoveryManager

   manager = StateRecoveryManager(agent_id)

   # Try to recover state
   state = manager.load_state()

   if state:
       # Restore agent from state
       agent.restore_from_state(state)
       print(f"✅ Recovered: {state['portfolio_value']}")
   ```

2. **Manual Recovery**
   ```bash
   # Check saved state
   ls /tmp/agent_state/

   # Restart agent with recovery
   python -m autonomous_agent.trading_agent --recover
   ```

3. **Inspect State File**
   ```bash
   # View state
   cat /tmp/agent_state/agent-001.json | jq .

   # Check backups
   ls /tmp/agent_state/agent-001_backups/
   ```

4. **Clean Restart** (if recovery fails)
   ```python
   # Delete corrupted state
   manager.delete_state()

   # Start fresh
   agent = AutonomousTradingAgent(config)
   agent.start()
   ```

**Prevention**:
- Save state checkpoints frequently (every 5-10 minutes)
- Handle exceptions properly (don't let them crash agent)
- Implement graceful shutdown
- Use process supervisor (systemd, supervisord)
- Monitor agent health
- Set up automatic restart on crash

### 7. Exchange Outage

**Symptoms**:
- All API requests failing
- "Service unavailable" (503)
- Website down
- No data updates

**Diagnosis**:
```bash
# Check exchange status page
curl https://www.binance.com/en/support/announcement

# Check API status
curl https://api.binance.com/api/v3/ping

# Check if it's just you
curl -I https://downforeveryoneorjustme.com/binance.com

# Check social media for announcements
```

**Recovery Steps**:
1. **Stop Trading Immediately**
   ```python
   agent.pause()  # Pause agent
   agent.cancel_all_orders()  # Cancel open orders
   ```

2. **Switch to Backup Exchange**
   ```python
   # Failover to Coinbase
   from exchanges.coinbase_client import CoinbaseClient

   backup_exchange = CoinbaseClient()
   agent.set_exchange(backup_exchange)
   ```

3. **Monitor Portfolio**
   - Check positions on exchange website/mobile app
   - Calculate current portfolio value
   - Determine if hedging is needed

4. **Wait for Recovery**
   - Monitor exchange status page
   - Don't panic trade
   - Resume when services restored

**Prevention**:
- Have backup exchange configured
- Monitor exchange status proactively
- Diversify across multiple exchanges
- Keep emergency funds on multiple exchanges
- Have manual trading access ready

### 8. Data Corruption

**Symptoms**:
- Database integrity errors
- Checksum failures
- Corrupted state files
- Inconsistent data

**Diagnosis**:
```bash
# Check database integrity
psql -U trading_user -d trading_db -c "SELECT * FROM pg_stat_database;"

# Check disk errors
sudo dmesg | grep -i "i/o error"

# Check file system
sudo fsck /dev/sda1

# Verify backups
ls -lh /backups/database/
```

**Recovery Steps**:
1. **Restore from Backup**
   ```bash
   # Stop agent
   systemctl stop trading-agent

   # Restore database
   pg_restore -U trading_user -d trading_db /backups/database/latest.dump

   # Restart agent
   systemctl start trading-agent
   ```

2. **Repair State Files**
   ```python
   manager = StateRecoveryManager(agent_id)

   # Try to restore from backup
   state = manager._restore_from_backup()

   if state:
       # Save repaired state
       manager.save_state(state)
   ```

3. **Reconcile with Exchange**
   ```python
   # Fetch current state from exchange
   exchange_balances = client.get_balances()
   exchange_orders = client.get_open_orders()

   # Update local state
   agent.reconcile_with_exchange(exchange_balances, exchange_orders)
   ```

**Prevention**:
- Regular automated backups (hourly for database, every 5 min for state)
- Use RAID for disk redundancy
- Implement checksums for state files
- Monitor disk health
- Test restore procedures regularly

## Dead Letter Queue Management

### View Failed Operations

```python
from utils.dead_letter_queue import DeadLetterQueue

dlq = DeadLetterQueue("/tmp/trading_dlq")

# List all failed operations
operations = dlq.list()
for op in operations:
    print(f"{op['operation_type']}: {op['failure_reason']}")

# Filter by type
trade_failures = dlq.list(operation_type="trade_execution")

# Filter by reason
rate_limited = dlq.list(failure_reason=FailureReason.RATE_LIMIT)

# Get statistics
stats = dlq.get_stats()
print(f"Total failures: {stats['total']}")
print(f"By type: {stats['by_type']}")
```

### Retry Failed Operations

```python
# Manual retry
def retry_trade(data):
    client = BinanceTradingClient()
    return client.place_market_order(
        symbol=data['symbol'],
        side=data['side'],
        quantity=data['quantity']
    )

# Retry specific operation
success = dlq.retry(operation_id, retry_trade)

# Batch retry all rate-limited operations
for op in dlq.list(failure_reason=FailureReason.RATE_LIMIT):
    time.sleep(10)  # Respect rate limits
    dlq.retry(op['id'], retry_trade)
```

### Cleanup DLQ

```python
# Remove old entries (older than 30 days)
removed = dlq.cleanup(max_age_days=30)
print(f"Removed {removed} old entries")

# Remove specific operation after manual resolution
dlq.remove(operation_id)
```

## Monitoring and Alerts

### Health Check Endpoints

```bash
# Check API health
curl http://localhost:8000/health/

# Check agent health
curl http://localhost:8000/api/v1/agents/agent-001/health

# Check database health
curl http://localhost:8000/health/database
```

### Log Analysis

```bash
# Find errors in last hour
grep -i error /var/log/trading_agent.log | tail -100

# Count error types
grep -i error /var/log/trading_agent.log | awk '{print $5}' | sort | uniq -c

# Find rate limit errors
grep "rate limit" /var/log/trading_agent.log

# Find auth errors
grep -E "401|403|unauthorized" /var/log/trading_agent.log
```

### Alert Configuration

```python
# Configure alerts for critical errors
from monitoring.alerts import AlertManager, AlertSeverity

alerts = AlertManager()

# Send alert on critical error
alerts.send_alert(
    title="Agent Crash",
    message=f"Agent {agent_id} crashed: {error}",
    severity=AlertSeverity.CRITICAL,
    channels=['telegram', 'slack']
)
```

## Maintenance Procedures

### Daily Checks
- [ ] Review error logs
- [ ] Check DLQ for failed operations
- [ ] Verify agent is running
- [ ] Check portfolio reconciliation
- [ ] Review trading performance

### Weekly Checks
- [ ] Analyze error patterns
- [ ] Update retry thresholds if needed
- [ ] Clean up old DLQ entries
- [ ] Review backup status
- [ ] Check disk space

### Monthly Checks
- [ ] Rotate API keys
- [ ] Test backup restoration
- [ ] Review and update error handling
- [ ] Update dependencies
- [ ] Security audit

## Emergency Contacts

### Critical Incidents
1. **Stop all agents immediately**: `./scripts/emergency_stop.sh`
2. **Check exchange status**: [Binance Status](https://www.binance.com/en/support/announcement)
3. **Review open positions**: Binance mobile app or website
4. **Alert team**: Post in #trading-alerts Slack channel

### Escalation Path
1. On-call engineer (immediate)
2. Team lead (if not resolved in 15 minutes)
3. CTO (if financial impact > $1000)

### Support Resources
- Exchange support: https://www.binance.com/en/support
- Database admin: dba@company.com
- DevOps team: #devops Slack channel
- Documentation: https://docs.trading-ai.com

## Appendix

### Error Code Reference

| Code | Type | Meaning | Action |
|------|------|---------|--------|
| 1001 | API | Timeout | Retry |
| 1002 | API | Rate limit | Backoff |
| 1003 | API | Invalid signature | Check time sync |
| 2001 | Database | Connection lost | Reconnect |
| 2002 | Database | Deadlock | Retry transaction |
| 3001 | Agent | Crash | State recovery |
| 3002 | Agent | Invalid state | Reset |
| 4001 | Exchange | Service unavailable | Failover |

### Useful Commands

```bash
# View agent status
systemctl status trading-agent

# View logs
journalctl -u trading-agent -f

# Restart agent
systemctl restart trading-agent

# Check database
psql -U trading_user -d trading_db -c "SELECT count(*) FROM trades;"

# Check Redis
redis-cli ping

# Check API
curl http://localhost:8000/health/

# Emergency stop
pkill -f trading_agent
```

### Configuration Files

- Agent config: `/etc/trading-agent/config.yml`
- Database config: `/etc/trading-agent/database.yml`
- API keys: `/etc/trading-agent/.env`
- Logs: `/var/log/trading-agent/`
- State: `/var/lib/trading-agent/state/`
- Backups: `/var/backups/trading-agent/`

## Best Practices

1. **Always test in testnet first** before deploying fixes to production
2. **Save state before making changes** to allow rollback
3. **Monitor after recovery** to ensure issue is fully resolved
4. **Document incidents** in post-mortem for future reference
5. **Update playbook** with new scenarios as they're discovered
6. **Regular drills** to practice emergency procedures
7. **Keep backups** of backups (3-2-1 rule)
8. **Test recovery procedures** regularly
9. **Automate** common recovery tasks
10. **Learn from failures** and improve resilience

---

**Last Updated**: 2025-01-15
**Version**: 1.0
**Maintainer**: Trading AI Team
