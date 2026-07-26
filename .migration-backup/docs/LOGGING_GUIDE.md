# Centralized Logging Guide

## Overview

The trading AI system uses centralized logging with Grafana Loki for log aggregation and analysis. All logs are collected in JSON format for structured querying and analysis.

## Architecture

```
┌─────────────────┐
│  Trading Agent  │──┐
└─────────────────┘  │
                     │
┌─────────────────┐  │   ┌──────────────┐   ┌──────────┐   ┌──────────┐
│   REST API      │──┼──▶│   Promtail   │──▶│   Loki   │──▶│ Grafana  │
└─────────────────┘  │   └──────────────┘   └──────────┘   └──────────┘
                     │   (Log Collection)   (Storage &     (Visualization)
┌─────────────────┐  │                       Query)
│   Database      │──┘
└─────────────────┘
```

## Setup

### Quick Start

```bash
# Start logging stack
cd docker
docker-compose -f logging-stack.yml up -d

# Access Grafana
open http://localhost:3001

# Login credentials
# Username: admin
# Password: admin
```

### Components

1. **Grafana Loki** (Port 3100)
   - Log aggregation and storage
   - Lightweight alternative to Elasticsearch
   - Efficient log storage and querying

2. **Promtail** (Port 9080)
   - Log shipper
   - Collects logs from containers and files
   - Sends logs to Loki

3. **Grafana** (Port 3001)
   - Log visualization
   - Dashboard creation
   - Alert configuration

## Logging Standards

### Log Format

All logs must be in JSON format with the following structure:

```json
{
  "timestamp": "2025-01-15T10:30:00.000Z",
  "time_ms": 1705315800000,
  "level": "INFO",
  "message": "Trade executed successfully",
  "service": "trading-agent",
  "environment": "production",
  "hostname": "trading-node-01",
  "logger": "agent.executor",
  "module": "trade_executor",
  "function": "execute_trade",
  "line": 142,
  "thread": 12345,
  "thread_name": "MainThread",
  "process": 1234,
  "process_name": "trading-agent",
  "trade_id": "trade-12345",
  "agent_id": "agent-001",
  "symbol": "BTCUSDT",
  "side": "BUY",
  "quantity": 0.1,
  "price": 45000.0,
  "value": 4500.0,
  "event_type": "trade_executed"
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| timestamp | string | ISO 8601 timestamp (UTC) |
| level | string | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| message | string | Human-readable message |
| service | string | Service name |
| environment | string | Environment (development, staging, production) |
| logger | string | Logger name |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| agent_id | string | Trading agent identifier |
| trade_id | string | Trade identifier |
| order_id | string | Order identifier |
| symbol | string | Trading symbol |
| user_id | string | User identifier |
| event_type | string | Event type for filtering |
| exception | object | Exception details (type, message, traceback) |
| extra | object | Additional contextual data |

### Log Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General informational messages
- **WARNING**: Warning messages (potential issues)
- **ERROR**: Error messages (failures that don't stop execution)
- **CRITICAL**: Critical messages (system-wide failures)

## Using Structured Logging

### Python Implementation

```python
from utils.structured_logging import setup_structured_logging, get_structured_logger

# Configure logging (once at startup)
setup_structured_logging(
    service_name="trading-agent",
    environment="production",
    level=logging.INFO,
    log_file="/var/log/trading-agent/agent.log"
)

# Get logger
logger = get_structured_logger(__name__)

# Basic logging
logger.info("Agent started")
logger.warning("High memory usage detected")
logger.error("Failed to connect to exchange")

# Logging with context
logger.info(
    "Request processed",
    request_id="req-123",
    duration_ms=150,
    status_code=200
)

# Specialized logging methods
logger.trade_executed(
    trade_id="trade-001",
    symbol="BTCUSDT",
    side="BUY",
    quantity=0.1,
    price=45000.0,
    strategy="dca_bot"
)

logger.agent_state_change(
    agent_id="agent-001",
    old_state="RUNNING",
    new_state="PAUSED",
    reason="manual_pause"
)
```

### FastAPI Integration

```python
from fastapi import FastAPI, Request
from utils.structured_logging import get_structured_logger
import time

app = FastAPI()
logger = get_structured_logger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Log request
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path}",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
        event_type="http_request"
    )

    return response
```

### Docker Configuration

Add logging driver to docker-compose.yml:

```yaml
services:
  trading-agent:
    image: trading-agent:latest
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
        labels: "service,environment"
    labels:
      service: "trading-agent"
      environment: "production"
```

## Querying Logs

### LogQL Basics

LogQL is Loki's query language, similar to PromQL.

**Basic query:**
```
{service="trading-agent"}
```

**Filter by log level:**
```
{service="trading-agent"} | json | level="ERROR"
```

**Search for text:**
```
{service="trading-agent"} |= "trade executed"
```

**Filter by field:**
```
{service="trading-agent"} | json | symbol="BTCUSDT"
```

**Aggregate counts:**
```
sum(count_over_time({service="trading-agent"}[5m])) by (level)
```

### Common Queries

**All errors in last hour:**
```
{service="trading-agent"} | json | level="ERROR" | line_format "{{.timestamp}} {{.message}}"
```

**Trades for specific agent:**
```
{service="trading-agent"} | json | event_type="trade_executed" | agent_id="agent-001"
```

**Failed orders:**
```
{service="trading-agent"} | json | event_type="order_placed" | status="FAILED"
```

**High-value trades:**
```
{service="trading-agent"} | json | event_type="trade_executed" | value > 10000
```

**Request latency (95th percentile):**
```
quantile_over_time(0.95, {service="trading-api"} | json | unwrap duration_ms [5m])
```

**Error rate:**
```
sum(rate({service="trading-agent"} | json | level="ERROR" [5m])) by (service)
```

## Log Retention

### Retention Policies

| Environment | Retention Period | Storage Limit |
|-------------|------------------|---------------|
| Development | 7 days | 10 GB |
| Staging | 30 days | 50 GB |
| Production | 90 days | 200 GB |

### Configuration

Edit `loki-config.yml`:

```yaml
table_manager:
  retention_deletes_enabled: true
  retention_period: 2160h  # 90 days
```

### Manual Cleanup

```bash
# Compact old chunks
docker exec trading-loki loki-cli compactor compact

# Check storage usage
docker exec trading-loki du -sh /loki/*
```

## Alerting

### Configure Alerts

Create alert rules in `loki-rules.yml`:

```yaml
groups:
  - name: trading_alerts
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          sum(rate({service="trading-agent"} | json | level="ERROR" [5m])) by (service)
          > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "{{ $labels.service }} has {{ $value }} errors/sec"

      # Circuit breaker opened
      - alert: CircuitBreakerOpen
        expr: |
          count_over_time({service="trading-agent"} | json | message=~"Circuit breaker OPENED.*" [5m])
          > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Circuit breaker opened"
          description: "A circuit breaker has opened, service may be degraded"

      # Agent crashed
      - alert: AgentCrashed
        expr: |
          count_over_time({service="trading-agent"} | json | level="CRITICAL" [5m])
          > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Trading agent crashed"
          description: "A trading agent has crashed, immediate attention required"
```

### Grafana Alerts

1. Go to Grafana → Alerting → Alert rules
2. Create new alert rule
3. Set query:
   ```
   count_over_time({service="trading-agent"} | json | level="ERROR" [5m]) > 10
   ```
4. Set notification channel (Slack, email, PagerDuty)

## Best Practices

### Do's

1. **Always log in JSON format** for structured queries
2. **Include context fields** (agent_id, trade_id, etc.)
3. **Use appropriate log levels** (don't log everything as INFO)
4. **Log business events** (trades, orders, state changes)
5. **Include timing information** (duration_ms, timestamp)
6. **Log errors with full context** (include exception details)
7. **Use consistent field names** across services

### Don'ts

1. **Don't log sensitive data** (API keys, passwords, PII)
2. **Don't log too verbosely** (avoid DEBUG in production)
3. **Don't use unstructured messages** (use JSON format)
4. **Don't log in tight loops** (use sampling)
5. **Don't ignore log retention** (clean up old logs)
6. **Don't forget error context** (always include relevant fields)

### Example: Good vs Bad Logging

**Bad:**
```python
logger.info("Trade executed")
```

**Good:**
```python
logger.trade_executed(
    trade_id="trade-001",
    symbol="BTCUSDT",
    side="BUY",
    quantity=0.1,
    price=45000.0,
    value=4500.0,
    strategy="dca_bot",
    agent_id="agent-001"
)
```

**Bad:**
```python
logger.error("Error occurred")
```

**Good:**
```python
logger.error(
    "Failed to place order",
    order_id="order-123",
    symbol="ETHUSDT",
    error_type="RateLimitExceeded",
    retry_count=3,
    exception=str(e)
)
```

## Grafana Dashboards

### Pre-built Dashboards

1. **System Overview**
   - Log volume by service
   - Error rate
   - Top errors
   - Service health

2. **Trading Activity**
   - Trades executed (count, volume)
   - Orders placed
   - Order fill rate
   - Strategy performance

3. **Agent Monitoring**
   - Agent state changes
   - Agent errors
   - Performance metrics
   - Resource usage

4. **API Performance**
   - Request rate
   - Latency (p50, p95, p99)
   - Error rate by endpoint
   - Status code distribution

### Creating Custom Dashboards

1. Go to Grafana → Dashboards → New
2. Add panel
3. Set data source: Loki
4. Enter LogQL query
5. Configure visualization
6. Save dashboard

## Troubleshooting

### No Logs Appearing

1. **Check Promtail is running:**
   ```bash
   docker ps | grep promtail
   docker logs trading-promtail
   ```

2. **Check Loki is accessible:**
   ```bash
   curl http://localhost:3100/ready
   ```

3. **Verify log format:**
   ```bash
   # Check if logs are JSON
   docker logs trading-agent --tail 10
   ```

4. **Check Promtail positions:**
   ```bash
   docker exec trading-promtail cat /tmp/positions.yaml
   ```

### High Storage Usage

1. **Check retention policy:**
   ```yaml
   # loki-config.yml
   table_manager:
     retention_period: 168h  # Reduce if needed
   ```

2. **Compact chunks manually:**
   ```bash
   docker exec trading-loki loki-cli compactor compact
   ```

3. **Clean up old data:**
   ```bash
   # Delete logs older than 30 days
   docker exec trading-loki find /loki/chunks -type f -mtime +30 -delete
   ```

### Query Performance Issues

1. **Add time range filter:**
   ```
   {service="trading-agent"}[1h]  # Instead of searching all time
   ```

2. **Use labels for filtering:**
   ```
   {service="trading-agent", level="ERROR"}  # Better than |= "ERROR"
   ```

3. **Index frequently-queried fields:**
   ```yaml
   # promtail-config.yml
   - labels:
       level:
       agent_id:
       event_type:
   ```

## Performance Tips

1. **Use label selectors** instead of line filters when possible
2. **Add time range** to all queries
3. **Limit result size** with `| limit 1000`
4. **Use sampling** for high-volume logs
5. **Create materialized views** for common queries
6. **Index important fields** as labels

## Security

1. **Enable authentication:**
   ```yaml
   # grafana.ini
   [auth]
   disable_login_form = false
   ```

2. **Use HTTPS:**
   ```yaml
   # nginx.conf
   server {
       listen 443 ssl;
       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;
       location / {
           proxy_pass http://grafana:3000;
       }
   }
   ```

3. **Restrict access by IP:**
   ```yaml
   # docker-compose.yml
   networks:
       logging:
           ipam:
               config:
                   - subnet: 172.20.0.0/16
                     ip_range: 172.20.5.0/24
   ```

4. **Don't log sensitive data:**
   - API keys
   - Passwords
   - Credit card numbers
   - Personal information

## Support

- Loki documentation: https://grafana.com/docs/loki/latest/
- LogQL reference: https://grafana.com/docs/loki/latest/logql/
- Grafana dashboards: https://grafana.com/grafana/dashboards/
- Community: https://community.grafana.com/

---

**Last Updated**: 2025-01-15
**Version**: 1.0
