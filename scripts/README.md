# Data Collection Scripts

Scripts for collecting market data from Coinbase, Uniswap, and other sources.

## Setup

1. **Set API keys** in `.env`:
```bash
# Coinbase
COINBASE_API_KEY=your_key_here
COINBASE_API_SECRET=your_secret_here

# Ethereum RPC (Alchemy or Infura)
ETHEREUM_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY

# Etherscan (for gas prices)
ETHERSCAN_API_KEY=your_key_here
```

2. **Ensure database is running**:
```bash
docker compose -f docker-compose.full.yml ps
```

## Scripts

### 1. `collect_coinbase_data.py` - Coinbase Historical Data

Collects OHLCV candlestick data from Coinbase.

**Basic usage:**
```bash
python3 scripts/collect_coinbase_data.py
```

**Custom parameters:**
```bash
# Collect specific symbols
python3 scripts/collect_coinbase_data.py --symbols BTC-USD,ETH-USD

# Collect last 30 days
python3 scripts/collect_coinbase_data.py --days 30

# Use 5-minute candles
python3 scripts/collect_coinbase_data.py --granularity 300

# Don't save to database (just test)
python3 scripts/collect_coinbase_data.py --no-save
```

**Granularity options:**
- `60` = 1 minute
- `300` = 5 minutes
- `900` = 15 minutes
- `3600` = 1 hour (default)
- `21600` = 6 hours
- `86400` = 1 day

**Example output:**
```
======================================================================
📊 COINBASE HISTORICAL DATA COLLECTION
======================================================================

⚙️  Configuration:
  Symbols: BTC-USD, ETH-USD
  Period: 2025-02-08 to 2025-02-15 (7 days)
  Granularity: 3600s
  Save to DB: True

======================================================================
Collecting BTC-USD...
======================================================================
✓ Retrieved 168 candles
✓ Saved 168 candles to database

======================================================================
📊 COLLECTION COMPLETE
======================================================================
Total candles collected: 336
Symbols processed: 2
```

---

## Scheduling

### Run Every Hour (Cron)

Add to crontab (`crontab -e`):
```bash
0 * * * * cd /path/to/trading-ai-working && python3 scripts/collect_coinbase_data.py >> logs/collect.log 2>&1
```

### Run as Systemd Service

Create `/etc/systemd/system/trading-collector.service`:
```ini
[Unit]
Description=Trading Data Collector
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/trading-ai-working
ExecStart=/usr/bin/python3 scripts/collect_coinbase_data.py
Restart=always
RestartSec=3600

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable trading-collector
sudo systemctl start trading-collector
```

---

## Monitoring

Check database:
```bash
# Access database
docker exec -it trading_timescaledb psql -U trading_user -d trading_db

# Count candles
SELECT exchange, symbol, interval, COUNT(*)
FROM ohlcv
GROUP BY exchange, symbol, interval;

# Latest data
SELECT * FROM ohlcv
WHERE symbol = 'BTC-USD'
ORDER BY timestamp DESC
LIMIT 10;
```

---

## Troubleshooting

### "401 Unauthorized" Error
- Set your Coinbase API keys in `.env`
- Verify keys are correct at https://www.coinbase.com/settings/api

### "Connection refused" Error
- Check Docker services are running: `docker compose ps`
- Restart services: `docker compose restart`

### "No candles retrieved"
- Check symbol name (use hyphen: `BTC-USD` not `BTCUSD`)
- Try smaller date range
- Check Coinbase API status

### Rate Limiting
- Script has built-in 0.5s delays
- Coinbase allows ~10 requests/second
- For large datasets, run during off-peak hours

---

## Next Steps

1. **Set up real-time streaming** (coming soon):
   - WebSocket connections for live data
   - Sub-second price updates
   - Order book snapshots

2. **DEX data collection** (coming soon):
   - Uniswap pool reserves
   - Swap events
   - Gas prices

3. **Analytics** (coming soon):
   - Arbitrage detection
   - Signal generation
   - Backtesting

---

## Support

- Check logs: `tail -f logs/collect.log`
- View Docker logs: `docker compose logs -f`
- Test collectors: Run Python files directly with `--no-save` flag
