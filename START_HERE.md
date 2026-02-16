# 🚀 START HERE - Complete Setup Guide

## ⚡ Quick Setup (3 Steps)

###  **Step 1: Start Docker Desktop**
```bash
# Open Docker Desktop application
open -a Docker

# Wait 30 seconds for it to start, then verify:
docker ps
```

### **Step 2: Install Python Dependencies**
```bash
cd /Users/silasmarkowicz/trading-ai-working
./install-deps.sh
```

### **Step 3: Start All Services**
```bash
# Start Docker services
docker compose -f docker-compose.full.yml up -d

# Wait 20 seconds, then check:
docker compose -f docker-compose.full.yml ps
```

### ✅ **Done! Access Your System:**
- **API**: http://localhost:8000/docs
- **Grafana**: http://localhost:3001 (admin/admin)
- **Database**: localhost:5432

---

## 📋 Detailed Step-by-Step

### Prerequisites

**Required**:
- ✅ macOS (you have this)
- ✅ Docker Desktop
- ✅ Python 3.9+ (you have 3.9.6)
- ✅ 8GB RAM minimum
- ✅ 20GB free disk space

**Optional**:
- Xcode Command Line Tools
- HomeBrew

### Step-by-Step Installation

#### 1. **Check Docker**

```bash
# Is Docker Desktop installed?
ls /Applications/Docker.app

# If not, download from: https://www.docker.com/products/docker-desktop

# Start Docker Desktop
open -a Docker

# Verify Docker is running (wait 30 seconds first)
docker --version
docker ps
```

#### 2. **Install Core Dependencies**

Option A - Automated (Recommended):
```bash
cd /Users/silasmarkowicz/trading-ai-working
./install-deps.sh
```

Option B - Manual:
```bash
pip3 install numpy pandas fastapi uvicorn sqlalchemy redis pytest python-dotenv web3
```

#### 3. **Configure Environment**

```bash
# Edit .env file with your API keys
nano .env
```

**Required keys** (get free API keys from):
- **Binance**: https://www.binance.com/en/my/settings/api-management
- **Etherscan**: https://etherscan.io/myapikey
- **Alchemy**: https://www.alchemy.com/ (for Web3)

Example `.env`:
```bash
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
ETHERSCAN_API_KEY=your_key_here
ALCHEMY_API_KEY=your_key_here
```

#### 4. **Start Docker Services**

```bash
# Pull all images (first time only, ~2GB download)
docker compose -f docker-compose.full.yml pull

# Start all services
docker compose -f docker-compose.full.yml up -d

# Check they're running
docker compose -f docker-compose.full.yml ps

# Expected output: 5 containers running
# - trading_timescaledb
# - trading_redis
# - trading_api
# - trading_prometheus
# - trading_grafana
```

#### 5. **Verify Everything Works**

```bash
# Test database connection
docker exec trading_timescaledb pg_isready -U trading_user -d trading_db

# Test Redis
docker exec trading_redis redis-cli ping

# Test API
curl http://localhost:8000/health

# Should return: {"status":"healthy"}
```

#### 6. **Access Services**

Open in your browser:
- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboards**: http://localhost:3001
- **Prometheus Metrics**: http://localhost:9090

#### 7. **Run Tests**

```bash
# Run unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

---

## 🔧 Troubleshooting

### Docker Won't Start

```bash
# Check if Docker is running
pgrep -f Docker

# If not, manually open Docker Desktop
open -a Docker

# Give it 60 seconds to fully start
```

### Port Already in Use

```bash
# Find what's using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Or change the port in docker-compose.full.yml
```

### Permission Denied

```bash
# Make scripts executable
chmod +x *.sh

# Or run with bash
bash quick-start.sh
```

### Database Connection Error

```bash
# Check if TimescaleDB is running
docker ps | grep timescale

# View logs
docker logs trading_timescaledb

# Restart it
docker compose -f docker-compose.full.yml restart timescaledb
```

---

## 🎯 What You Can Do Now

### 1. **Test the API**

Visit http://localhost:8000/docs and try:
- `/health` - Check system health
- `/api/v1/market/ticker/BTCUSDT` - Get BTC price

### 2. **View Monitoring**

Visit http://localhost:3001:
- Login: admin / admin
- Explore pre-built dashboards

### 3. **Query Database**

```bash
docker exec -it trading_timescaledb psql -U trading_user -d trading_db

# Inside psql:
\dt  # List tables
SELECT * FROM ohlcv LIMIT 10;
```

### 4. **Run Python Code**

```python
# Test database connection
from src.database.database_manager import DatabaseManager

db = DatabaseManager(
    host='localhost',
    port=5432,
    database='trading_db',
    user='trading_user',
    password='trading_password'
)

print("✓ Database connected!")
```

### 5. **Test Real-Time Data**

```python
from src.realtime.binance_websocket import BinanceWebSocket
from src.realtime.websocket_manager import WebSocketConfig

config = WebSocketConfig(
    url='wss://stream.binance.com:9443/ws',
    symbols=['BTCUSDT'],
    streams=['ticker']
)

ws = BinanceWebSocket(config)
# ws.connect()  # Uncomment to test
```

### 6. **Calculate Risk Metrics**

```python
from src.risk_management.var_calculator import VaRCalculator
import numpy as np

calc = VaRCalculator(confidence_level=0.95)

# Generate sample returns
returns = np.random.normal(0.001, 0.02, 252)  # 1 year daily returns

# Calculate VaR
result = calc.calculate_var(
    returns=returns,
    method='historical',
    portfolio_value=100000
)

print(f"Portfolio Value: $100,000")
print(f"VaR (95%): ${result.var:,.2f}")
print(f"CVaR (95%): ${result.cvar:,.2f}")
print(f"Max expected loss: ${result.var:,.2f} (1 day, 95% confidence)")
```

---

## 📚 Next Steps

### Beginner Path
1. ✅ Complete this setup
2. 📊 Explore API at http://localhost:8000/docs
3. 🔍 View Grafana dashboards
4. 🧪 Run some tests: `pytest tests/unit/ -v`
5. 📖 Read `/Users/silasmarkowicz/trading-ai-working/docs/FULL_SETUP_GUIDE.md`

### Intermediate Path
1. 🔑 Add real API keys to `.env`
2. 📥 Collect historical data
3. 🤖 Train basic models
4. 📈 Run backtests
5. 💰 Start paper trading

### Advanced Path
1. 🧠 Train deep learning models
2. 🔗 Set up on-chain analytics
3. ⚡ Implement MEV detection
4. 🌐 Deploy to cloud (AWS/GCP)
5. 💸 Enable live trading

---

## 🆘 Getting Help

**Check these first:**
1. View logs: `docker compose -f docker-compose.full.yml logs -f`
2. Check services: `docker compose -f docker-compose.full.yml ps`
3. Read FULL_SETUP_GUIDE.md for detailed troubleshooting

**Common issues:**
- Docker not running → Open Docker Desktop app
- Port conflicts → Change ports in docker-compose.full.yml
- Permission errors → Run `chmod +x *.sh`
- Database errors → Check logs: `docker logs trading_timescaledb`

---

## ✅ Verification Checklist

Before moving forward, verify:
- [ ] Docker Desktop is running
- [ ] All 5 containers are up: `docker compose -f docker-compose.full.yml ps`
- [ ] API responds: `curl http://localhost:8000/health`
- [ ] Can access Grafana: http://localhost:3001
- [ ] Database is accessible: `docker exec trading_timescaledb pg_isready`
- [ ] Tests pass: `pytest tests/unit/ -v`

---

**You're ready to build! 🚀**
