# Trading AI Dashboard Guide

## Overview

The Trading AI Dashboard provides real-time monitoring and control of autonomous trading agents. Built with Streamlit, it offers an intuitive interface for portfolio management, performance tracking, and risk analysis.

## Features

### 1. Real-Time Monitoring
- **Portfolio Value**: Live tracking of total portfolio value with historical chart
- **P&L Tracking**: Total and daily profit/loss with percentage changes
- **Position Monitoring**: Real-time view of all open positions with unrealized P&L
- **Trade Feed**: Live feed of executed trades with timestamps and details

### 2. Agent Control
- **Start/Stop/Pause**: Full control over agent execution
- **Strategy Toggle**: Enable or disable individual strategies on the fly
- **Risk Limit Updates**: Adjust risk parameters without restarting

### 3. Performance Analytics
- **Sharpe Ratio**: Risk-adjusted return measurement
- **Win Rate**: Percentage of profitable trades
- **Max Drawdown**: Largest peak-to-trough decline
- **Profit Factor**: Ratio of gross profit to gross loss
- **Strategy Breakdown**: Performance by individual strategy

### 4. Risk Management
- **Value at Risk (VaR)**: Potential loss at 95% confidence
- **Conditional VaR (CVaR)**: Expected loss beyond VaR
- **Exposure Analysis**: Long/short/net exposure breakdown
- **Volatility Tracking**: Portfolio volatility metrics
- **Beta**: Market correlation measurement

### 5. Market Data
- **Live Prices**: Real-time ticker data for major cryptocurrencies
- **24h Changes**: Price changes and volume
- **Quick Access**: Key market data at a glance

### 6. Multi-Agent Support
- **Agent Switching**: Easily switch between multiple agents
- **Agent Creation**: Create new agents directly from dashboard
- **Agent Comparison**: Compare performance across agents

## Installation

### Quick Start

```bash
cd src/dashboard
./run_dashboard.sh
```

The dashboard will be available at http://localhost:8501

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run streamlit_app.py
```

### Docker Deployment

```bash
# Build image
docker build -t trading-ai-dashboard .

# Run container
docker run -p 8501:8501 \
  -e API_BASE_URL=http://api:8000 \
  -e API_KEY=your_api_key \
  trading-ai-dashboard
```

### Docker Compose

```yaml
version: '3.8'

services:
  dashboard:
    build: ./src/dashboard
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://api:8000
      - API_KEY=${API_KEY}
    depends_on:
      - api
    restart: unless-stopped
```

## Configuration

### API Connection

Edit `.streamlit/secrets.toml`:

```toml
API_BASE_URL = "http://localhost:8000"
API_KEY = "sk_your_api_key"
```

### Theme Customization

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
port = 8501
maxUploadSize = 200
```

## Usage Guide

### Starting an Agent

1. **Select or Create Agent**
   - Use sidebar dropdown to select existing agent
   - Or expand "Create New Agent" to create a new one

2. **Configure Agent**
   - Set initial capital
   - Choose paper trading or live trading
   - Select strategies to enable

3. **Start Trading**
   - Click "▶️ Start" button
   - Monitor status indicator (green = running)
   - Watch portfolio value update in real-time

### Monitoring Performance

#### Portfolio Dashboard
- **Portfolio Value Chart**: Shows value over time (7-day default)
- **Key Metrics**: Total value, P&L, cash balance, positions, trades
- **Performance Metrics**: Sharpe, drawdown, win rate, profit factor

#### Strategy Performance
- **Bar Chart**: Visual comparison of strategy P&L
- **Color Coding**: Win rate shown by color intensity
- **Hover Details**: See exact numbers on hover

#### Positions Table
- **Symbol**: Trading pair
- **Quantity**: Amount held
- **Entry/Current Price**: Buy price vs current price
- **Unrealized P&L**: Current profit/loss ($ and %)

#### Trade Feed
- **Recent Trades**: Last 10-20 trades
- **Timestamp**: Execution time
- **Details**: Symbol, side (BUY/SELL), quantity, price
- **Strategy**: Which strategy generated the trade

### Risk Management

#### Risk Metrics Panel
- **VaR 95%**: Maximum expected loss at 95% confidence
- **CVaR 95%**: Expected loss if VaR is exceeded
- **Volatility**: Portfolio price volatility
- **Beta**: Correlation with market

#### Exposure Breakdown
- **Long Exposure**: Total long positions value
- **Short Exposure**: Total short positions value
- **Net Exposure**: Long minus short
- **Gross Exposure**: Total absolute exposure

### Agent Controls

#### State Management
- **▶️ Start**: Begin trading
- **⏸️ Pause**: Temporarily pause (maintains positions)
- **▶️ Resume**: Resume from pause
- **⏹️ Stop**: Stop trading (close positions recommended)
- **🗑️ Delete**: Permanently delete agent (stopped only)

#### Strategy Control
- Enable/disable strategies in agent settings
- Changes take effect immediately
- Monitor strategy-specific performance

### Auto-Refresh

- **Toggle**: Enable/disable in sidebar
- **Interval**: Updates every 5 seconds
- **Manual Refresh**: Refresh button or F5

## Dashboard Layout

```
┌─────────────────────────────────────────────────┐
│  Sidebar                                        │
│  - Agent Selection                              │
│  - Create Agent                                 │
│  - Auto-Refresh Toggle                          │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  Header: Agent Status & Controls                │
│  [Agent State] [Pause/Resume] [Stop] [Delete]   │
└─────────────────────────────────────────────────┘

┌──────────────┬──────────────┬──────────────────┐
│  Portfolio   │  Total P&L   │  Cash Balance   │
│  Value       │              │                 │
└──────────────┴──────────────┴──────────────────┘

┌──────────────┬──────────────┬──────────────────┐
│  Sharpe      │  Max         │  Win Rate       │
│  Ratio       │  Drawdown    │                 │
└──────────────┴──────────────┴──────────────────┘

┌───────────────────────┬─────────────────────────┐
│  Portfolio Value      │  Strategy Performance   │
│  [Line Chart]         │  [Bar Chart]            │
└───────────────────────┴─────────────────────────┘

┌───────────────────────┬─────────────────────────┐
│  Open Positions       │  Recent Trades          │
│  [Table]              │  [Table]                │
└───────────────────────┴─────────────────────────┘

┌─────────────────────────────────────────────────┐
│  Risk Metrics                                   │
│  VaR, CVaR, Volatility, Beta, Exposure          │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  Market Data                                    │
│  BTC-USD    ETH-USD    SOL-USD                  │
└─────────────────────────────────────────────────┘
```

## Features in Detail

### Portfolio Value Chart
- **Type**: Time series line chart with filled area
- **Data**: Historical portfolio values
- **Period**: Default 7 days, configurable
- **Interactivity**: Hover for exact values, zoom, pan

### Strategy Performance Chart
- **Type**: Bar chart with color gradient
- **X-axis**: Strategy names
- **Y-axis**: Total P&L
- **Color**: Win rate (green = high, red = low)

### Position Management
- **Real-time Updates**: Prices update with auto-refresh
- **P&L Calculation**: Automatic unrealized P&L calculation
- **Color Coding**: Green for profit, red for loss

### Trade Feed
- **Chronological Order**: Most recent trades first
- **Filtering**: Can be filtered by symbol or strategy
- **Details**: Complete trade information

## Troubleshooting

### Dashboard Won't Start

```bash
# Check if port 8501 is available
lsof -i :8501

# Kill existing process if needed
kill -9 $(lsof -t -i :8501)

# Start dashboard
streamlit run streamlit_app.py
```

### API Connection Issues

1. **Check API is running**
   ```bash
   curl http://localhost:8000/health/
   ```

2. **Verify API key**
   - Check `.streamlit/secrets.toml`
   - Ensure API key is correct

3. **Check firewall**
   - Ensure port 8000 is accessible
   - Check CORS settings if running remotely

### Data Not Loading

1. **Check API endpoints**
   ```bash
   curl -H "X-API-Key: your_key" http://localhost:8000/api/v1/agents/
   ```

2. **Review browser console**
   - Open developer tools (F12)
   - Check for JavaScript errors

3. **Check Streamlit logs**
   - Terminal shows Streamlit logs
   - Look for error messages

### Performance Issues

1. **Disable auto-refresh** if CPU usage is high
2. **Reduce chart data points** by changing period
3. **Close unused browser tabs**
4. **Use production deployment** (not development mode)

## Production Deployment

### Using Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git add src/dashboard/
   git commit -m "Add dashboard"
   git push
   ```

2. **Deploy on Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Connect GitHub repository
   - Select `src/dashboard/streamlit_app.py`
   - Add secrets (API_BASE_URL, API_KEY)

### Using Docker

```bash
# Build
docker build -t trading-ai-dashboard:latest .

# Run
docker run -d \
  --name dashboard \
  -p 8501:8501 \
  -e API_BASE_URL=https://api.example.com \
  -e API_KEY=sk_prod_key \
  --restart unless-stopped \
  trading-ai-dashboard:latest
```

### Using Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dashboard
spec:
  replicas: 2
  selector:
    matchLabels:
      app: dashboard
  template:
    metadata:
      labels:
        app: dashboard
    spec:
      containers:
      - name: dashboard
        image: trading-ai-dashboard:latest
        ports:
        - containerPort: 8501
        env:
        - name: API_BASE_URL
          value: "http://api-service:8000"
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: api-key
---
apiVersion: v1
kind: Service
metadata:
  name: dashboard-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8501
  selector:
    app: dashboard
```

### Behind Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name dashboard.example.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

## Security Best Practices

1. **API Keys**
   - Never commit API keys to version control
   - Use environment variables or secrets management
   - Rotate keys regularly

2. **Authentication**
   - Add authentication layer (e.g., OAuth2)
   - Use HTTPS in production
   - Implement rate limiting

3. **Access Control**
   - Restrict dashboard access by IP
   - Use VPN for remote access
   - Implement audit logging

4. **Data Security**
   - Encrypt data in transit (HTTPS)
   - Don't expose sensitive data in URLs
   - Sanitize user inputs

## Keyboard Shortcuts

- **R**: Rerun the app
- **C**: Clear cache
- **Ctrl+S**: Open settings menu
- **Esc**: Close sidebar

## Tips & Best Practices

1. **Keep Dashboard Open**: Auto-refresh works best with tab active
2. **Use Paper Trading First**: Test strategies before live trading
3. **Monitor Risk Metrics**: Regularly check VaR and drawdown
4. **Review Trade Feed**: Understand what strategies are doing
5. **Set Alerts**: Use API alerts for important events
6. **Regular Backups**: Export performance data periodically

## Advanced Features

### Custom Metrics
Add custom metrics by modifying `streamlit_app.py`:
```python
st.metric("Custom Metric", calculate_custom_metric())
```

### Additional Charts
Add charts using Plotly:
```python
fig = px.line(df, x='time', y='value')
st.plotly_chart(fig)
```

### Data Export
Export data for analysis:
```python
if st.button("Export Data"):
    df.to_csv('portfolio_data.csv')
```

## Support

- **Documentation**: https://docs.trading-ai.example.com
- **API Docs**: http://localhost:8000/docs
- **Issues**: https://github.com/trading-ai/dashboard/issues
- **Community**: https://discord.gg/trading-ai

## Changelog

### v1.0.0 (2025-01-15)
- Initial release
- Real-time portfolio monitoring
- Agent control interface
- Performance analytics
- Risk management dashboard
- Market data integration
