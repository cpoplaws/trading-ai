"""
Trading AI Real-Time Dashboard

Streamlit-based dashboard for monitoring autonomous trading agents.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from typing import Dict, List

# Configuration
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")
API_KEY = st.secrets.get("API_KEY", "sk_test_12345")

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Page config
st.set_page_config(
    page_title="Trading AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .positive {
        color: #00c853;
    }
    .negative {
        color: #d32f2f;
    }
    .agent-running {
        color: #00c853;
        font-weight: bold;
    }
    .agent-stopped {
        color: #d32f2f;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def api_get(endpoint: str):
    """Make GET request to API."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", headers=headers, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def api_post(endpoint: str, data: Dict = None):
    """Make POST request to API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            headers=headers,
            json=data,
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"${value:,.2f}"


def format_percent(value: float) -> str:
    """Format value as percentage."""
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def get_color(value: float) -> str:
    """Get color based on positive/negative value."""
    return "positive" if value >= 0 else "negative"


# Sidebar
with st.sidebar:
    st.title("📈 Trading AI")
    st.markdown("---")

    # Agent selection
    agents = api_get("/api/v1/agents/")
    if agents:
        agent_options = {
            f"{a['agent_id'][:8]}... (${a['portfolio_value']:,.0f})": a['agent_id']
            for a in agents
        }
        selected_agent_label = st.selectbox(
            "Select Agent",
            options=list(agent_options.keys()) if agent_options else ["No agents"]
        )
        selected_agent = agent_options.get(selected_agent_label)
    else:
        selected_agent = None
        st.warning("No agents found")

    st.markdown("---")

    # Create new agent
    with st.expander("➕ Create New Agent"):
        initial_capital = st.number_input("Initial Capital", value=10000.0, step=1000.0)
        paper_trading = st.checkbox("Paper Trading", value=True)
        strategies = st.multiselect(
            "Strategies",
            ["dca_bot", "momentum", "mean_reversion", "market_making"],
            default=["dca_bot"]
        )

        if st.button("Create Agent"):
            data = {
                "initial_capital": initial_capital,
                "paper_trading": paper_trading,
                "enabled_strategies": strategies,
                "check_interval_seconds": 5,
                "max_daily_loss": initial_capital * 0.05
            }
            result = api_post("/api/v1/agents/", data)
            if result:
                st.success(f"Agent created: {result['agent_id']}")
                st.rerun()

    st.markdown("---")

    # Auto-refresh
    auto_refresh = st.checkbox("Auto Refresh (5s)", value=True)
    if auto_refresh:
        time.sleep(5)
        st.rerun()


# Main content
if not selected_agent:
    st.title("Welcome to Trading AI Dashboard")
    st.info("Select or create an agent to get started")
    st.stop()

# Get agent data
agent_status = api_get(f"/api/v1/agents/{selected_agent}")
portfolio_summary = api_get("/api/v1/portfolio/summary")
positions = api_get("/api/v1/portfolio/positions")
trades = api_get("/api/v1/portfolio/trades?limit=20")
metrics = api_get(f"/api/v1/agents/{selected_agent}/metrics")

if not agent_status:
    st.error("Failed to load agent data")
    st.stop()

# Header with agent controls
col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

with col1:
    state = agent_status['state']
    state_class = "agent-running" if state == "running" else "agent-stopped"
    st.markdown(f"# Agent Status: <span class='{state_class}'>{state.upper()}</span>", unsafe_allow_html=True)

with col2:
    if state == "running":
        if st.button("⏸️ Pause", use_container_width=True):
            api_post(f"/api/v1/agents/{selected_agent}/pause")
            st.rerun()
    elif state == "paused":
        if st.button("▶️ Resume", use_container_width=True):
            api_post(f"/api/v1/agents/{selected_agent}/resume")
            st.rerun()
    else:
        if st.button("▶️ Start", use_container_width=True):
            api_post(f"/api/v1/agents/{selected_agent}/start")
            st.rerun()

with col3:
    if state == "running" or state == "paused":
        if st.button("⏹️ Stop", use_container_width=True):
            api_post(f"/api/v1/agents/{selected_agent}/stop")
            st.rerun()

with col4:
    if state == "stopped":
        if st.button("🗑️ Delete", use_container_width=True):
            api_post(f"/api/v1/agents/{selected_agent}", method="DELETE")
            st.rerun()

st.markdown("---")

# Key Metrics
if portfolio_summary:
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Portfolio Value",
            format_currency(portfolio_summary['total_value_usd']),
            delta=format_currency(portfolio_summary['daily_pnl'])
        )

    with col2:
        pnl_color = get_color(portfolio_summary['total_pnl'])
        st.metric(
            "Total P&L",
            format_currency(portfolio_summary['total_pnl']),
            delta=format_percent(portfolio_summary['total_pnl_percent'])
        )

    with col3:
        st.metric(
            "Cash Balance",
            format_currency(portfolio_summary['cash_balance_usd'])
        )

    with col4:
        st.metric(
            "Positions",
            portfolio_summary['num_positions']
        )

    with col5:
        st.metric(
            "Total Trades",
            agent_status['total_trades']
        )

st.markdown("---")

# Performance Metrics
if metrics:
    st.subheader("📊 Performance Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")

    with col2:
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")

    with col3:
        st.metric("Win Rate", f"{metrics['win_rate']:.1%}")

    with col4:
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")

    with col5:
        st.metric("Avg Win", format_currency(metrics['avg_win']))

st.markdown("---")

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Portfolio Value")

    # Get performance data
    performance = api_get("/api/v1/portfolio/performance?period=7d")
    if performance and performance.get('equity_curve'):
        df = pd.DataFrame(performance['equity_curve'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.1)'
        ))

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Value (USD)",
            hovermode='x unified',
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🎯 Strategy Performance")

    # Get strategy performance
    strat_perf = api_get(f"/api/v1/agents/{selected_agent}/strategies/performance")
    if strat_perf:
        df = pd.DataFrame(strat_perf)

        fig = px.bar(
            df,
            x='strategy',
            y='total_pnl',
            color='win_rate',
            color_continuous_scale='RdYlGn',
            labels={'total_pnl': 'Total P&L', 'strategy': 'Strategy', 'win_rate': 'Win Rate'},
            height=400
        )

        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Positions and Trades
col1, col2 = st.columns(2)

with col1:
    st.subheader("💼 Open Positions")

    if positions:
        positions_df = pd.DataFrame(positions)
        positions_df['Unrealized P&L'] = positions_df.apply(
            lambda row: f"{format_currency(row['unrealized_pnl'])} ({format_percent(row['unrealized_pnl_percent'])})",
            axis=1
        )

        st.dataframe(
            positions_df[[
                'symbol',
                'quantity',
                'entry_price',
                'current_price',
                'Unrealized P&L'
            ]].rename(columns={
                'symbol': 'Symbol',
                'quantity': 'Qty',
                'entry_price': 'Entry',
                'current_price': 'Current'
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No open positions")

with col2:
    st.subheader("📝 Recent Trades")

    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df['executed_at'] = pd.to_datetime(trades_df['executed_at'])
        trades_df['time'] = trades_df['executed_at'].dt.strftime('%H:%M:%S')

        st.dataframe(
            trades_df[[
                'time',
                'symbol',
                'side',
                'quantity',
                'price',
                'strategy'
            ]].rename(columns={
                'time': 'Time',
                'symbol': 'Symbol',
                'side': 'Side',
                'quantity': 'Qty',
                'price': 'Price',
                'strategy': 'Strategy'
            }).head(10),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No trades yet")

st.markdown("---")

# Risk Metrics
st.subheader("⚠️ Risk Metrics")

risk_metrics = api_get("/api/v1/portfolio/risk-metrics")
if risk_metrics:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("VaR (95%)", format_currency(risk_metrics['var_95']))

    with col2:
        st.metric("CVaR (95%)", format_currency(risk_metrics['cvar_95']))

    with col3:
        st.metric("Volatility", f"{risk_metrics['volatility']:.1%}")

    with col4:
        st.metric("Beta", f"{risk_metrics['beta']:.2f}")

    # Exposure breakdown
    st.markdown("#### Exposure")
    exposure = risk_metrics['exposure']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Long", format_currency(exposure['long']))
    with col2:
        st.metric("Short", format_currency(exposure['short']))
    with col3:
        st.metric("Net", format_currency(exposure['net']))
    with col4:
        st.metric("Gross", format_currency(exposure['gross']))

st.markdown("---")

# Market Data
st.subheader("💹 Market Data")

col1, col2, col3 = st.columns(3)

for i, symbol in enumerate(['BTC-USD', 'ETH-USD', 'SOL-USD']):
    ticker = api_get(f"/api/v1/market/ticker/{symbol}")
    if ticker:
        with [col1, col2, col3][i]:
            price_change = ticker['price_change_percent_24h']
            color = "🟢" if price_change >= 0 else "🔴"

            st.markdown(f"### {color} {symbol}")
            st.metric(
                "Price",
                format_currency(ticker['price']),
                delta=format_percent(price_change)
            )
            st.caption(f"24h Vol: {format_currency(ticker['volume_24h'])}")

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh: {auto_refresh}")
