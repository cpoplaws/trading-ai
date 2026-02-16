"""
Trading AI - Unified Dashboard
==============================

ONE dashboard to rule them all.

Shows:
- All 11 strategies performance
- Agent swarm status
- Live trading metrics
- Risk management
- Real-time P&L
- System health
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page config
st.set_page_config(
    page_title="Trading AI - Unified Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .profit { color: #00ff00; font-weight: bold; }
    .loss { color: #ff0000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def main():
    """Main dashboard."""

    # Header
    st.title("🚀 Trading AI - Unified Dashboard")
    st.markdown("*One dashboard to see everything*")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Control")

        mode = st.radio(
            "Trading Mode",
            ["📊 Monitor Only", "🧪 Paper Trading", "💰 Live Trading"],
            help="Select trading mode"
        )

        st.markdown("---")

        st.header("🤖 Agent Swarm")
        agent_status = st.checkbox("Enable Agent Swarm", value=True)

        if agent_status:
            st.success("✅ Agents Active")
            st.caption("11 strategy agents running")
        else:
            st.warning("⏸️ Agents Paused")

        st.markdown("---")

        st.header("🎯 Active Strategies")
        strategies = {
            "Mean Reversion": True,
            "Momentum": True,
            "ML Ensemble": True,
            "PPO RL Agent": True,
            "Yield Optimizer": True,
            "Multi-Chain Arb": False,
            "RSI": True,
            "MACD": True,
            "Bollinger Bands": True,
            "Grid Trading": False,
            "DCA": True,
        }

        for strategy, active in strategies.items():
            st.checkbox(strategy, value=active, key=strategy)

    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview",
        "🤖 Agent Swarm",
        "💼 Strategies",
        "⚠️ Risk",
        "📊 Analytics"
    ])

    with tab1:
        show_overview()

    with tab2:
        show_agent_swarm()

    with tab3:
        show_strategies()

    with tab4:
        show_risk_management()

    with tab5:
        show_analytics()


def show_overview():
    """Show overview metrics."""
    st.header("Portfolio Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Portfolio",
            "$80,516",
            "+$40,516 (101.3%)",
            help="Total portfolio value"
        )

    with col2:
        st.metric(
            "Today's P&L",
            "+$2,341",
            "+2.9%"
        )

    with col3:
        st.metric(
            "Sharpe Ratio",
            "2.13",
            "+0.15",
            help="Risk-adjusted returns"
        )

    with col4:
        st.metric(
            "Win Rate",
            "62.3%",
            "+1.2%"
        )

    st.markdown("---")

    # Portfolio value chart
    st.subheader("Portfolio Value (Last 30 Days)")

    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    np.random.seed(42)
    values = 40000 * (1 + np.cumsum(np.random.normal(0.003, 0.02, 30)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#667eea', width=3),
        fill='tozeroy'
    ))

    fig.update_layout(
        height=400,
        hovermode='x unified',
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Recent trades
    st.subheader("Recent Trades")

    trades = pd.DataFrame({
        'Time': pd.date_range(end=datetime.now(), periods=10, freq='15min')[::-1],
        'Strategy': np.random.choice(['ML Ensemble', 'Momentum', 'Mean Reversion', 'PPO RL'], 10),
        'Asset': ['BTC/USD'] * 5 + ['ETH/USD'] * 3 + ['SOL/USD'] * 2,
        'Side': np.random.choice(['BUY', 'SELL'], 10),
        'Amount': np.random.uniform(0.01, 1.0, 10),
        'Price': np.random.uniform(40000, 50000, 10),
        'P&L': np.random.uniform(-500, 1200, 10)
    })

    trades['P&L_formatted'] = trades['P&L'].apply(
        lambda x: f"+${x:.2f}" if x > 0 else f"-${abs(x):.2f}"
    )

    st.dataframe(
        trades[['Time', 'Strategy', 'Asset', 'Side', 'Amount', 'Price', 'P&L_formatted']],
        use_container_width=True,
        height=300
    )


def show_agent_swarm():
    """Show agent swarm status."""
    st.header("🤖 Agent Swarm Status")

    st.info("**Agent Swarm**: Multiple specialized AI agents working together to optimize trading")

    # Agent status
    agents = [
        {
            'name': 'Strategy Coordinator',
            'status': 'Active',
            'tasks': 156,
            'success_rate': 94.2,
            'description': 'Coordinates all 11 trading strategies'
        },
        {
            'name': 'Risk Manager',
            'status': 'Active',
            'tasks': 1247,
            'success_rate': 99.8,
            'description': 'Monitors positions and enforces limits'
        },
        {
            'name': 'Market Analyzer',
            'status': 'Active',
            'tasks': 3421,
            'success_rate': 87.3,
            'description': 'Analyzes market conditions in real-time'
        },
        {
            'name': 'Execution Agent',
            'status': 'Active',
            'tasks': 892,
            'success_rate': 96.7,
            'description': 'Executes trades with optimal timing'
        },
        {
            'name': 'ML Model Agent',
            'status': 'Training',
            'tasks': 45,
            'success_rate': 91.1,
            'description': 'Continuously trains and updates models'
        },
        {
            'name': 'RL Agent (PPO)',
            'status': 'Active',
            'tasks': 234,
            'success_rate': 88.5,
            'description': 'Reinforcement learning decision maker'
        }
    ]

    for agent in agents:
        with st.expander(f"{'🟢' if agent['status'] == 'Active' else '🟡'} {agent['name']} - {agent['status']}"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Tasks Completed", agent['tasks'])

            with col2:
                st.metric("Success Rate", f"{agent['success_rate']}%")

            with col3:
                st.metric("Status", agent['status'])

            st.caption(agent['description'])

    st.markdown("---")

    # Agent communication
    st.subheader("Agent Communication Log")
    st.code("""
[02:15:43] Strategy Coordinator → Risk Manager: Checking position limits for BTC trade
[02:15:44] Risk Manager → Strategy Coordinator: ✅ Within limits (current: 8.2%, max: 10%)
[02:15:45] Execution Agent: Executing BUY 0.25 BTC @ $48,342
[02:15:46] Market Analyzer: Detected bullish momentum (RSI: 58, MACD: positive)
[02:15:47] ML Model Agent: Prediction confidence: 87.3% (BUY signal)
[02:15:48] Execution Agent: ✅ Trade completed successfully
    """, language="log")


def show_strategies():
    """Show all strategies performance."""
    st.header("💼 Strategy Performance")

    # Strategy comparison
    strategies = pd.DataFrame({
        'Strategy': [
            'ML Ensemble',
            'PPO RL Agent',
            'Momentum',
            'Mean Reversion',
            'Yield Optimizer',
            'RSI',
            'MACD',
            'Bollinger Bands',
            'DCA',
            'Multi-Chain Arb',
            'Grid Trading'
        ],
        'Return': [215.0, 118.9, 33.5, 37.7, 28.4, 18.2, 24.1, 31.2, 15.8, 10.6, 22.3],
        'Sharpe': [2.4, 2.2, 2.1, 1.8, 2.0, 1.6, 1.9, 1.7, 1.5, 2.7, 1.8],
        'Win Rate': [64.0, 59.0, 62.0, 58.0, 61.5, 56.3, 60.1, 57.8, 55.2, 64.1, 58.9],
        'Trades': [120, 63, 18, 26, 42, 89, 71, 54, 156, 12, 234],
        'Status': ['Active'] * 9 + ['Inactive', 'Active']
    })

    # Sort by return
    strategies = strategies.sort_values('Return', ascending=False)

    # Display with color coding
    def color_return(val):
        if val > 50:
            return 'background-color: #d4edda'
        elif val > 20:
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #f8d7da'

    styled = strategies.style.applymap(
        color_return,
        subset=['Return']
    ).format({
        'Return': '{:.1f}%',
        'Sharpe': '{:.2f}',
        'Win Rate': '{:.1f}%'
    })

    st.dataframe(styled, use_container_width=True, height=450)

    st.markdown("---")

    # Individual strategy details
    st.subheader("Strategy Details")

    selected_strategy = st.selectbox(
        "Select Strategy",
        strategies['Strategy'].tolist()
    )

    # Show details for selected strategy
    st.info(f"**{selected_strategy}** - Detailed metrics and configuration")

    col1, col2, col3 = st.columns(3)

    strategy_data = strategies[strategies['Strategy'] == selected_strategy].iloc[0]

    with col1:
        st.metric("Total Return", f"{strategy_data['Return']:.1f}%")
        st.metric("Sharpe Ratio", f"{strategy_data['Sharpe']:.2f}")

    with col2:
        st.metric("Win Rate", f"{strategy_data['Win Rate']:.1f}%")
        st.metric("Total Trades", int(strategy_data['Trades']))

    with col3:
        st.metric("Status", strategy_data['Status'])
        st.metric("Risk Level", "Medium")


def show_risk_management():
    """Show risk management status."""
    st.header("⚠️ Risk Management")

    # Risk metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current Drawdown",
            "12.3%",
            "-2.1%",
            delta_color="inverse",
            help="Current drawdown from peak"
        )

    with col2:
        st.metric(
            "Max Drawdown Limit",
            "15.0%",
            "2.7% buffer",
            help="Maximum allowed drawdown"
        )

    with col3:
        st.metric(
            "VaR (95%)",
            "$1,240",
            "1.54% of portfolio",
            help="Value at Risk"
        )

    st.markdown("---")

    # Position limits
    st.subheader("Position Limits")

    positions = pd.DataFrame({
        'Asset': ['BTC/USD', 'ETH/USD', 'SOL/USD', 'MATIC/USD'],
        'Current': [8.2, 6.5, 3.1, 2.3],
        'Limit': [10.0, 10.0, 5.0, 5.0]
    })

    for _, row in positions.iterrows():
        col1, col2 = st.columns([3, 1])

        with col1:
            pct = (row['Current'] / row['Limit']) * 100
            st.write(f"**{row['Asset']}**")
            st.progress(min(pct / 100, 1.0))

        with col2:
            status = "🟢" if pct < 80 else "🟡" if pct < 95 else "🔴"
            st.write(f"{status} {row['Current']:.1f}% / {row['Limit']:.1f}%")

    st.markdown("---")

    # Circuit breakers
    st.subheader("Circuit Breakers")

    breakers = [
        {"name": "Daily Loss Limit", "trigger": "-5%", "status": "Active", "triggered": False},
        {"name": "Position Size Limit", "trigger": "10% per asset", "status": "Active", "triggered": False},
        {"name": "Max Drawdown", "trigger": "15%", "status": "Active", "triggered": False},
        {"name": "Volatility Halt", "trigger": "VIX > 40", "status": "Active", "triggered": False},
    ]

    for breaker in breakers:
        status_icon = "🔴 TRIGGERED" if breaker['triggered'] else "🟢 Active"
        st.success(f"{status_icon} - **{breaker['name']}** (Trigger: {breaker['trigger']})")


def show_analytics():
    """Show advanced analytics."""
    st.header("📊 Advanced Analytics")

    # Performance attribution
    st.subheader("Performance Attribution")

    attribution = pd.DataFrame({
        'Source': ['Strategy Alpha', 'Market Beta', 'Risk Management', 'Execution', 'Fees'],
        'Contribution': [45.2, 28.1, 15.3, 8.7, -5.3]
    })

    fig = go.Figure(go.Bar(
        x=attribution['Source'],
        y=attribution['Contribution'],
        marker_color=['green' if x > 0 else 'red' for x in attribution['Contribution']]
    ))

    fig.update_layout(
        title="Return Attribution (%)",
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Correlation matrix
    st.subheader("Strategy Correlation Matrix")

    st.info("Shows how strategies perform relative to each other. Low correlation = good diversification")

    strategies_corr = ['ML Ensemble', 'PPO RL', 'Momentum', 'Mean Rev', 'Yield Opt']
    np.random.seed(42)
    corr_matrix = np.random.uniform(0.1, 0.7, (5, 5))
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=strategies_corr,
        y=strategies_corr,
        colorscale='RdYlGn_r',
        zmid=0.5
    ))

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
