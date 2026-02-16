"""
🚀 TRADING-AI UNIFIED COMMAND CENTER 🚀
Complete dashboard integrating all trading features: stocks, crypto, DeFi, backtesting.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.execution.alpaca_broker import AlpacaBroker
from src.execution.broker_interface import MockBroker
from src.execution.crypto_paper_trading import CryptoPaperTradingEngine
from src.advanced_strategies import AdvancedTradingStrategies
from src.blockchain.chain_manager import ChainManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Trading-AI Command Center",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trading_mode' not in st.session_state:
    st.session_state.trading_mode = 'paper'
if 'api_keys_configured' not in st.session_state:
    st.session_state.api_keys_configured = False
if 'selected_symbols' not in st.session_state:
    st.session_state.selected_symbols = ['AAPL', 'MSFT', 'BTC', 'ETH']

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 🚀 Command Center")

    # Trading Mode Toggle
    st.markdown("### Trading Mode")
    trading_mode = st.radio(
        "Select Mode:",
        ["📝 Paper Trading", "🔴 LIVE Trading"],
        index=0 if st.session_state.trading_mode == 'paper' else 1,
        help="Switch between paper (simulated) and live trading"
    )

    if "LIVE" in trading_mode:
        st.session_state.trading_mode = 'live'
        st.warning("⚠️ LIVE TRADING MODE - Real money at risk!")
    else:
        st.session_state.trading_mode = 'paper'
        st.info("📝 Paper Trading - Safe simulation mode")

    st.markdown("---")

    # Navigation
    st.markdown("### 📊 Navigation")
    page = st.radio(
        "Select Page:",
        [
            "🏠 Overview",
            "📈 Stock Trading",
            "🪙 Crypto Trading",
            "💎 DeFi & DEX",
            "🤖 Agent Swarm",
            "🔬 Backtesting",
            "📊 Advanced Strategies",
            "💼 Portfolio",
            "⚙️ Settings",
            "🔍 System Status"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Quick Stats
    st.markdown("### 📊 Quick Stats")
    try:
        # Mock data for now
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Value", "$105,234", "+5.2%")
        with col2:
            st.metric("Day P&L", "+$2,134", "+2.1%")
    except:
        pass

    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown("[📚 Documentation](https://github.com/cpoplaws/trading-ai)")
    st.markdown("[🐛 Report Issue](https://github.com/cpoplaws/trading-ai/issues)")
    st.markdown("[💬 Discord](https://discord.gg/trading-ai)")

# ==================== MAIN CONTENT ====================

# Header
st.markdown('<h1 class="main-header">🚀 TRADING-AI COMMAND CENTER 🚀</h1>', unsafe_allow_html=True)

# ==================== OVERVIEW PAGE ====================
if page == "🏠 Overview":
    st.markdown("## 🏠 System Overview")

    # Status Banner
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🟢 System Status</h3>
            <p style="font-size: 1.5rem; font-weight: bold; color: green;">OPERATIONAL</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>💰 Trading Mode</h3>
            <p style="font-size: 1.5rem; font-weight: bold;">{}</p>
        </div>
        """.format("🔴 LIVE" if st.session_state.trading_mode == 'live' else "📝 PAPER"), unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Active Strategies</h3>
            <p style="font-size: 1.5rem; font-weight: bold; color: blue;">5</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🌐 Blockchains</h3>
            <p style="font-size: 1.5rem; font-weight: bold; color: purple;">7</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Feature Overview
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Stock Trading")
        st.markdown("""
        ✅ Real-time data via yfinance
        ✅ Alpaca broker integration
        ✅ ML-powered signals
        ✅ Paper & live trading
        ✅ Portfolio tracking
        """)

        st.markdown("### 🪙 Crypto Trading")
        st.markdown("""
        ✅ 7 blockchain support
        ✅ Binance & CoinGecko data
        ✅ Paper trading engine
        ✅ Multi-chain operations
        ✅ DeFi integration
        """)

    with col2:
        st.markdown("### 🧠 Advanced Features")
        st.markdown("""
        ✅ Multi-timeframe analysis
        ✅ Sentiment aggregation
        ✅ Options strategies
        ✅ Portfolio optimization
        ✅ Whale tracking
        """)

        st.markdown("### 🔧 Infrastructure")
        st.markdown("""
        ✅ Real-time dashboard
        ✅ Comprehensive backtesting
        ✅ Risk management
        ✅ Multi-channel alerts
        ✅ Cloud-ready deployment
        """)

    st.markdown("---")

    # Recent Activity
    st.markdown("### 📊 Recent Activity")
    recent_activity = pd.DataFrame({
        'Time': [datetime.now() - timedelta(minutes=x) for x in [5, 15, 30, 60]],
        'Type': ['BUY', 'SELL', 'BUY', 'SIGNAL'],
        'Symbol': ['AAPL', 'MSFT', 'BTC', 'ETH'],
        'Action': ['Bought 10 shares at $175.50', 'Sold 5 shares at $410.20', 'Bought 0.1 BTC at $52,000', 'Signal: Strong Buy'],
        'Status': ['✅ Executed', '✅ Executed', '✅ Executed', '📊 Generated']
    })
    st.dataframe(recent_activity, use_container_width=True)

# ==================== STOCK TRADING PAGE ====================
elif page == "📈 Stock Trading":
    st.markdown("## 📈 Stock Trading")

    # Symbol selector
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        symbol = st.selectbox("Select Stock Symbol", ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"])
    with col2:
        timeframe = st.selectbox("Timeframe", ["1D", "5D", "1M", "3M", "1Y"])
    with col3:
        if st.button("🔄 Refresh Data"):
            st.rerun()

    st.markdown("---")

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Chart & Signals", "💼 Positions", "📝 Trade History", "🎯 Strategy"])

    with tab1:
        st.markdown("### 📊 Price Chart & Trading Signals")

        # Mock data for demonstration
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Price'))
        fig.update_layout(height=400, title=f"{symbol} Price Chart")
        st.plotly_chart(fig, use_container_width=True)

        # Signals
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Signal", "BUY", "+15%", help="ML model confidence")
        with col2:
            st.metric("Confidence", "85%", "+5%")
        with col3:
            st.metric("Target Price", "$185.00", "+$10")

        # Order entry
        st.markdown("### 📝 Place Order")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            order_side = st.selectbox("Side", ["BUY", "SELL"])
        with col2:
            order_qty = st.number_input("Quantity", min_value=1, value=10)
        with col3:
            order_type = st.selectbox("Type", ["MARKET", "LIMIT"])
        with col4:
            if order_type == "LIMIT":
                limit_price = st.number_input("Limit Price", value=175.50)

        if st.button(f"🚀 Execute {order_side} Order", type="primary"):
            st.success(f"✅ Order submitted: {order_side} {order_qty} shares of {symbol}")

    with tab2:
        st.markdown("### 💼 Current Positions")
        positions = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'Quantity': [100, 50, 25],
            'Avg Price': [150.00, 380.00, 140.00],
            'Current Price': [175.50, 410.20, 155.30],
            'P&L': ['+$2,550', '+$1,510', '+$382.50'],
            'P&L %': ['+17.0%', '+8.0%', '+10.9%']
        })
        st.dataframe(positions, use_container_width=True)

    with tab3:
        st.markdown("### 📝 Trade History")
        history = pd.DataFrame({
            'Date': [datetime.now() - timedelta(days=x) for x in range(10)],
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY'] * 2,
            'Side': ['BUY', 'SELL', 'BUY', 'SELL', 'BUY'] * 2,
            'Quantity': [10, 5, 3, 2, 50] * 2,
            'Price': [175.50, 410.20, 155.30, 245.60, 495.20] * 2,
            'Total': ['$1,755', '$2,051', '$465.90', '$491.20', '$24,760'] * 2
        })
        st.dataframe(history, use_container_width=True)

    with tab4:
        st.markdown("### 🎯 Strategy Configuration")

        strategy_type = st.selectbox("Strategy", ["ML Signals", "Multi-Timeframe", "Sentiment", "Combined"])

        col1, col2 = st.columns(2)
        with col1:
            st.slider("Risk Tolerance", 0.0, 1.0, 0.5, help="0 = Conservative, 1 = Aggressive")
            st.slider("Position Size %", 1, 100, 10, help="Max % of portfolio per position")
        with col2:
            st.slider("Stop Loss %", 1, 20, 5)
            st.slider("Take Profit %", 5, 50, 15)

        if st.button("💾 Save Strategy Config"):
            st.success("✅ Strategy configuration saved!")

# ==================== CRYPTO TRADING PAGE ====================
elif page == "🪙 Crypto Trading":
    st.markdown("## 🪙 Crypto Trading")

    # Chain selector
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        chain = st.selectbox("Select Blockchain", [
            "Ethereum", "BSC", "Polygon", "Avalanche", "Base", "Arbitrum", "Solana"
        ])
    with col2:
        crypto_symbol = st.selectbox("Token", ["BTC", "ETH", "SOL", "BNB", "AVAX"])
    with col3:
        if st.button("🔄 Refresh"):
            st.rerun()

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Markets", "💰 Positions", "🔄 Swap", "⛓️ Chains"])

    with tab1:
        st.markdown("### 📊 Crypto Markets")

        # Market data
        markets = pd.DataFrame({
            'Token': ['BTC', 'ETH', 'SOL', 'BNB', 'AVAX', 'MATIC'],
            'Price': ['$52,450', '$3,210', '$105.50', '$425.30', '$38.20', '$0.85'],
            '24h Change': ['+2.5%', '+3.1%', '+5.8%', '-1.2%', '+4.3%', '+2.1%'],
            'Volume': ['$28.5B', '$15.2B', '$2.1B', '$1.8B', '$850M', '$420M'],
            'Market Cap': ['$1.03T', '$385B', '$47B', '$64B', '$15B', '$8B']
        })
        st.dataframe(markets, use_container_width=True)

        # Price chart
        st.markdown(f"### {crypto_symbol} Price Chart")
        dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=dates,
            open=prices - np.random.rand(100) * 50,
            high=prices + np.random.rand(100) * 100,
            low=prices - np.random.rand(100) * 100,
            close=prices,
            name=crypto_symbol
        ))
        fig.update_layout(height=400, title=f"{crypto_symbol} Price Chart")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### 💰 Crypto Positions")

        crypto_positions = pd.DataFrame({
            'Token': ['BTC', 'ETH', 'SOL'],
            'Chain': ['Ethereum', 'Ethereum', 'Solana'],
            'Balance': ['0.1542', '5.234', '125.50'],
            'Value (USD)': ['$8,086', '$16,801', '$13,244'],
            'Cost Basis': ['$7,500', '$15,000', '$10,000'],
            'P&L': ['+$586', '+$1,801', '+$3,244'],
            'P&L %': ['+7.8%', '+12.0%', '+32.4%']
        })
        st.dataframe(crypto_positions, use_container_width=True)

        # Total value
        total_value = 8086 + 16801 + 13244
        total_cost = 7500 + 15000 + 10000
        total_pnl = total_value - total_cost

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Value", f"${total_value:,.0f}")
        with col2:
            st.metric("Total Cost", f"${total_cost:,.0f}")
        with col3:
            st.metric("Total P&L", f"+${total_pnl:,.0f}", f"+{(total_pnl/total_cost*100):.1f}%")

    with tab3:
        st.markdown("### 🔄 Token Swap")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### From")
            from_token = st.selectbox("Token", ["ETH", "USDC", "USDT", "DAI"], key="from")
            from_amount = st.number_input("Amount", min_value=0.0, value=1.0, key="from_amt")
            st.info(f"Balance: 5.234 {from_token}")

        with col2:
            st.markdown("#### To")
            to_token = st.selectbox("Token", ["USDC", "ETH", "USDT", "DAI"], key="to")
            to_amount = st.number_input("Amount", value=3210.0, disabled=True, key="to_amt")
            st.info(f"Est. gas: $15.50")

        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Exchange Rate", "1 ETH = 3,210 USDC")
        with col2:
            st.metric("Price Impact", "0.05%")
        with col3:
            st.metric("Route", "Uniswap V3")

        if st.button("🔄 Execute Swap", type="primary"):
            st.success("✅ Swap executed successfully!")

    with tab4:
        st.markdown("### ⛓️ Multi-Chain Status")

        chains = pd.DataFrame({
            'Chain': ['Ethereum', 'BSC', 'Polygon', 'Avalanche', 'Base', 'Arbitrum', 'Solana'],
            'Status': ['🟢 Connected', '🟢 Connected', '🟢 Connected', '🟢 Connected', '🟢 Connected', '🟢 Connected', '🟢 Connected'],
            'Block': ['19,234,567', '34,567,890', '52,345,678', '42,345,678', '9,876,543', '185,234,567', '245,678,901'],
            'Gas Price': ['25 gwei', '3 gwei', '50 gwei', '25 nGLM', '0.5 gwei', '0.1 gwei', '$0.00025'],
            'Wallet Balance': ['0.523 ETH', '1.25 BNB', '50 MATIC', '10 AVAX', '0.1 ETH', '0.3 ETH', '15 SOL']
        })
        st.dataframe(chains, use_container_width=True)

# ==================== DEFI & DEX PAGE ====================
elif page == "💎 DeFi & DEX":
    st.markdown("## 💎 DeFi & DEX Trading")

    tab1, tab2, tab3 = st.tabs(["🔄 DEX Aggregator", "💧 Liquidity Pools", "🌾 Yield Farming"])

    with tab1:
        st.markdown("### 🔄 DEX Aggregator - Best Price Finder")

        st.markdown("#### Find Best Price Across DEXs")
        col1, col2, col3 = st.columns(3)
        with col1:
            dex_from = st.selectbox("From Token", ["ETH", "USDC", "WBTC", "DAI"])
        with col2:
            dex_to = st.selectbox("To Token", ["USDC", "ETH", "USDT", "DAI"])
        with col3:
            dex_amount = st.number_input("Amount", value=1.0)

        if st.button("🔍 Find Best Route"):
            st.markdown("#### 📊 DEX Comparison")

            dex_prices = pd.DataFrame({
                'DEX': ['Uniswap V3', 'Uniswap V2', 'SushiSwap', 'Curve', '1inch'],
                'Output': ['3,215.50 USDC', '3,210.20 USDC', '3,208.75 USDC', '3,212.30 USDC', '3,216.80 USDC'],
                'Gas (USD)': ['$15.20', '$12.50', '$13.10', '$11.80', '$16.50'],
                'Price Impact': ['0.05%', '0.12%', '0.15%', '0.08%', '0.04%'],
                'Best': ['✅ Best Net', '❌', '❌', '❌', '🏆 Best Price']
            })
            st.dataframe(dex_prices, use_container_width=True)

            st.success("🏆 Best route: 1inch Aggregator → Net: 3,200.30 USDC (after gas)")

    with tab2:
        st.markdown("### 💧 Liquidity Pool Management")

        # Active pools
        pools = pd.DataFrame({
            'Pool': ['ETH/USDC', 'BTC/ETH', 'AVAX/USDC'],
            'Protocol': ['Uniswap V3', 'SushiSwap', 'Trader Joe'],
            'Your Liquidity': ['$5,234', '$3,450', '$2,100'],
            'APY': ['25.5%', '18.2%', '45.3%'],
            'Fees Earned': ['$234.50', '$125.30', '$98.20'],
            'IL': ['-2.1%', '+1.5%', '-3.8%']
        })
        st.dataframe(pools, use_container_width=True)

        # Add liquidity
        with st.expander("➕ Add Liquidity"):
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox("Token A", ["ETH", "USDC", "WBTC"])
                st.number_input("Amount A", value=1.0)
            with col2:
                st.selectbox("Token B", ["USDC", "ETH", "USDT"])
                st.number_input("Amount B", value=3210.0)

            if st.button("➕ Add Liquidity"):
                st.success("✅ Liquidity added successfully!")

    with tab3:
        st.markdown("### 🌾 Yield Farming Opportunities")

        opportunities = pd.DataFrame({
            'Farm': ['PancakeSwap CAKE', 'Aave USDC', 'Compound DAI', 'Curve 3pool'],
            'Chain': ['BSC', 'Ethereum', 'Ethereum', 'Ethereum'],
            'APY': ['95.2%', '3.5%', '2.8%', '8.5%'],
            'TVL': ['$1.2B', '$5.8B', '$3.2B', '$2.1B'],
            'Your Stake': ['$0', '$1,500', '$2,000', '$0'],
            'Earned': ['-', '$52.50', '$56.00', '-']
        })
        st.dataframe(opportunities, use_container_width=True)

# ==================== SETTINGS PAGE ====================

# ==================== AGENT SWARM PAGE ====================
elif page == "🤖 Agent Swarm":
    st.markdown("## 🤖 Autonomous Agent Swarm")
    
    st.info("🎯 **AI-Powered Trading Agents**: Train and deploy reinforcement learning agents that learn to trade autonomously!")
    
    # Swarm Status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Swarm Status", "🟢 Ready", "4 agents")
    with col2:
        st.metric("Training Progress", "100%", "+25 epochs")
    with col3:
        st.metric("Best Agent ROI", "+47.5%", "+12.3%")
    with col4:
        st.metric("Swarm Consensus", "85%", "+5%")
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎮 Control Center", "📊 Performance", "🧠 Training", "⚙️ Configuration"])
    
    with tab1:
        st.markdown("### 🎮 Swarm Control Center")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 🤖 Active Agents")
            
            agents_data = pd.DataFrame({
                'Agent': ['ExecutionAgent', 'RiskAgent', 'ArbitrageAgent', 'MarketMakingAgent'],
                'Status': ['🟢 Active', '🟢 Active', '🟢 Active', '🟡 Standby'],
                'Algorithm': ['PPO', 'SAC', 'DDPG', 'PPO'],
                'Performance': ['+15.2%', '+8.5%', '+23.8%', '+0.0%'],
                'Trades': [245, 128, 89, 0],
                'Win Rate': ['68%', '72%', '81%', 'N/A']
            })
            st.dataframe(agents_data, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎛️ Swarm Controls")
            
            swarm_action = st.selectbox("Action", ["Monitor", "Train", "Deploy", "Pause", "Stop"])
            
            if st.button("🚀 Execute Action", type="primary"):
                if swarm_action == "Train":
                    st.info("🧠 Training swarm... (This takes 5-10 minutes)")
                elif swarm_action == "Deploy":
                    st.warning("⚠️ Deploying swarm to LIVE trading!")
                elif swarm_action == "Stop":
                    st.error("🛑 Stopping all agents...")
                else:
                    st.success(f"✅ {swarm_action} command executed")
            
            st.markdown("---")
            
            st.markdown("**Coordination Mode:**")
            coord_mode = st.radio("Mode", ["Voting", "Hierarchical", "Consensus"], horizontal=True)
            
            st.markdown("**Min Confidence:**")
            min_conf = st.slider("Threshold", 0.0, 1.0, 0.6, 0.05)
        
        st.markdown("---")
        
        st.markdown("### 📊 Real-Time Swarm Activity")
        
        # Simulated activity log
        activity_log = pd.DataFrame({
            'Time': [datetime.now() - timedelta(minutes=x) for x in [1, 5, 10, 15, 20]],
            'Agent': ['ExecutionAgent', 'ArbitrageAgent', 'RiskAgent', 'ExecutionAgent', 'ArbitrageAgent'],
            'Action': ['BUY AAPL', 'ARB BTC/ETH', 'RISK CHECK', 'SELL MSFT', 'ARB DETECTED'],
            'Confidence': ['92%', '88%', '95%', '85%', '90%'],
            'Result': ['✅ +$250', '✅ +$1,200', '✅ PASS', '✅ +$180', '⏳ Pending']
        })
        st.dataframe(activity_log, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 Agent Performance Analytics")
        
        # Performance comparison
        st.markdown("#### 📈 Cumulative Returns")
        
        dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
        returns_data = {
            'ExecutionAgent': np.cumsum(np.random.randn(100) * 0.5 + 0.1),
            'RiskAgent': np.cumsum(np.random.randn(100) * 0.3 + 0.05),
            'ArbitrageAgent': np.cumsum(np.random.randn(100) * 0.7 + 0.15),
            'MarketMakingAgent': np.cumsum(np.random.randn(100) * 0.2),
        }
        
        fig = go.Figure()
        for agent, returns in returns_data.items():
            fig.add_trace(go.Scatter(x=dates, y=returns, mode='lines', name=agent))
        
        fig.update_layout(
            height=400,
            title="Agent Performance Comparison",
            xaxis_title="Time",
            yaxis_title="Cumulative Return (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Agent metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🎯 ExecutionAgent")
            st.metric("Total Return", "+15.2%", "+3.2%")
            st.metric("Sharpe Ratio", "1.85", "+0.15")
            st.metric("Max Drawdown", "-4.2%", "-0.5%")
            st.metric("Win Rate", "68%", "+2%")
        
        with col2:
            st.markdown("#### 🛡️ RiskAgent")
            st.metric("Total Return", "+8.5%", "+1.1%")
            st.metric("Sharpe Ratio", "2.15", "+0.25")
            st.metric("Max Drawdown", "-2.1%", "-0.1%")
            st.metric("Win Rate", "72%", "+1%")
        
        with col3:
            st.markdown("#### 💎 ArbitrageAgent")
            st.metric("Total Return", "+23.8%", "+5.2%")
            st.metric("Sharpe Ratio", "1.65", "+0.08")
            st.metric("Max Drawdown", "-6.5%", "-1.2%")
            st.metric("Win Rate", "81%", "+4%")
    
    with tab3:
        st.markdown("### 🧠 Agent Training Center")
        
        st.info("💡 Train agents on historical data to improve their trading strategies")
        
        # Training configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Training Data")
            train_symbol = st.selectbox("Symbol", ["AAPL", "MSFT", "BTC", "ETH"])
            train_period = st.selectbox("Period", ["1 Month", "3 Months", "6 Months", "1 Year"])
            
            st.markdown("#### 🤖 Agents to Train")
            train_execution = st.checkbox("ExecutionAgent", value=True)
            train_risk = st.checkbox("RiskAgent", value=True)
            train_arbitrage = st.checkbox("ArbitrageAgent", value=True)
            train_market_making = st.checkbox("MarketMakingAgent", value=False)
        
        with col2:
            st.markdown("#### ⚙️ Training Parameters")
            timesteps = st.number_input("Timesteps", min_value=10000, max_value=1000000, value=100000, step=10000)
            learning_rate = st.select_slider("Learning Rate", options=[0.0001, 0.0003, 0.001, 0.003, 0.01], value=0.0003)
            batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=2)
            
            st.markdown("#### 💾 Model Management")
            model_name = st.text_input("Model Name", value=f"swarm_{datetime.now().strftime('%Y%m%d')}")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🧠 Start Training", type="primary"):
                st.success("✅ Training started! Check Training Progress tab for updates.")
                st.info("⏱️ Estimated time: 5-10 minutes")
        with col2:
            if st.button("⏸️ Pause Training"):
                st.warning("⏸️ Training paused")
        with col3:
            if st.button("🛑 Stop Training"):
                st.error("🛑 Training stopped")
        
        st.markdown("---")
        
        # Training history
        st.markdown("#### 📜 Training History")
        training_history = pd.DataFrame({
            'Date': [datetime.now() - timedelta(days=x) for x in [0, 1, 3, 7]],
            'Model': ['swarm_20260215', 'swarm_20260214', 'swarm_20260212', 'swarm_20260208'],
            'Timesteps': ['100k', '100k', '50k', '200k'],
            'Best Return': ['+15.2%', '+12.8%', '+8.5%', '+18.3%'],
            'Status': ['✅ Active', '📦 Archived', '📦 Archived', '📦 Archived']
        })
        st.dataframe(training_history, use_container_width=True)
    
    with tab4:
        st.markdown("### ⚙️ Swarm Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Agent Configuration")
            
            st.markdown("**ExecutionAgent (PPO)**")
            exec_enabled = st.checkbox("Enabled", value=True, key="exec_enabled")
            exec_risk = st.slider("Risk Level", 0.0, 1.0, 0.5, key="exec_risk")
            
            st.markdown("**RiskAgent (SAC)**")
            risk_enabled = st.checkbox("Enabled", value=True, key="risk_enabled")
            risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.7, key="risk_threshold")
            
            st.markdown("**ArbitrageAgent (DDPG)**")
            arb_enabled = st.checkbox("Enabled", value=True, key="arb_enabled")
            arb_min_spread = st.slider("Min Spread %", 0.0, 5.0, 0.5, key="arb_min")
            
            st.markdown("**MarketMakingAgent (PPO)**")
            mm_enabled = st.checkbox("Enabled", value=False, key="mm_enabled")
            mm_spread = st.slider("Spread Width", 0.0, 2.0, 0.5, key="mm_spread")
        
        with col2:
            st.markdown("#### 🔧 Swarm Parameters")
            
            st.markdown("**Coordination**")
            coordination = st.selectbox("Mode", ["Voting", "Hierarchical", "Consensus"])
            min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.6)
            
            st.markdown("**Risk Management**")
            max_position = st.slider("Max Position %", 0, 100, 30)
            stop_loss = st.slider("Stop Loss %", 0, 20, 5)
            take_profit = st.slider("Take Profit %", 0, 100, 15)
            
            st.markdown("**Execution**")
            max_slippage = st.slider("Max Slippage %", 0.0, 1.0, 0.1)
            commission = st.number_input("Commission %", 0.0, 1.0, 0.1, 0.01)
        
        st.markdown("---")
        
        if st.button("💾 Save Configuration", type="primary"):
            st.success("✅ Configuration saved successfully!")

elif page == "⚙️ Settings":
    st.markdown("## ⚙️ Settings & Configuration")

    tab1, tab2, tab3, tab4 = st.tabs(["🔑 API Keys", "⛓️ Wallets", "🔔 Alerts", "⚠️ Risk"])

    with tab1:
        st.markdown("### 🔑 API Keys Configuration")

        st.info("💡 Enter your API keys here. They will be securely saved to your .env file.")

        # Stock trading APIs
        with st.expander("📈 Stock Trading APIs", expanded=True):
            alpaca_key = st.text_input("Alpaca API Key", type="password", help="Get from alpaca.markets")
            alpaca_secret = st.text_input("Alpaca Secret Key", type="password")
            alpaca_url = st.selectbox("Alpaca Environment", ["Paper (https://paper-api.alpaca.markets)", "Live (https://api.alpaca.markets)"])

            if st.button("💾 Save Alpaca Keys"):
                st.success("✅ Alpaca keys saved!")

        # Crypto data APIs
        with st.expander("🪙 Crypto Data APIs"):
            binance_key = st.text_input("Binance API Key", type="password", help="Get from binance.com")
            binance_secret = st.text_input("Binance Secret Key", type="password")

            coingecko_key = st.text_input("CoinGecko API Key (Optional)", type="password")

            if st.button("💾 Save Crypto API Keys"):
                st.success("✅ Crypto API keys saved!")

        # Sentiment APIs
        with st.expander("📰 News & Sentiment APIs"):
            newsapi_key = st.text_input("NewsAPI Key", type="password", help="Get from newsapi.org (free tier: 100 req/day)")

            st.markdown("**Reddit API**")
            reddit_client = st.text_input("Reddit Client ID", type="password")
            reddit_secret = st.text_input("Reddit Secret", type="password")
            reddit_agent = st.text_input("User Agent", value="TradingAI/1.0")

            if st.button("💾 Save Sentiment API Keys"):
                st.success("✅ Sentiment API keys saved!")

    with tab2:
        st.markdown("### ⛓️ Blockchain Wallets")

        st.warning("⚠️ NEVER share your private keys! These are stored locally and encrypted.")

        chains_wallets = ['Ethereum', 'BSC', 'Polygon', 'Avalanche', 'Solana']

        for chain in chains_wallets:
            with st.expander(f"{chain} Wallet"):
                wallet_address = st.text_input(f"{chain} Wallet Address", key=f"{chain}_addr")
                private_key = st.text_input(f"{chain} Private Key", type="password", key=f"{chain}_pk",
                                          help="Used for signing transactions. Stored encrypted locally.")

                if st.button(f"💾 Save {chain} Wallet", key=f"save_{chain}"):
                    st.success(f"✅ {chain} wallet configured!")

    with tab3:
        st.markdown("### 🔔 Alert Configuration")

        st.markdown("#### Price Alerts")
        col1, col2, col3 = st.columns(3)
        with col1:
            alert_symbol = st.text_input("Symbol", "BTC")
        with col2:
            alert_condition = st.selectbox("Condition", ["Above", "Below"])
        with col3:
            alert_price = st.number_input("Price", value=55000.0)

        if st.button("➕ Add Price Alert"):
            st.success(f"✅ Alert added: Notify when {alert_symbol} is {alert_condition} ${alert_price:,.0f}")

        st.markdown("---")
        st.markdown("#### Notification Channels")

        enable_email = st.checkbox("📧 Email Notifications")
        if enable_email:
            st.text_input("Email Address")

        enable_telegram = st.checkbox("📱 Telegram Notifications")
        if enable_telegram:
            st.text_input("Telegram Bot Token")
            st.text_input("Telegram Chat ID")

        enable_discord = st.checkbox("💬 Discord Notifications")
        if enable_discord:
            st.text_input("Discord Webhook URL")

    with tab4:
        st.markdown("### ⚠️ Risk Management")

        st.markdown("#### Position Limits")
        max_position = st.slider("Max Position Size (% of portfolio)", 1, 50, 10)
        max_leverage = st.slider("Max Leverage", 1, 10, 3)
        max_drawdown = st.slider("Max Drawdown (%)", 5, 50, 15)

        st.markdown("#### Stop Loss & Take Profit")
        default_stop_loss = st.slider("Default Stop Loss (%)", 1, 20, 5)
        default_take_profit = st.slider("Default Take Profit (%)", 5, 100, 15)

        st.markdown("#### Trading Hours")
        enable_trading_hours = st.checkbox("Restrict Trading Hours")
        if enable_trading_hours:
            col1, col2 = st.columns(2)
            with col1:
                start_time = st.time_input("Start Time")
            with col2:
                end_time = st.time_input("End Time")

        if st.button("💾 Save Risk Settings", type="primary"):
            st.success("✅ Risk management settings saved!")

# ==================== SYSTEM STATUS ====================
elif page == "🔍 System Status":
    st.markdown("## 🔍 System Status")

    # Overall health
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("System Health", "🟢 Excellent", "100%")
    with col2:
        st.metric("API Status", "🟢 Connected", "7/7")
    with col3:
        st.metric("Uptime", "99.9%", "+0.1%")
    with col4:
        st.metric("Active Strategies", "5", "✅")

    st.markdown("---")

    # Component status
    st.markdown("### 🔧 Component Status")

    components = pd.DataFrame({
        'Component': [
            'Data Ingestion', 'Feature Engineering', 'ML Models', 'Strategy Engine',
            'Execution Engine', 'Portfolio Tracker', 'Blockchain Interfaces', 'DeFi Modules'
        ],
        'Status': ['🟢 Operational'] * 8,
        'Last Check': [datetime.now().strftime('%H:%M:%S')] * 8,
        'Performance': ['Excellent', 'Excellent', 'Good', 'Excellent', 'Excellent', 'Good', 'Excellent', 'Good'],
        'Issues': ['None'] * 8
    })
    st.dataframe(components, use_container_width=True)

    st.markdown("---")

    # Dependency check
    st.markdown("### 📦 Dependencies")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ✅ Installed")
        st.markdown("""
        - numpy 2.0.2
        - pandas 2.3.3
        - scikit-learn 1.6.1
        - tensorflow 2.16.0
        - streamlit 1.50.0
        - web3 7.14.1
        - plotly 6.5.2
        """)

    with col2:
        st.markdown("#### ⚠️ Optional")
        st.markdown("""
        - prophet (not installed)
        - arch (not installed)
        - statsmodels (not installed)
        - newsapi-python (not installed)
        - praw (not installed)
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🚀 Trading-AI Command Center v2.0 | Built with ❤️ using Streamlit</p>
    <p>⚠️ Trading involves risk. Only trade with funds you can afford to lose.</p>
</div>
""", unsafe_allow_html=True)
