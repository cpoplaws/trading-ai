"""
Real-time Trading Dashboard using Streamlit.
Displays portfolio performance, signals, backtests, and live trading data.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.execution.broker_interface import AlpacaBroker, MockBroker
from src.advanced_strategies import AdvancedTradingStrategies

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Trading-AI Command Center",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive {
        color: #00ff00;
    }
    .negative {
        color: #ff0000;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def load_signals(symbol: str) -> pd.DataFrame:
    """Load trading signals from CSV."""
    try:
        signal_path = f"signals/{symbol}_signals.csv"
        if os.path.exists(signal_path):
            df = pd.read_csv(signal_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading signals for {symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_backtest_report(symbol: str) -> str:
    """Load backtest report from text file."""
    try:
        report_path = f"backtests/{symbol}_backtest_report.txt"
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                return f.read()
        return f"No backtest report found for {symbol}"
    except Exception as e:
        logger.error(f"Error loading backtest for {symbol}: {e}")
        return f"Error loading backtest: {str(e)}"

@st.cache_data(ttl=300)
def load_processed_data(symbol: str) -> pd.DataFrame:
    """Load processed market data."""
    try:
        data_path = f"data/processed/{symbol}.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        return pd.DataFrame()

def get_broker_connection():
    """Get broker connection (Alpaca or Mock)."""
    try:
        broker = AlpacaBroker(paper_trading=True)
        # Test connection
        account = broker.get_account()
        if account:
            return broker, "Alpaca (Paper Trading)"
    except Exception as e:
        logger.warning(f"Alpaca connection failed, using MockBroker: {e}")
    
    return MockBroker(), "Mock Broker (Demo Mode)"

def plot_price_chart(df: pd.DataFrame, symbol: str):
    """Create interactive price chart with signals."""
    if df.empty:
        st.warning(f"No data available for {symbol}")
        return
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{symbol} Price & Signals', 'RSI', 'Volume')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add moving averages if available
    if 'sma_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['sma_20'], 
                      name='SMA 20', line=dict(color='orange', width=1)),
            row=1, col=1
        )
    
    if 'sma_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['sma_50'], 
                      name='SMA 50', line=dict(color='blue', width=1)),
            row=1, col=1
        )
    
    # Add buy/sell signals if available
    if 'signal' in df.columns:
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]
        
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals['date'],
                    y=buy_signals['close'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(symbol='triangle-up', size=15, color='green')
                ),
                row=1, col=1
            )
        
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals['date'],
                    y=sell_signals['close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(symbol='triangle-down', size=15, color='red')
                ),
                row=1, col=1
            )
    
    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['rsi'], name='RSI', 
                      line=dict(color='purple')),
            row=2, col=1
        )
        # Overbought/Oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Volume
    if 'volume' in df.columns:
        colors = ['red' if close_price < open_price else 'green' 
                 for close_price, open_price in zip(df['close'], df['open'])]
        fig.add_trace(
            go.Bar(x=df['date'], y=df['volume'], name='Volume', 
                  marker_color=colors),
            row=3, col=1
        )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def plot_backtest_performance(df: pd.DataFrame):
    """Plot backtest equity curve and drawdown."""
    if df.empty or 'portfolio_value' not in df.columns:
        st.warning("No backtest performance data available")
        return
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Portfolio Value', 'Drawdown %')
    )
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['portfolio_value'], 
                  name='Portfolio Value', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Calculate drawdown if available
    if 'drawdown' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['drawdown'] * 100, 
                      name='Drawdown', fill='tozeroy', 
                      line=dict(color='red', width=1)),
            row=2, col=1
        )
    
    fig.update_layout(height=600, showlegend=True, hovermode='x unified')
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def dashboard_overview():
    """Main dashboard overview page."""
    st.title("📈 Trading-AI Command Center")
    st.markdown("---")
    
    # Broker connection status
    broker, broker_type = get_broker_connection()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🔌 Broker Status")
        if "Mock" in broker_type:
            st.warning(f"**{broker_type}** - Configure Alpaca API keys for live connection")
        else:
            st.success(f"**{broker_type}** - Connected ✓")
    
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Get account info
    st.markdown("---")
    st.subheader("💼 Portfolio Overview")
    
    try:
        account = broker.get_account()
        positions = broker.get_positions()
        
        # Display account metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            equity = float(account.get('equity', 100000))
            st.metric("Total Equity", f"${equity:,.2f}")
        
        with col2:
            cash = float(account.get('cash', 100000))
            st.metric("Cash Available", f"${cash:,.2f}")
        
        with col3:
            buying_power = float(account.get('buying_power', 100000))
            st.metric("Buying Power", f"${buying_power:,.2f}")
        
        with col4:
            num_positions = len(positions)
            st.metric("Open Positions", num_positions)
        
        # Display positions table
        if positions:
            st.markdown("### Current Positions")
            positions_data = []
            for pos in positions:
                positions_data.append({
                    'Symbol': pos['symbol'],
                    'Qty': pos['qty'],
                    'Entry Price': f"${pos['avg_entry_price']:.2f}",
                    'Current Price': f"${pos['current_price']:.2f}",
                    'Market Value': f"${pos['market_value']:.2f}",
                    'P&L': f"${pos['unrealized_pl']:.2f}",
                    'P&L %': f"{pos['unrealized_plpc']:.2%}"
                })
            
            st.dataframe(
                pd.DataFrame(positions_data),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No open positions")
    
    except Exception as e:
        st.error(f"Error loading portfolio data: {str(e)}")

def signals_analysis():
    """Signals analysis page."""
    st.title("🎯 Trading Signals Analysis")
    st.markdown("---")
    
    # Symbol selector
    available_symbols = ['AAPL', 'MSFT', 'SPY']
    symbol = st.selectbox("Select Symbol", available_symbols)
    
    # Load data
    signals_df = load_signals(symbol)
    data_df = load_processed_data(symbol)
    
    if not signals_df.empty:
        # Recent signals summary
        col1, col2, col3, col4 = st.columns(4)
        
        recent_signals = signals_df.tail(50)
        buy_signals = len(recent_signals[recent_signals['signal'] == 1])
        sell_signals = len(recent_signals[recent_signals['signal'] == -1])
        hold_signals = len(recent_signals[recent_signals['signal'] == 0])
        
        with col1:
            st.metric("Buy Signals (Last 50)", buy_signals)
        with col2:
            st.metric("Sell Signals (Last 50)", sell_signals)
        with col3:
            st.metric("Hold Signals (Last 50)", hold_signals)
        with col4:
            if 'confidence' in signals_df.columns:
                avg_confidence = signals_df.tail(50)['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
        
        # Plot price chart with signals
        st.markdown("### Price Chart with Signals")
        if not data_df.empty:
            # Merge signals with price data
            merged_df = data_df.merge(
                signals_df[['date', 'signal', 'confidence']], 
                on='date', 
                how='left'
            )
            plot_price_chart(merged_df, symbol)
        
        # Recent signals table
        st.markdown("### Recent Signals")
        display_signals = signals_df.tail(20).copy()
        display_signals['signal_type'] = display_signals['signal'].map({
            1: '🟢 BUY', -1: '🔴 SELL', 0: '⚪ HOLD'
        })
        
        st.dataframe(
            display_signals[['date', 'signal_type', 'confidence', 'close']].sort_values('date', ascending=False),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(f"No signals data available for {symbol}")

def backtest_results():
    """Backtest results page."""
    st.title("📊 Backtest Performance")
    st.markdown("---")
    
    # Symbol selector
    available_symbols = ['AAPL', 'MSFT', 'SPY']
    symbol = st.selectbox("Select Symbol", available_symbols)
    
    # Load and display backtest report
    report = load_backtest_report(symbol)
    
    st.markdown("### Performance Report")
    st.text(report)
    
    # Load data for visualization
    signals_df = load_signals(symbol)
    if not signals_df.empty and 'portfolio_value' in signals_df.columns:
        st.markdown("### Equity Curve")
        plot_backtest_performance(signals_df)

def advanced_strategies():
    """Advanced strategies analysis page."""
    st.title("🧠 Advanced Strategies")
    st.markdown("---")
    
    st.info("Loading advanced strategy analysis...")
    
    # Symbol selector
    available_symbols = ['AAPL', 'MSFT', 'SPY']
    symbols = st.multiselect("Select Symbols", available_symbols, default=['AAPL'])
    
    if not symbols:
        st.warning("Please select at least one symbol")
        return
    
    try:
        # Initialize advanced trading system
        trading_system = AdvancedTradingStrategies(symbols=symbols)
        
        # Load market data for selected symbols
        market_data = {}
        current_prices = {}
        
        for symbol in symbols:
            data_df = load_processed_data(symbol)
            if not data_df.empty:
                market_data[symbol] = {'1d': data_df}
                current_prices[symbol] = data_df['close'].iloc[-1]
        
        if not market_data:
            st.warning("No market data available for selected symbols")
            return
        
        # Get comprehensive analysis
        st.markdown("### Portfolio Dashboard")
        dashboard = trading_system.get_portfolio_dashboard(market_data, current_prices)
        
        if 'error' in dashboard:
            st.error(f"Error generating dashboard: {dashboard['error']}")
            return
        
        # Display portfolio summary
        if dashboard.get('portfolio_summary'):
            summary = dashboard['portfolio_summary']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Buy Signals", summary.get('buy_signals', 0))
            with col2:
                st.metric("Sell Signals", summary.get('sell_signals', 0))
            with col3:
                st.metric("Hold Signals", summary.get('hold_signals', 0))
            with col4:
                st.metric("Avg Confidence", f"{summary.get('average_confidence', 0):.1%}")
            
            # Display top opportunities
            if summary.get('top_opportunities'):
                st.markdown("#### 📋 Top Trading Opportunities")
                for i, opp in enumerate(summary['top_opportunities'][:5], 1):
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    with col1:
                        st.write(f"**{i}. {opp['symbol']}**")
                    with col2:
                        action_color = "green" if opp['signal'] == 'BUY' else "red"
                        st.markdown(f"<span style='color:{action_color}'><b>{opp['signal']}</b></span>", 
                                  unsafe_allow_html=True)
                    with col3:
                        st.write(f"Conf: {opp['confidence']:.1%}")
                    with col4:
                        st.write(f"Return: {opp['expected_return']:.2%}")
                st.markdown("---")
            
            # Risk assessment
            st.markdown("#### ⚠️ Portfolio Risk Assessment")
            st.write(f"**Risk Level:** {summary.get('risk_assessment', 'Unknown')}")
        
        # Display individual symbol signals
        st.markdown("### Individual Symbol Analysis")
        for symbol in symbols:
            symbol_signals = dashboard.get('symbol_signals', {}).get(symbol, {})
            
            if 'error' in symbol_signals:
                st.error(f"{symbol}: {symbol_signals['error']}")
                continue
            
            with st.expander(f"📊 {symbol} Detailed Signals"):
                agg_signal = symbol_signals.get('aggregated_signal', {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Aggregated Signal**")
                    st.write(f"Action: **{agg_signal.get('signal', 'N/A')}**")
                    st.write(f"Confidence: {agg_signal.get('confidence', 0):.2%}")
                    st.write(f"Expected Return: {agg_signal.get('expected_return', 0):.2%}")
                
                with col2:
                    st.markdown("**Strategy Breakdown**")
                    individual_signals = symbol_signals.get('individual_signals', {})
                    for strategy, signal_data in individual_signals.items():
                        if isinstance(signal_data, dict):
                            signal_val = signal_data.get('signal', 'N/A')
                            st.write(f"{strategy}: {signal_val}")
    
    except Exception as e:
        st.error(f"Error loading advanced strategies: {str(e)}")
        logger.error(f"Advanced strategies error: {e}", exc_info=True)

def system_status():
    """System status and configuration page."""
    st.title("⚙️ System Status")
    st.markdown("---")
    
    # Check data availability
    st.subheader("📁 Data Availability")
    
    symbols = ['AAPL', 'MSFT', 'SPY']
    data_status = []
    
    for symbol in symbols:
        raw_exists = os.path.exists(f"data/raw/{symbol}.csv")
        processed_exists = os.path.exists(f"data/processed/{symbol}.csv")
        signals_exists = os.path.exists(f"signals/{symbol}_signals.csv")
        model_exists = os.path.exists(f"models/model_{symbol}.joblib")
        backtest_exists = os.path.exists(f"backtests/{symbol}_backtest_report.txt")
        
        data_status.append({
            'Symbol': symbol,
            'Raw Data': '✓' if raw_exists else '✗',
            'Processed Data': '✓' if processed_exists else '✗',
            'Signals': '✓' if signals_exists else '✗',
            'Model': '✓' if model_exists else '✗',
            'Backtest': '✓' if backtest_exists else '✗'
        })
    
    st.dataframe(pd.DataFrame(data_status), hide_index=True, use_container_width=True)
    
    # System configuration
    st.markdown("---")
    st.subheader("🔧 Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Environment Variables**")
        env_vars = {
            'ALPACA_API_KEY': '✓ Set' if os.getenv('ALPACA_API_KEY') else '✗ Not Set',
            'NEWS_API_KEY': '✓ Set' if os.getenv('NEWS_API_KEY') else '✗ Not Set',
            'REDDIT_CLIENT_ID': '✓ Set' if os.getenv('REDDIT_CLIENT_ID') else '✗ Not Set',
            'FRED_API_KEY': '✓ Set' if os.getenv('FRED_API_KEY') else '✗ Not Set',
        }
        for key, status in env_vars.items():
            st.write(f"{key}: {status}")
    
    with col2:
        st.markdown("**System Info**")
        st.write(f"Python: {sys.version.split()[0]}")
        st.write(f"Streamlit: {st.__version__}")
        st.write(f"Working Directory: {os.getcwd()}")
        st.write(f"Dashboard Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main dashboard application."""
    
    # Sidebar navigation
    st.sidebar.title("🎛️ Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Overview", "🎯 Signals", "📈 Backtests", "🧠 Advanced Strategies", "⚙️ System Status"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Quick Links")
    st.sidebar.markdown("[📖 Documentation](https://github.com/cpoplaws/trading-ai)")
    st.sidebar.markdown("[🐛 Report Issue](https://github.com/cpoplaws/trading-ai/issues)")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Quick Actions")
    if st.sidebar.button("Run Pipeline", use_container_width=True):
        st.sidebar.info("Pipeline execution from dashboard coming soon!")
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Trading-AI v1.0**
    
    Autonomous AI Trading System
    
    ⚠️ This is for educational purposes only.
    Not financial advice.
    """)
    
    # Route to selected page
    if page == "📊 Overview":
        dashboard_overview()
    elif page == "🎯 Signals":
        signals_analysis()
    elif page == "📈 Backtests":
        backtest_results()
    elif page == "🧠 Advanced Strategies":
        advanced_strategies()
    elif page == "⚙️ System Status":
        system_status()

if __name__ == "__main__":
    main()
