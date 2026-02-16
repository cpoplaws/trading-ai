"""
Live Trading Demo - End-to-End Pipeline with Real-Time Dashboard Integration
Demonstrates: Data fetching → Feature engineering → Signal generation → Trade execution
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.data_ingestion.data_fetcher import fetch_data
from src.feature_engineering.technical_indicators import add_all_indicators
from src.modeling.model_predictor import load_model, generate_predictions
from src.strategy.signal_generator import SignalGenerator
from src.execution.broker_interface import AlpacaBroker, MockBroker
from src.execution.portfolio_tracker import PortfolioTracker
from src.advanced_strategies import AdvancedTradingStrategies
from src.advanced_strategies.sentiment_analyzer import SentimentAnalyzer
from src.data_ingestion.macro_data import MacroDataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveTradingDemo:
    """
    Demonstrate end-to-end live trading pipeline.
    """
    
    def __init__(self, symbols: list, initial_capital: float = 100000.0, 
                 paper_trading: bool = True):
        """
        Initialize live trading demo.
        
        Args:
            symbols: List of stock symbols to trade
            initial_capital: Starting capital
            paper_trading: Use paper trading (True) or live trading (False)
        """
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.paper_trading = paper_trading
        
        # Initialize components
        logger.info("Initializing trading components...")
        
        # Broker
        try:
            self.broker = AlpacaBroker(paper_trading=paper_trading)
            logger.info(f"Connected to Alpaca {'Paper' if paper_trading else 'Live'} Trading")
        except Exception as e:
            logger.warning(f"Alpaca connection failed, using MockBroker: {e}")
            self.broker = MockBroker()
        
        # Portfolio tracker
        self.tracker = PortfolioTracker(initial_capital=initial_capital)
        
        # Advanced strategies
        self.advanced_system = AdvancedTradingStrategies(symbols=symbols)
        
        # Sentiment analyzer
        self.sentiment = SentimentAnalyzer()
        
        # Macro data
        self.macro = MacroDataFetcher()
        
        # Signal generator
        self.signal_gen = SignalGenerator()
        
        logger.info("✅ All components initialized successfully")
    
    def fetch_latest_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Fetch latest market data for symbol."""
        logger.info(f"Fetching data for {symbol}...")
        
        try:
            df = fetch_data(symbol, period='3mo')
            
            if df.empty:
                logger.warning(f"No data fetched for {symbol}")
                return pd.DataFrame()
            
            # Add technical indicators
            df = add_all_indicators(df)
            
            logger.info(f"✅ Fetched {len(df)} bars for {symbol}")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def generate_signals(self, symbol: str, data: pd.DataFrame) -> dict:
        """Generate trading signals using multiple strategies."""
        logger.info(f"Generating signals for {symbol}...")
        
        try:
            # Get basic ML signal
            predictions = generate_predictions(symbol, data)
            
            if predictions is None or len(predictions) == 0:
                logger.warning(f"No predictions for {symbol}, using rule-based signals")
                ml_signal = 0
                ml_confidence = 0.5
            else:
                ml_signal = predictions[-1]
                ml_confidence = 0.7
            
            # Get advanced strategy signals
            current_price = data['close'].iloc[-1]
            market_data = {symbol: {'1d': data}}
            current_prices = {symbol: current_price}
            
            try:
                dashboard = self.advanced_system.get_portfolio_dashboard(market_data, current_prices)
                symbol_signals = dashboard.get('symbol_signals', {}).get(symbol, {})
                agg_signal = symbol_signals.get('aggregated_signal', {})
                
                advanced_signal_val = agg_signal.get('signal', 'HOLD')
                advanced_signal = 1 if advanced_signal_val == 'BUY' else -1 if advanced_signal_val == 'SELL' else 0
                advanced_confidence = agg_signal.get('confidence', 0.5)
            except Exception as e:
                logger.warning(f"Advanced signals failed: {e}")
                advanced_signal = 0
                advanced_confidence = 0.3
            
            # Get sentiment signal
            try:
                sentiment_data = self.sentiment.aggregate_sentiment_signals(symbol)
                sentiment_signal = 1 if sentiment_data.get('overall_sentiment', 0) > 0.1 else \
                                 -1 if sentiment_data.get('overall_sentiment', 0) < -0.1 else 0
                sentiment_confidence = sentiment_data.get('confidence', 0.5)
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
                sentiment_signal = 0
                sentiment_confidence = 0.3
            
            # Aggregate signals with weights
            weights = {
                'ml': 0.3,
                'advanced': 0.4,
                'sentiment': 0.3
            }
            
            final_signal = (
                ml_signal * weights['ml'] * ml_confidence +
                advanced_signal * weights['advanced'] * advanced_confidence +
                sentiment_signal * weights['sentiment'] * sentiment_confidence
            )
            
            # Determine action
            if final_signal > 0.3:
                action = 'BUY'
            elif final_signal < -0.3:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            confidence = abs(final_signal)
            
            return {
                'symbol': symbol,
                'action': action,
                'signal_strength': final_signal,
                'confidence': confidence,
                'ml_signal': ml_signal,
                'advanced_signal': advanced_signal,
                'sentiment_signal': sentiment_signal,
                'timestamp': datetime.now()
            }
        
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'signal_strength': 0,
                'confidence': 0,
                'error': str(e)
            }
    
    def execute_trade(self, symbol: str, signal: dict, current_price: float) -> bool:
        """Execute trade based on signal."""
        try:
            action = signal['action']
            confidence = signal['confidence']
            
            if action == 'HOLD':
                logger.info(f"⏸️  {symbol}: HOLD signal - No action taken")
                return False
            
            # Get account info
            account = self.broker.get_account()
            equity = float(account.get('equity', self.initial_capital))
            
            # Calculate position size (risk 1% per trade, scaled by confidence)
            risk_per_trade = equity * 0.01 * confidence
            qty = int(risk_per_trade / current_price)
            
            if qty == 0:
                logger.info(f"⚠️  {symbol}: Position size too small, skipping")
                return False
            
            # Check existing position
            positions = self.broker.get_positions()
            current_position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if action == 'BUY':
                if current_position and int(current_position['qty']) > 0:
                    logger.info(f"⚠️  {symbol}: Already long, skipping")
                    return False
                
                # Place buy order
                logger.info(f"🟢 BUY {qty} shares of {symbol} @ ${current_price:.2f} (Confidence: {confidence:.2%})")
                order = self.broker.place_order(
                    symbol=symbol,
                    qty=qty,
                    side='buy',
                    order_type='market'
                )
                
                if order:
                    # Record in tracker
                    self.tracker.update_position(symbol, qty, current_price, current_price)
                    self.tracker.record_trade(symbol, 'BUY', qty, current_price)
                    logger.info(f"✅ Buy order executed: {order.get('id', 'N/A')}")
                    return True
            
            elif action == 'SELL':
                if not current_position or int(current_position['qty']) <= 0:
                    logger.info(f"⚠️  {symbol}: No position to sell, skipping")
                    return False
                
                # Sell existing position
                position_qty = int(current_position['qty'])
                logger.info(f"🔴 SELL {position_qty} shares of {symbol} @ ${current_price:.2f} (Confidence: {confidence:.2%})")
                
                order = self.broker.place_order(
                    symbol=symbol,
                    qty=position_qty,
                    side='sell',
                    order_type='market'
                )
                
                if order:
                    # Record in tracker
                    self.tracker.update_position(symbol, -position_qty, current_price, current_price)
                    self.tracker.record_trade(symbol, 'SELL', position_qty, current_price)
                    logger.info(f"✅ Sell order executed: {order.get('id', 'N/A')}")
                    return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return False
    
    def run_cycle(self) -> dict:
        """Run one complete trading cycle."""
        logger.info("\n" + "="*80)
        logger.info("🔄 Starting trading cycle...")
        logger.info("="*80 + "\n")
        
        cycle_results = {
            'timestamp': datetime.now(),
            'signals': {},
            'trades_executed': 0,
            'errors': []
        }
        
        # Get macro context
        try:
            macro_summary = self.macro.get_macro_summary()
            regime = macro_summary.get('regime', {})
            logger.info(f"📊 Economic Regime: {regime.get('regime', 'UNKNOWN')} (Confidence: {regime.get('confidence', 0):.1%})")
        except Exception as e:
            logger.warning(f"Macro data unavailable: {e}")
        
        # Process each symbol
        for symbol in self.symbols:
            try:
                logger.info(f"\n📈 Processing {symbol}...")
                
                # 1. Fetch latest data
                data = self.fetch_latest_data(symbol)
                
                if data.empty:
                    logger.warning(f"⚠️  {symbol}: No data available, skipping")
                    continue
                
                current_price = data['close'].iloc[-1]
                logger.info(f"   Current price: ${current_price:.2f}")
                
                # 2. Generate signals
                signal = self.generate_signals(symbol, data)
                cycle_results['signals'][symbol] = signal
                
                logger.info(f"   Signal: {signal['action']} (Strength: {signal['signal_strength']:.3f}, Confidence: {signal['confidence']:.2%})")
                
                # 3. Execute trade if signal is strong enough
                if signal['confidence'] > 0.6:
                    if self.execute_trade(symbol, signal, current_price):
                        cycle_results['trades_executed'] += 1
                else:
                    logger.info(f"   ⏸️  Signal confidence too low, no action")
            
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                cycle_results['errors'].append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        # Update portfolio metrics
        logger.info("\n" + "-"*80)
        logger.info("📊 Portfolio Update")
        logger.info("-"*80)
        
        dashboard = self.tracker.get_dashboard()
        pnl = dashboard['pnl']
        
        logger.info(f"Total PnL: ${pnl['total_pnl']:.2f} ({pnl['total_pnl_pct']:.2f}%)")
        logger.info(f"Unrealized PnL: ${pnl['unrealized_pnl']:.2f}")
        logger.info(f"Current Equity: ${pnl['current_equity']:.2f}")
        
        # Check risk limits
        risk_check = self.tracker.check_risk_limits()
        if not risk_check['risk_ok']:
            logger.warning("⚠️  Risk limit violations detected:")
            for violation in risk_check['violations']:
                logger.warning(f"   - {violation['type']}: {violation['current']:.2f}% (limit: {violation['limit']:.2f}%)")
        
        logger.info("\n" + "="*80)
        logger.info(f"✅ Cycle complete - {cycle_results['trades_executed']} trades executed")
        logger.info("="*80 + "\n")
        
        return cycle_results
    
    def run_continuous(self, interval_minutes: int = 60, max_cycles: int = None):
        """Run continuous trading with specified interval."""
        logger.info(f"🚀 Starting continuous trading mode")
        logger.info(f"   Interval: {interval_minutes} minutes")
        logger.info(f"   Symbols: {', '.join(self.symbols)}")
        logger.info(f"   Max cycles: {max_cycles if max_cycles else 'Unlimited'}")
        logger.info(f"   Paper trading: {'Yes' if self.paper_trading else 'No'}")
        logger.info("")
        
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                
                if max_cycles and cycle_count > max_cycles:
                    logger.info(f"✅ Reached maximum cycles ({max_cycles}), stopping")
                    break
                
                # Run trading cycle
                results = self.run_cycle()
                
                # Wait for next cycle
                if max_cycles is None or cycle_count < max_cycles:
                    logger.info(f"⏰ Next cycle in {interval_minutes} minutes...")
                    time.sleep(interval_minutes * 60)
        
        except KeyboardInterrupt:
            logger.info("\n⚠️  Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
        finally:
            logger.info("\n" + "="*80)
            logger.info("📈 Final Portfolio Summary")
            logger.info("="*80)
            
            dashboard = self.tracker.get_dashboard()
            performance = dashboard['performance']
            
            logger.info(f"\nTotal Return: {performance['total_return_pct']:.2f}%")
            logger.info(f"Max Drawdown: {performance['max_drawdown_pct']:.2f}%")
            logger.info(f"Win Rate: {performance['win_rate_pct']:.1f}%")
            logger.info(f"Total Trades: {performance['total_trades']}")
            logger.info(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
            
            logger.info("\n👋 Live trading demo ended")


def main():
    """Main entry point."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║           🚀 TRADING-AI LIVE DEMO 🚀                     ║
    ║                                                           ║
    ║     End-to-End Automated Trading Pipeline                ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    SYMBOLS = ['AAPL', 'MSFT', 'SPY']
    INITIAL_CAPITAL = 100000.0
    PAPER_TRADING = True
    INTERVAL_MINUTES = 60  # Run every hour
    MAX_CYCLES = 5  # Run 5 cycles for demo
    
    # Create and run demo
    demo = LiveTradingDemo(
        symbols=SYMBOLS,
        initial_capital=INITIAL_CAPITAL,
        paper_trading=PAPER_TRADING
    )
    
    print(f"""
    Configuration:
    • Symbols: {', '.join(SYMBOLS)}
    • Initial Capital: ${INITIAL_CAPITAL:,.2f}
    • Mode: {'Paper Trading' if PAPER_TRADING else 'Live Trading'}
    • Interval: {INTERVAL_MINUTES} minutes
    • Max Cycles: {MAX_CYCLES}
    
    Press Ctrl+C to stop at any time.
    """)
    
    input("Press Enter to start trading demo...")
    
    demo.run_continuous(
        interval_minutes=INTERVAL_MINUTES,
        max_cycles=MAX_CYCLES
    )


if __name__ == "__main__":
    main()
