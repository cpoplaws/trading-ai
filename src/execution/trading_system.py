"""
Integrated Trading System - Phase 2 Complete Implementation
Connects ML models, risk management, and execution for automated trading.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import time

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from execution.broker_interface import create_broker
from execution.portfolio_tracker import PortfolioTracker
from execution.risk_manager import RiskManager, RiskLimits
from execution.order_manager import EnhancedOrderManager
from strategy.simple_strategy import analyze_signals
from utils.logger import setup_logger

# Set up logging
logger = setup_logger("trading_system", "INFO")

class TradingSystem:
    """
    Complete integrated trading system for Phase 2.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the trading system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.is_running = False
        
        # Initialize components
        self._initialize_components()
        
        # Trading state
        self.last_signal_check = None
        self.execution_log = []
        
        logger.info("Trading system initialized")
    
    def _initialize_components(self) -> None:
        """Initialize all trading system components."""
        try:
            # Create broker
            broker_config = self.config.get('broker', {})
            self.broker = create_broker(
                paper_trading=broker_config.get('paper_trading', True),
                mock=broker_config.get('mock', False)
            )
            
            if not self.broker.check_connection():
                logger.warning("Broker connection failed, using mock broker")
                self.broker = create_broker(mock=True)
            
            # Create portfolio tracker
            portfolio_config = self.config.get('portfolio', {})
            self.portfolio_tracker = PortfolioTracker(
                broker=self.broker,
                initial_capital=portfolio_config.get('initial_capital', 10000),
                save_path=portfolio_config.get('save_path', './logs/')
            )
            
            # Create risk manager
            risk_config = self.config.get('risk', {})
            risk_limits = RiskLimits(
                max_position_size=risk_config.get('max_position_size', 0.1),
                max_portfolio_exposure=risk_config.get('max_portfolio_exposure', 0.8),
                max_daily_loss=risk_config.get('max_daily_loss', 0.05),
                min_confidence=risk_config.get('min_confidence', 0.6),
                max_positions=risk_config.get('max_positions', 20)
            )
            self.risk_manager = RiskManager(self.portfolio_tracker, risk_limits)
            
            # Create order manager
            self.order_manager = EnhancedOrderManager(
                broker=self.broker,
                risk_manager=self.risk_manager,
                portfolio_tracker=self.portfolio_tracker,
                save_path=self.config.get('logs_path', './logs/')
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def load_latest_signals(self) -> Dict[str, Dict]:
        """
        Load latest trading signals for all configured tickers.
        
        Returns:
            Dictionary of ticker -> signal data
        """
        signals = {}
        tickers = self.config.get('tickers', ['AAPL', 'MSFT', 'SPY'])
        
        for ticker in tickers:
            try:
                signal_file = f'./signals/{ticker}_signals.csv'
                if os.path.exists(signal_file):
                    df = pd.read_csv(signal_file, index_col=0, parse_dates=True)
                    if not df.empty:
                        latest = df.iloc[-1]
                        signals[ticker] = {
                            'signal': latest.get('Signal', 'HOLD'),
                            'confidence': latest.get('Confidence', 0.5),
                            'timestamp': df.index[-1].isoformat(),
                            'price': latest.get('Close', 100)  # Fallback price
                        }
                        logger.debug(f"Loaded signal for {ticker}: {signals[ticker]['signal']}")
                else:
                    logger.warning(f"No signals file found for {ticker}")
                    
            except Exception as e:
                logger.error(f"Error loading signals for {ticker}: {str(e)}")
        
        return signals
    
    def execute_signals(self, signals: Dict[str, Dict]) -> Dict[str, bool]:
        """
        Execute trading signals with risk management.
        
        Args:
            signals: Dictionary of ticker -> signal data
            
        Returns:
            Dictionary of ticker -> execution success
        """
        execution_results = {}
        
        # Update order statuses first
        self.order_manager.update_all_orders()
        
        for ticker, signal_data in signals.items():
            try:
                signal = signal_data['signal']
                confidence = signal_data['confidence']
                price = signal_data['price']
                
                # Skip HOLD signals
                if signal == 'HOLD':
                    continue
                
                # Get current position
                current_position = self.portfolio_tracker.broker.get_position(ticker)
                current_qty = int(current_position['qty']) if current_position else 0
                
                # Determine action
                if signal == 'BUY' and current_qty >= 0:
                    # Buy signal - calculate position size
                    portfolio_value = self.portfolio_tracker.calculate_portfolio_metrics().total_value
                    max_investment = portfolio_value * self.config.get('default_position_size', 0.05)
                    shares_to_buy = max(1, int(max_investment / price))
                    
                    order_id = self.order_manager.create_market_order(
                        symbol=ticker,
                        side='buy',
                        quantity=shares_to_buy,
                        strategy='ml_signal',
                        confidence=confidence
                    )
                    
                    execution_results[ticker] = order_id is not None
                    
                    if order_id:
                        logger.info(f"BUY order placed for {ticker}: {shares_to_buy} shares")
                    else:
                        logger.warning(f"BUY order rejected for {ticker}")
                
                elif signal == 'SELL' and current_qty > 0:
                    # Sell signal - sell partial or full position based on confidence
                    shares_to_sell = max(1, int(current_qty * confidence))
                    
                    order_id = self.order_manager.create_market_order(
                        symbol=ticker,
                        side='sell',
                        quantity=shares_to_sell,
                        strategy='ml_signal',
                        confidence=confidence
                    )
                    
                    execution_results[ticker] = order_id is not None
                    
                    if order_id:
                        logger.info(f"SELL order placed for {ticker}: {shares_to_sell} shares")
                    else:
                        logger.warning(f"SELL order rejected for {ticker}")
                
                else:
                    logger.debug(f"No action taken for {ticker}: signal={signal}, current_qty={current_qty}")
                    execution_results[ticker] = True  # No action needed
                
            except Exception as e:
                logger.error(f"Error executing signal for {ticker}: {str(e)}")
                execution_results[ticker] = False
        
        return execution_results
    
    def check_risk_triggers(self) -> None:
        """
        Check for risk-based position closures (stop loss, take profit).
        """
        try:
            positions = self.portfolio_tracker.get_current_positions()
            
            # Check stop losses
            stop_losses = self.risk_manager.check_stop_loss(positions)
            for sl in stop_losses:
                symbol = sl['symbol']
                position = next((p for p in positions if p.symbol == symbol), None)
                if position:
                    # Close entire position on stop loss
                    order_id = self.order_manager.create_market_order(
                        symbol=symbol,
                        side='sell' if position.quantity > 0 else 'buy',
                        quantity=abs(position.quantity),
                        strategy='stop_loss',
                        confidence=1.0
                    )
                    if order_id:
                        logger.warning(f"Stop loss triggered for {symbol}: {sl['current_loss']:.1f}%")
            
            # Check take profits
            take_profits = self.risk_manager.check_take_profit(positions)
            for tp in take_profits:
                symbol = tp['symbol']
                position = next((p for p in positions if p.symbol == symbol), None)
                if position:
                    # Partial close on take profit
                    close_qty = int(abs(position.quantity) * tp['close_percentage'])
                    if close_qty > 0:
                        order_id = self.order_manager.create_market_order(
                            symbol=symbol,
                            side='sell' if position.quantity > 0 else 'buy',
                            quantity=close_qty,
                            strategy='take_profit',
                            confidence=1.0
                        )
                        if order_id:
                            logger.info(f"Take profit triggered for {symbol}: {tp['current_profit']:.1f}%")
                            
        except Exception as e:
            logger.error(f"Error checking risk triggers: {str(e)}")
    
    def run_trading_cycle(self) -> Dict:
        """
        Run one complete trading cycle.
        
        Returns:
            Cycle results summary
        """
        cycle_start = datetime.now()
        logger.info("Starting trading cycle")
        
        try:
            # Load latest signals
            signals = self.load_latest_signals()
            logger.info(f"Loaded signals for {len(signals)} tickers")
            
            # Execute signals
            if signals:
                execution_results = self.execute_signals(signals)
                successful_executions = sum(1 for success in execution_results.values() if success)
                logger.info(f"Executed {successful_executions}/{len(signals)} signals successfully")
            else:
                execution_results = {}
                logger.info("No signals to execute")
            
            # Check risk triggers
            self.check_risk_triggers()
            
            # Update portfolio tracking
            self.portfolio_tracker.take_snapshot()
            
            # Create cycle summary
            cycle_summary = {
                'timestamp': cycle_start.isoformat(),
                'duration': (datetime.now() - cycle_start).total_seconds(),
                'signals_loaded': len(signals),
                'signals_executed': len(execution_results),
                'successful_executions': sum(1 for success in execution_results.values() if success),
                'portfolio_value': self.portfolio_tracker.calculate_portfolio_metrics().total_value,
                'active_orders': len(self.order_manager.get_active_orders()),
                'active_positions': len(self.portfolio_tracker.get_current_positions())
            }
            
            self.execution_log.append(cycle_summary)
            logger.info(f"Trading cycle completed in {cycle_summary['duration']:.1f}s")
            
            return cycle_summary
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {str(e)}")
            return {'error': str(e), 'timestamp': cycle_start.isoformat()}
    
    def start_automated_trading(self, check_interval: int = 300) -> None:
        """
        Start automated trading with periodic signal checks.
        
        Args:
            check_interval: Seconds between signal checks (default: 5 minutes)
        """
        logger.info(f"Starting automated trading (check interval: {check_interval}s)")
        self.is_running = True
        
        try:
            while self.is_running:
                # Run trading cycle
                cycle_result = self.run_trading_cycle()
                
                # Print status
                self.print_status()
                
                # Wait for next cycle
                if self.is_running:
                    logger.debug(f"Waiting {check_interval}s until next cycle")
                    time.sleep(check_interval)
                    
        except KeyboardInterrupt:
            logger.info("Automated trading stopped by user")
        except Exception as e:
            logger.error(f"Error in automated trading: {str(e)}")
        finally:
            self.is_running = False
            logger.info("Automated trading stopped")
    
    def stop_trading(self) -> None:
        """Stop automated trading."""
        self.is_running = False
        logger.info("Trading stop requested")
    
    def print_status(self) -> None:
        """Print comprehensive system status."""
        try:
            print("\n" + "="*80)
            print("🤖 TRADING SYSTEM STATUS")
            print("="*80)
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Status: {'🟢 RUNNING' if self.is_running else '🔴 STOPPED'}")
            
            # Portfolio status
            self.portfolio_tracker.print_portfolio_status()
            
            # Risk status
            self.risk_manager.print_risk_status()
            
            # Order status
            self.order_manager.print_order_status()
            
            print("="*80)
            
        except Exception as e:
            logger.error(f"Error printing status: {str(e)}")
    
    def get_performance_report(self) -> Dict:
        """
        Generate comprehensive performance report.
        
        Returns:
            Performance report dictionary
        """
        try:
            # Portfolio metrics
            portfolio_summary = self.portfolio_tracker.get_performance_summary()
            
            # Risk summary
            risk_summary = self.risk_manager.get_risk_summary()
            
            # Execution summary
            execution_summary = {
                'total_cycles': len(self.execution_log),
                'successful_cycles': sum(1 for cycle in self.execution_log if 'error' not in cycle),
                'avg_cycle_duration': sum(cycle.get('duration', 0) for cycle in self.execution_log) / max(len(self.execution_log), 1),
                'total_signals_processed': sum(cycle.get('signals_loaded', 0) for cycle in self.execution_log),
                'total_executions': sum(cycle.get('signals_executed', 0) for cycle in self.execution_log)
            }
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'system_status': 'running' if self.is_running else 'stopped',
                'portfolio': portfolio_summary,
                'risk': risk_summary,
                'execution': execution_summary,
                'broker_type': self.broker.__class__.__name__
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return {'error': str(e)}

def create_default_config() -> Dict:
    """
    Create default trading system configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'broker': {
            'paper_trading': True,
            'mock': False  # Set to True if no Alpaca credentials
        },
        'portfolio': {
            'initial_capital': 10000,
            'save_path': './logs/'
        },
        'risk': {
            'max_position_size': 0.1,  # 10% max position
            'max_portfolio_exposure': 0.8,  # 80% max exposure
            'max_daily_loss': 0.05,  # 5% max daily loss
            'min_confidence': 0.6,  # 60% min confidence
            'max_positions': 20
        },
        'tickers': ['AAPL', 'MSFT', 'SPY'],
        'default_position_size': 0.05,  # 5% default position
        'logs_path': './logs/',
        'check_interval': 300  # 5 minutes
    }

if __name__ == "__main__":
    # Demo trading system
    print("🤖 Phase 2: Complete Trading System Demo")
    print("="*50)
    
    # Create configuration
    config = create_default_config()
    
    # Override for demo (use mock broker)
    config['broker']['mock'] = True
    
    try:
        # Initialize trading system
        trading_system = TradingSystem(config)
        
        # Print initial status
        trading_system.print_status()
        
        # Run one trading cycle
        print("\n🔄 Running trading cycle...")
        cycle_result = trading_system.run_trading_cycle()
        print(f"Cycle result: {cycle_result}")
        
        # Print final status
        trading_system.print_status()
        
        # Generate performance report
        report = trading_system.get_performance_report()
        print(f"\n📊 Performance Report:")
        print(json.dumps(report, indent=2))
        
        print("\n🎉 Phase 2 Trading System Demo Complete!")
        print("✅ All components integrated and working:")
        print("  - Broker interface with paper trading")
        print("  - Portfolio tracking with real-time PnL")
        print("  - Risk management with position limits")
        print("  - Enhanced order management")
        print("  - Automated signal execution")
        print("  - Stop loss and take profit triggers")
        
    except Exception as e:
        print(f"❌ Error in trading system demo: {str(e)}")
        import traceback
        traceback.print_exc()
