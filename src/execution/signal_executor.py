"""
Automated signal execution system for Phase 2.
"""
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import time

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from execution.broker_interface import create_broker, AlpacaBroker, MockBroker
from utils.logger import setup_logger
from strategy.simple_strategy import analyze_signals

# Set up logging
logger = setup_logger("signal_executor", "INFO")

class SignalExecutor:
    """
    Automated signal execution system that connects ML signals to real broker trades.
    """
    
    def __init__(self, broker_type: str = "mock", initial_capital: float = 10000):
        """
        Initialize the signal executor.
        
        Args:
            broker_type: 'alpaca', 'mock', or 'paper'
            initial_capital: Starting capital for mock broker
        """
        self.broker_type = broker_type
        self.initial_capital = initial_capital
        
        # Create broker instance
        if broker_type == "mock":
            self.broker = create_broker(mock=True)
        elif broker_type == "alpaca":
            self.broker = create_broker(paper_trading=False, mock=False)
        else:  # paper trading
            self.broker = create_broker(paper_trading=True, mock=False)
        
        # Test connection
        if not self.broker.check_connection():
            logger.warning("Broker connection failed, falling back to mock")
            self.broker = create_broker(mock=True)
            self.broker_type = "mock"
        
        self.execution_log = []
        logger.info(f"Signal executor initialized with {self.broker_type} broker")
    
    def load_latest_signals(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load the latest signals for given tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary of ticker -> signals DataFrame
        """
        signals_data = {}
        
        for ticker in tickers:
            signal_file = f'./signals/{ticker}_signals.csv'
            if os.path.exists(signal_file):
                try:
                    df = pd.read_csv(signal_file, index_col=0, parse_dates=True)
                    # Get the most recent signal
                    latest_signal = df.iloc[-1]
                    signals_data[ticker] = latest_signal
                    logger.info(f"Loaded latest signal for {ticker}: {latest_signal['Signal']}")
                except Exception as e:
                    logger.error(f"Error loading signals for {ticker}: {str(e)}")
            else:
                logger.warning(f"No signals file found for {ticker}")
        
        return signals_data
    
    def should_execute_signal(self, ticker: str, signal: str, confidence: float) -> bool:
        """
        Determine if a signal should be executed based on various criteria.
        
        Args:
            ticker: Stock ticker
            signal: Trading signal ('BUY' or 'SELL')
            confidence: Signal confidence (0-1)
            
        Returns:
            True if signal should be executed
        """
        # Basic rules for signal execution
        min_confidence = 0.6  # Only execute signals with >60% confidence
        
        if confidence < min_confidence:
            logger.info(f"Skipping {ticker} {signal} - confidence too low ({confidence:.2f})")
            return False
        
        # Check market hours (simplified - assume always open for demo)
        current_time = datetime.now().time()
        market_open = datetime.strptime("09:30", "%H:%M").time()
        market_close = datetime.strptime("16:00", "%H:%M").time()
        
        # For demo purposes, always execute during "market hours"
        if True:  # current_time >= market_open and current_time <= market_close:
            return True
        else:
            logger.info(f"Market closed - queuing {ticker} {signal} for next session")
            return False
    
    def execute_signals(self, tickers: List[str], position_size: float = 0.1) -> Dict[str, bool]:
        """
        Execute trading signals for given tickers.
        
        Args:
            tickers: List of ticker symbols
            position_size: Fraction of portfolio per position
            
        Returns:
            Dictionary of ticker -> execution success
        """
        logger.info(f"Executing signals for {len(tickers)} tickers...")
        
        # Load latest signals
        signals_data = self.load_latest_signals(tickers)
        execution_results = {}
        
        # Get current portfolio state
        try:
            portfolio_value = self.broker.get_portfolio_value()
            buying_power = self.broker.get_buying_power()
            logger.info(f"Portfolio value: ${portfolio_value:,.2f}, Buying power: ${buying_power:,.2f}")
        except Exception as e:
            logger.error(f"Error getting portfolio info: {str(e)}")
            portfolio_value = self.initial_capital
            buying_power = self.initial_capital
        
        # Execute signals for each ticker
        for ticker in tickers:
            if ticker not in signals_data:
                logger.warning(f"No signal data for {ticker}")
                execution_results[ticker] = False
                continue
            
            signal_data = signals_data[ticker]
            signal = signal_data['Signal']
            confidence = signal_data.get('Confidence', 0.5)
            
            # Check if we should execute this signal
            if not self.should_execute_signal(ticker, signal, confidence):
                execution_results[ticker] = False
                continue
            
            # Execute the signal
            try:
                success = self.broker.execute_signal(
                    symbol=ticker,
                    signal=signal,
                    confidence=confidence,
                    position_size=position_size
                )
                
                execution_results[ticker] = success
                
                # Log the execution
                execution_log_entry = {
                    'timestamp': datetime.now(),
                    'ticker': ticker,
                    'signal': signal,
                    'confidence': confidence,
                    'success': success,
                    'portfolio_value': portfolio_value
                }
                self.execution_log.append(execution_log_entry)
                
                if success:
                    logger.info(f"✅ Executed {signal} for {ticker} (confidence: {confidence:.2f})")
                else:
                    logger.error(f"❌ Failed to execute {signal} for {ticker}")
                
                # Small delay between trades
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Exception executing signal for {ticker}: {str(e)}")
                execution_results[ticker] = False
        
        # Summary
        successful_executions = sum(1 for success in execution_results.values() if success)
        logger.info(f"Signal execution complete: {successful_executions}/{len(tickers)} successful")
        
        return execution_results
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get current portfolio summary.
        
        Returns:
            Dictionary with portfolio information
        """
        try:
            account_info = self.broker.get_account_info()
            
            if hasattr(self.broker, 'get_positions'):
                positions = self.broker.get_positions()
            else:
                positions = []
            
            summary = {
                'broker_type': self.broker_type,
                'portfolio_value': account_info.get('portfolio_value', 0),
                'cash': account_info.get('cash', 0),
                'buying_power': account_info.get('buying_power', 0),
                'num_positions': len(positions),
                'positions': positions[:5],  # Show first 5 positions
                'total_executions': len(self.execution_log),
                'last_execution': self.execution_log[-1] if self.execution_log else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {str(e)}")
            return {'error': str(e)}
    
    def run_daily_execution(self, tickers: List[str] = None) -> bool:
        """
        Run daily signal execution for specified tickers.
        
        Args:
            tickers: List of tickers to trade
            
        Returns:
            True if execution completed successfully
        """
        if tickers is None:
            tickers = ['AAPL', 'MSFT', 'SPY']
        
        logger.info(f"Starting daily execution for {tickers}")
        
        try:
            # Execute signals
            results = self.execute_signals(tickers)
            
            # Get updated portfolio summary
            summary = self.get_portfolio_summary()
            logger.info(f"Portfolio summary: {summary}")
            
            # Save execution log
            self.save_execution_log()
            
            successful_trades = sum(1 for success in results.values() if success)
            logger.info(f"Daily execution complete: {successful_trades}/{len(tickers)} trades executed")
            
            return successful_trades > 0
            
        except Exception as e:
            logger.error(f"Error in daily execution: {str(e)}")
            return False
    
    def save_execution_log(self, output_dir: str = './logs/'):
        """
        Save execution log to file.
        
        Args:
            output_dir: Directory to save log files
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            if self.execution_log:
                log_df = pd.DataFrame(self.execution_log)
                log_file = os.path.join(output_dir, f'execution_log_{datetime.now().strftime("%Y%m%d")}.csv')
                log_df.to_csv(log_file, index=False)
                logger.info(f"Execution log saved to {log_file}")
            
        except Exception as e:
            logger.error(f"Error saving execution log: {str(e)}")

def main():
    """
    Main function to demonstrate Phase 2 signal execution.
    """
    print("🚀 Phase 2: Automated Signal Execution")
    print("=" * 50)
    
    # Initialize signal executor with mock broker for demo
    executor = SignalExecutor(broker_type="mock", initial_capital=10000)
    
    # Run daily execution
    tickers = ['AAPL', 'MSFT', 'SPY']
    success = executor.run_daily_execution(tickers)
    
    if success:
        print("\\n✅ Phase 2 signal execution completed successfully!")
        
        # Show portfolio summary
        summary = executor.get_portfolio_summary()
        print(f"\\n📊 Portfolio Summary:")
        print(f"   Broker: {summary.get('broker_type', 'Unknown')}")
        print(f"   Portfolio Value: ${summary.get('portfolio_value', 0):,.2f}")
        print(f"   Cash: ${summary.get('cash', 0):,.2f}")
        print(f"   Positions: {summary.get('num_positions', 0)}")
        print(f"   Total Executions: {summary.get('total_executions', 0)}")
        
    else:
        print("\\n❌ Phase 2 execution encountered issues")
    
    print("\\n🎯 Phase 2 Complete! Ready for Phase 3: Advanced ML Models")

if __name__ == "__main__":
    main()
