"""
Comprehensive backtesting engine for trading strategies.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple, Optional, List
import logging
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

class PortfolioBacktester:
    """
    Portfolio backtesting engine with comprehensive performance metrics.
    """
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting portfolio value
            commission: Commission rate per trade (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.reset()
    
    def reset(self):
        """Reset the backtester state."""
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        self.trade_log = []
        
    def load_signals_and_prices(self, signal_file: str, price_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load signals and price data.
        
        Args:
            signal_file: Path to signals CSV
            price_file: Path to price data CSV
            
        Returns:
            Tuple of (signals_df, prices_df)
        """
        try:
            signals = pd.read_csv(signal_file, index_col=0, parse_dates=True)
            
            # Load price data with proper header handling
            prices = pd.read_csv(price_file, index_col=0, parse_dates=True, skiprows=[1, 2])
            
            # Ensure price columns are numeric
            price_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
            for col in price_columns:
                if col in prices.columns:
                    prices[col] = pd.to_numeric(prices[col], errors='coerce')
            
            prices = prices.dropna()
            
            # Align dates
            common_dates = signals.index.intersection(prices.index)
            signals = signals.loc[common_dates]
            prices = prices.loc[common_dates]
            
            logger.info(f"Loaded {len(signals)} signals and {len(prices)} price points")
            return signals, prices
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def execute_trade(self, ticker: str, signal: str, price: float, confidence: float, 
                     date: pd.Timestamp, position_size: float = 0.1) -> bool:
        """
        Execute a trade based on signal.
        
        Args:
            ticker: Stock ticker
            signal: 'BUY' or 'SELL'
            price: Current price
            confidence: Signal confidence (0-1)
            date: Trade date
            position_size: Fraction of portfolio to allocate (0-1)
            
        Returns:
            True if trade executed, False otherwise
        """
        try:
            # Adjust position size based on confidence
            adjusted_size = position_size * confidence
            
            if signal == 'BUY':
                # Calculate maximum shares we can buy
                max_investment = self.cash * adjusted_size
                commission_cost = max_investment * self.commission
                net_investment = max_investment - commission_cost
                shares_to_buy = int(net_investment / price)
                
                if shares_to_buy > 0 and net_investment > 0:
                    total_cost = shares_to_buy * price + commission_cost
                    
                    if total_cost <= self.cash:
                        # Execute buy
                        self.cash -= total_cost
                        self.positions[ticker] = self.positions.get(ticker, 0) + shares_to_buy
                        
                        trade = {
                            'date': date,
                            'ticker': ticker,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': price,
                            'total_cost': total_cost,
                            'commission': commission_cost,
                            'confidence': confidence,
                            'portfolio_value': self.get_portfolio_value({ticker: price})
                        }
                        self.trade_log.append(trade)
                        logger.debug(f"BUY {shares_to_buy} {ticker} @ ${price:.2f}")
                        return True
            
            elif signal == 'SELL' and ticker in self.positions and self.positions[ticker] > 0:
                # Sell all or partial position
                shares_to_sell = int(self.positions[ticker] * confidence)
                
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * price
                    commission_cost = proceeds * self.commission
                    net_proceeds = proceeds - commission_cost
                    
                    # Execute sell
                    self.cash += net_proceeds
                    self.positions[ticker] -= shares_to_sell
                    
                    if self.positions[ticker] <= 0:
                        del self.positions[ticker]
                    
                    trade = {
                        'date': date,
                        'ticker': ticker,
                        'action': 'SELL',
                        'shares': shares_to_sell,
                        'price': price,
                        'total_proceeds': net_proceeds,
                        'commission': commission_cost,
                        'confidence': confidence,
                        'portfolio_value': self.get_portfolio_value({ticker: price})
                    }
                    self.trade_log.append(trade)
                    logger.debug(f"SELL {shares_to_sell} {ticker} @ ${price:.2f}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return False
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value.
        
        Args:
            current_prices: Dict of ticker -> current price
            
        Returns:
            Total portfolio value
        """
        total_value = self.cash
        
        for ticker, shares in self.positions.items():
            if ticker in current_prices:
                total_value += shares * current_prices[ticker]
        
        return total_value
    
    def run_backtest(self, signals_file: str, prices_file: str, 
                    position_size: float = 0.1) -> Dict:
        """
        Run complete backtest on signal file.
        
        Args:
            signals_file: Path to signals CSV
            prices_file: Path to prices CSV
            position_size: Fraction of portfolio per trade
            
        Returns:
            Dictionary with backtest results
        """
        try:
            self.reset()
            ticker = os.path.basename(signals_file).split('_')[0]
            
            logger.info(f"Starting backtest for {ticker}")
            
            signals, prices = self.load_signals_and_prices(signals_file, prices_file)
            
            # Run backtest day by day
            for date in signals.index:
                if date in prices.index:
                    signal_row = signals.loc[date]
                    price_row = prices.loc[date]
                    
                    current_price = price_row['Close'] if 'Close' in price_row else price_row['close']
                    signal = signal_row['Signal']
                    confidence = signal_row.get('Confidence', 0.5)
                    
                    # Execute trade
                    self.execute_trade(
                        ticker=ticker,
                        signal=signal,
                        price=current_price,
                        confidence=confidence,
                        date=date,
                        position_size=position_size
                    )
                    
                    # Record portfolio value
                    portfolio_val = self.get_portfolio_value({ticker: current_price})
                    self.portfolio_history.append({
                        'date': date,
                        'portfolio_value': portfolio_val,
                        'cash': self.cash,
                        'positions_value': portfolio_val - self.cash,
                        'price': current_price
                    })
            
            # Calculate final metrics
            results = self.calculate_performance_metrics(ticker)
            logger.info(f"Backtest completed for {ticker}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return {'error': str(e)}
    
    def calculate_performance_metrics(self, ticker: str) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Dictionary with performance metrics
        """
        if not self.portfolio_history:
            return {'error': 'No portfolio history available'}
        
        portfolio_df = pd.DataFrame(self.portfolio_history).set_index('date')
        trades_df = pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame()
        
        # Basic metrics
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Daily returns
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        portfolio_df['cum_return'] = (portfolio_df['portfolio_value'] / self.initial_capital) - 1
        
        # Risk metrics
        volatility = portfolio_df['daily_return'].std() * np.sqrt(252)  # Annualized
        max_drawdown = self.calculate_max_drawdown(portfolio_df['cum_return'])
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_return = total_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Trade metrics
        num_trades = len(trades_df)
        winning_trades = 0
        losing_trades = 0
        
        if not trades_df.empty:
            # Calculate trade profitability
            for i in range(len(trades_df) - 1):
                if trades_df.iloc[i]['action'] == 'BUY' and i + 1 < len(trades_df):
                    buy_price = trades_df.iloc[i]['price']
                    sell_price = trades_df.iloc[i + 1]['price']
                    if sell_price > buy_price:
                        winning_trades += 1
                    else:
                        losing_trades += 1
        
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        
        # Buy and hold comparison
        if len(portfolio_df) > 0:
            buy_hold_return = (portfolio_df['price'].iloc[-1] - portfolio_df['price'].iloc[0]) / portfolio_df['price'].iloc[0]
        else:
            buy_hold_return = 0
        
        results = {
            'ticker': ticker,
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'cumulative_return': portfolio_df['cum_return'].iloc[-1] if len(portfolio_df) > 0 else 0,
            'buy_hold_return': buy_hold_return,
            'buy_hold_return_pct': buy_hold_return * 100,
            'excess_return': total_return - buy_hold_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'final_cash': self.cash,
            'final_positions': dict(self.positions),
            'total_days': len(portfolio_df),
            'avg_daily_return': portfolio_df['daily_return'].mean() if len(portfolio_df) > 0 else 0
        }
        
        return results
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def save_backtest_report(self, results: Dict, output_dir: str = './backtests/'):
        """
        Save comprehensive backtest report.
        
        Args:
            results: Backtest results dictionary
            output_dir: Directory to save reports
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            ticker = results.get('ticker', 'unknown')
            
            # Save detailed results
            report_file = os.path.join(output_dir, f'{ticker}_backtest_report.txt')
            
            with open(report_file, 'w') as f:
                f.write(f"BACKTEST REPORT - {ticker}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Test Period: {results.get('total_days', 0)} days\n")
                f.write(f"Initial Capital: ${results['initial_capital']:,.2f}\n")
                f.write(f"Final Value: ${results['final_value']:,.2f}\n")
                f.write(f"Total Return: {results['total_return_pct']:.2f}%\n")
                f.write(f"Buy & Hold Return: {results['buy_hold_return_pct']:.2f}%\n")
                f.write(f"Excess Return: {results['excess_return']*100:.2f}%\n")
                f.write(f"Volatility (Annual): {results['volatility']*100:.2f}%\n")
                f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}\n")
                f.write(f"Max Drawdown: {results['max_drawdown']*100:.2f}%\n")
                f.write(f"Total Trades: {results['num_trades']}\n")
                f.write(f"Win Rate: {results['win_rate_pct']:.1f}%\n")
                f.write(f"Final Cash: ${results['final_cash']:,.2f}\n")
                f.write(f"Final Positions: {results['final_positions']}\n")
            
            # Save trade log if available
            if self.trade_log:
                trades_file = os.path.join(output_dir, f'{ticker}_trades.csv')
                trades_df = pd.DataFrame(self.trade_log)
                trades_df.to_csv(trades_file, index=False)
            
            # Save portfolio history
            if self.portfolio_history:
                portfolio_file = os.path.join(output_dir, f'{ticker}_portfolio_history.csv')
                portfolio_df = pd.DataFrame(self.portfolio_history)
                portfolio_df.to_csv(portfolio_file, index=False)
            
            logger.info(f"Backtest report saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving backtest report: {str(e)}")

def log_backtest_results(results: Dict, log_path: str = "./logs/backtest_results.log") -> None:
    """
    Persist key backtest metrics to a log file.
    """
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} | {results.get('ticker', 'unknown')} | "
                f"Cumulative: {results.get('cumulative_return', 0):.4f} | "
                f"Sharpe: {results.get('sharpe_ratio', 0):.4f} | "
                f"MaxDD: {results.get('max_drawdown', 0):.4f}\n"
            )
    except Exception as e:
        logger.error(f"Failed to log backtest results: {e}")

def plot_equity_curve(portfolio_history: List[Dict], ticker: str, output_path: str = "./logs/equity_curve.png") -> Optional[str]:
    """
    Generate and save equity curve visualization.
    """
    if not portfolio_history:
        return None
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(portfolio_history)
        if 'date' in df.columns:
            df = df.sort_values('date')
        plt.figure(figsize=(10, 4))
        plt.plot(df['date'], df['portfolio_value'], label='Equity Curve')
        plt.title(f'Equity Curve - {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
    except Exception as e:
        logger.error(f"Failed to plot equity curve: {e}")
        return None

def run_backtest(signal_file: str, ticker: Optional[str] = None, initial_capital: float = 10000) -> Dict:
    """
    Convenience function to run a single backtest.
    
    Args:
        signal_file: Path to signals CSV file
        ticker: Stock ticker (auto-detected if None)
        initial_capital: Starting capital
        
    Returns:
        Backtest results dictionary
    """
    if ticker is None:
        ticker = os.path.basename(signal_file).split('_')[0]
    
    # Construct price file path
    price_file = f'./data/processed/{ticker}.csv'
    
    if not os.path.exists(price_file):
        price_file = f'./data/raw/{ticker}.csv'
    
    if not os.path.exists(price_file):
        logger.error(f"Price file not found for {ticker}")
        return {'error': f'Price file not found for {ticker}'}
    
    backtester = PortfolioBacktester(initial_capital=initial_capital)
    results = backtester.run_backtest(signal_file, price_file)
    
    # Save detailed report
    backtester.save_backtest_report(results)
    log_backtest_results(results)
    results['equity_curve_path'] = plot_equity_curve(
        backtester.portfolio_history,
        ticker,
        output_path="./logs/equity_curve.png" if ticker is None else f"./logs/{ticker}_equity_curve.png"
    )
    
    return results

def run_portfolio_backtest(tickers: Optional[List[str]] = None, initial_capital: float = 10000) -> Dict:
    """
    Run backtest across multiple tickers as a portfolio.
    
    Args:
        tickers: List of tickers to include
        initial_capital: Starting capital
        
    Returns:
        Portfolio backtest results
    """
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'SPY']
    
    portfolio_results = {}
    total_return = 0
    
    for ticker in tickers:
        signal_file = f'./signals/{ticker}_signals.csv'
        if os.path.exists(signal_file):
            results = run_backtest(signal_file, ticker, initial_capital / len(tickers))
            portfolio_results[ticker] = results
            if 'total_return' in results:
                total_return += results['total_return']
    
    portfolio_results['portfolio_summary'] = {
        'total_tickers': len(tickers),
        'successful_tests': len([r for r in portfolio_results.values() if 'error' not in r]),
        'avg_return': total_return / len(tickers) if tickers else 0,
        'initial_capital': initial_capital
    }
    
    return portfolio_results

# Legacy function for backward compatibility
def backtest(signal_file, log_path='./logs/backtest.log'):
    """Legacy backtest function - now calls the comprehensive backtest."""
    results = run_backtest(signal_file)
    
    if 'error' not in results:
        # Save to legacy log format
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'a') as f:
            f.write(f"{datetime.now()} | {os.path.basename(signal_file)} | "
                    f"Return: {results['total_return_pct']:.2f}%, "
                    f"Sharpe: {results['sharpe_ratio']:.2f}, "
                    f"Drawdown: {results['max_drawdown']*100:.2f}%\n")
    
    return results

if __name__ == "__main__":
    # Test backtest functionality
    logger.info("Running backtests...")
    
    # Single ticker backtest
    results = run_backtest('./signals/AAPL_signals.csv')
    if 'error' not in results:
        print(f"AAPL Backtest: {results['total_return_pct']:.2f}% return")
    
    # Portfolio backtest
    portfolio_results = run_portfolio_backtest()
    print(f"Portfolio Summary: {portfolio_results.get('portfolio_summary', {})}")
