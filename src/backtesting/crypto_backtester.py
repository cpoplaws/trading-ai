"""
Comprehensive crypto backtesting engine for blockchain assets.

Integrates:
- Historical crypto data fetching
- Paper trading execution
- Strategy evaluation
- Performance analytics
- Multi-chain support
"""
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class CryptoBacktester:
    """
    Comprehensive backtesting engine for crypto trading strategies.
    
    Features:
    - Historical data simulation
    - Paper trading execution
    - Multi-asset portfolio
    - Performance metrics and visualization
    - Strategy comparison
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_bps: float = 10.0,
        slippage_bps: float = 30.0,
        gas_cost_usd: float = 5.0,
    ):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital in USD
            commission_bps: Trading commission in basis points
            slippage_bps: Expected slippage in basis points
            gas_cost_usd: Average gas cost per transaction
        """
        from src.execution.crypto_paper_trading import CryptoPaperTradingEngine
        from src.data_ingestion.historical_crypto_data import HistoricalCryptoDataFetcher
        
        self.paper_trading = CryptoPaperTradingEngine(
            initial_capital=initial_capital,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            gas_cost_usd=gas_cost_usd,
        )
        
        self.data_fetcher = HistoricalCryptoDataFetcher()
        
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.results: Dict[str, Any] = {}
        
        logger.info("Initialized CryptoBacktester")
    
    def load_historical_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = '1h',
        chain: str = 'ethereum',
    ) -> None:
        """
        Load historical data for backtesting.
        
        Args:
            symbols: List of crypto symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
            start_date: Backtest start date
            end_date: Backtest end date
            interval: Data interval
            chain: Blockchain name
        """
        logger.info(f"Loading historical data for {len(symbols)} symbols...")
        
        self.historical_data = self.data_fetcher.fetch_multi_asset_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            source='simulated',  # Can be changed to 'binance' or 'coingecko'
        )
        
        # Add chain information
        self.chain = chain
        
        # Add technical indicators
        for symbol in self.historical_data:
            self.historical_data[symbol] = self.data_fetcher.add_technical_indicators(
                self.historical_data[symbol]
            )
        
        logger.info(f"Loaded data for {len(self.historical_data)} symbols")
    
    def run_backtest(
        self,
        strategy_func: Callable,
        strategy_params: Dict = None,
    ) -> Dict[str, Any]:
        """
        Run backtest with a trading strategy.
        
        Args:
            strategy_func: Strategy function that takes (data, params) and returns signals
            strategy_params: Strategy parameters
            
        Returns:
            Backtest results dictionary
        """
        if not self.historical_data:
            raise ValueError("No historical data loaded. Call load_historical_data() first.")
        
        strategy_params = strategy_params or {}
        
        logger.info("Starting backtest...")
        
        # Reset paper trading engine
        self.paper_trading.reset()
        
        # Get all timestamps (use first symbol as reference)
        first_symbol = list(self.historical_data.keys())[0]
        timestamps = self.historical_data[first_symbol].index
        
        # Run backtest bar by bar
        for i, timestamp in enumerate(timestamps):
            # Skip initial bars for indicator warmup
            if i < 50:
                continue
            
            # Get current prices
            current_prices = {}
            for symbol, data in self.historical_data.items():
                if timestamp in data.index:
                    current_prices[f"{symbol}_{self.chain}"] = data.loc[timestamp, 'Close']
            
            # Generate signals for each asset
            for symbol, data in self.historical_data.items():
                if timestamp not in data.index:
                    continue
                
                # Get data up to current timestamp
                hist_data = data.loc[:timestamp]
                
                # Generate trading signal
                signal = strategy_func(hist_data, strategy_params)
                
                if signal is None:
                    continue
                
                # Execute trades based on signal
                if signal['action'] == 'BUY':
                    # Calculate position size
                    portfolio_value = self.paper_trading.get_portfolio_value(current_prices)
                    position_size = signal.get('position_size', 0.1)  # Default 10%
                    trade_value = portfolio_value * position_size
                    
                    current_price = current_prices[f"{symbol}_{self.chain}"]
                    quantity = trade_value / current_price
                    
                    # Place and execute buy order
                    from src.execution.crypto_paper_trading import OrderSide
                    order = self.paper_trading.place_order(
                        symbol=symbol,
                        chain=self.chain,
                        side=OrderSide.BUY,
                        quantity=quantity,
                    )
                    
                    self.paper_trading.execute_order(order, current_price, timestamp)
                
                elif signal['action'] == 'SELL':
                    # Check if we have a position
                    position_key = f"{symbol}_{self.chain}"
                    if position_key in self.paper_trading.positions:
                        position = self.paper_trading.positions[position_key]
                        
                        # Place and execute sell order
                        from src.execution.crypto_paper_trading import OrderSide
                        order = self.paper_trading.place_order(
                            symbol=symbol,
                            chain=self.chain,
                            side=OrderSide.SELL,
                            quantity=position.quantity,
                        )
                        
                        current_price = current_prices[f"{symbol}_{self.chain}"]
                        self.paper_trading.execute_order(order, current_price, timestamp)
            
            # Record portfolio value
            self.paper_trading.record_portfolio_value(timestamp, current_prices)
        
        # Close all positions at end
        final_prices = {
            f"{symbol}_{self.chain}": data.iloc[-1]['Close']
            for symbol, data in self.historical_data.items()
        }
        self.paper_trading.close_all_positions(final_prices)
        
        # Calculate performance metrics
        metrics = self.paper_trading.get_performance_metrics()
        
        self.results = {
            'metrics': metrics,
            'portfolio_values': self.paper_trading.portfolio_values,
            'trades': self.paper_trading.trade_history,
            'positions': self.paper_trading.get_open_positions(),
        }
        
        logger.info("Backtest completed")
        return self.results
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot backtest results.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.results:
            logger.warning("No results to plot. Run backtest first.")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Portfolio value over time
        timestamps, values = zip(*self.results['portfolio_values'])
        axes[0].plot(timestamps, values, label='Portfolio Value', linewidth=2)
        axes[0].axhline(y=self.paper_trading.initial_capital, color='gray', 
                       linestyle='--', label='Initial Capital')
        axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Value ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        values_series = pd.Series(values, index=timestamps)
        cummax = values_series.cummax()
        drawdown = (values_series - cummax) / cummax * 100
        axes[1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        axes[1].plot(drawdown.index, drawdown, color='darkred', linewidth=1)
        axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)
        
        # Returns distribution
        returns = values_series.pct_change().dropna()
        axes[2].hist(returns * 100, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[2].axvline(x=returns.mean() * 100, color='red', linestyle='--', 
                       label=f'Mean: {returns.mean()*100:.2f}%')
        axes[2].set_title('Returns Distribution', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Return (%)')
        axes[2].set_ylabel('Frequency')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate detailed backtest report.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Report as string
        """
        if not self.results:
            return "No results available. Run backtest first."
        
        metrics = self.results['metrics']
        
        report = f"""
{'='*60}
CRYPTO BACKTESTING REPORT
{'='*60}

PERFORMANCE SUMMARY
{'-'*60}
Initial Capital:        ${metrics['initial_capital']:,.2f}
Final Value:           ${metrics['final_value']:,.2f}
Total Return:          {metrics['total_return_pct']:,.2f}%
Max Drawdown:          {metrics['max_drawdown_pct']:.2f}%
Sharpe Ratio:          {metrics['sharpe_ratio']:.2f}

TRADING STATISTICS
{'-'*60}
Total Trades:          {metrics['total_trades']}
Winning Trades:        {metrics['winning_trades']}
Losing Trades:         {metrics['losing_trades']}
Win Rate:              {metrics['win_rate_pct']:.2f}%

RISK METRICS
{'-'*60}
Average Return:        {metrics['avg_return_pct']:.4f}%
Volatility:            {metrics['volatility_pct']:.2f}%

RECENT TRADES
{'-'*60}
"""
        
        # Add recent trades
        recent_trades = self.results['trades'][-10:]  # Last 10 trades
        for trade in recent_trades:
            report += f"\n{trade['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
            report += f"{trade['side'].upper()} {trade['quantity']:.4f} {trade['symbol']} "
            report += f"@ ${trade['price']:.2f}"
        
        report += f"\n\n{'='*60}\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def compare_strategies(
        self,
        strategies: Dict[str, Callable],
        strategy_params: Dict[str, Dict] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.
        
        Args:
            strategies: Dict mapping strategy names to strategy functions
            strategy_params: Dict mapping strategy names to their parameters
            
        Returns:
            DataFrame with comparison results
        """
        strategy_params = strategy_params or {}
        results = []
        
        for strategy_name, strategy_func in strategies.items():
            logger.info(f"Testing strategy: {strategy_name}")
            
            params = strategy_params.get(strategy_name, {})
            backtest_results = self.run_backtest(strategy_func, params)
            
            metrics = backtest_results['metrics']
            metrics['strategy'] = strategy_name
            results.append(metrics)
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.set_index('strategy')
        
        return comparison_df


# Example strategy functions
def simple_sma_crossover_strategy(data: pd.DataFrame, params: Dict) -> Optional[Dict]:
    """
    Simple SMA crossover strategy.
    
    Buy when fast SMA crosses above slow SMA.
    Sell when fast SMA crosses below slow SMA.
    """
    if len(data) < 50:
        return None
    
    # Get current and previous values
    current = data.iloc[-1]
    previous = data.iloc[-2]
    
    # Check for crossover
    if previous['SMA_20'] <= previous['SMA_50'] and current['SMA_20'] > current['SMA_50']:
        return {'action': 'BUY', 'position_size': 0.2}
    elif previous['SMA_20'] >= previous['SMA_50'] and current['SMA_20'] < current['SMA_50']:
        return {'action': 'SELL'}
    
    return None


def rsi_strategy(data: pd.DataFrame, params: Dict) -> Optional[Dict]:
    """
    RSI-based mean reversion strategy.
    
    Buy when RSI < 30 (oversold).
    Sell when RSI > 70 (overbought).
    """
    if len(data) < 20:
        return None
    
    current_rsi = data.iloc[-1]['RSI']
    
    if current_rsi < 30:
        return {'action': 'BUY', 'position_size': 0.15}
    elif current_rsi > 70:
        return {'action': 'SELL'}
    
    return None


if __name__ == "__main__":
    # Example usage
    print("=== Crypto Backtester Test ===\n")
    
    backtester = CryptoBacktester(initial_capital=100000)
    
    # Load historical data
    start_date = datetime.now() - timedelta(days=90)
    end_date = datetime.now()
    
    backtester.load_historical_data(
        symbols=['BTCUSDT', 'ETHUSDT'],
        start_date=start_date,
        end_date=end_date,
        interval='1d',
        chain='ethereum',
    )
    
    # Run backtest with SMA crossover strategy
    results = backtester.run_backtest(simple_sma_crossover_strategy)
    
    # Print results
    print(backtester.generate_report())
    
    # Compare strategies
    strategies = {
        'SMA Crossover': simple_sma_crossover_strategy,
        'RSI Mean Reversion': rsi_strategy,
    }
    
    comparison = backtester.compare_strategies(strategies)
    print("\nStrategy Comparison:")
    print(comparison[['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate_pct']])
    
    print("\n✅ Crypto backtester test completed!")
