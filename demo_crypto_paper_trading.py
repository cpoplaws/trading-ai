#!/usr/bin/env python3
"""
Comprehensive Crypto Paper Trading Demo

Demonstrates the complete paper trading infrastructure for blockchain assets:
1. Historical data fetching and preparation
2. Strategy development and testing
3. Paper trading execution
4. Performance analysis and reporting
5. Multi-strategy comparison

Run this to test the complete paper trading system on historical crypto data.
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_paper_trading_engine():
    """Demo the crypto paper trading engine."""
    from src.execution.crypto_paper_trading import CryptoPaperTradingEngine, OrderSide
    
    print("\n" + "="*70)
    print(" "*20 + "PAPER TRADING ENGINE DEMO")
    print("="*70 + "\n")
    
    # Initialize engine
    engine = CryptoPaperTradingEngine(
        initial_capital=100000,
        commission_bps=10,  # 0.1%
        slippage_bps=30,    # 0.3%
        gas_cost_usd=5,
    )
    
    print(f"✅ Initialized with ${engine.initial_capital:,.2f} capital\n")
    
    # Simulate a trading sequence
    print("📊 Simulating Trading Sequence:\n")
    
    # Trade 1: Buy BTC
    print("1. Buying 1 BTC...")
    order1 = engine.place_order(
        symbol="BTC",
        chain="ethereum",
        side=OrderSide.BUY,
        quantity=1.0,
    )
    engine.execute_order(order1, current_price=50000.0)
    
    # Update position
    engine.update_positions({"BTC_ethereum": 51000.0})
    print(f"   Position value: ${51000.0:,.2f}")
    print(f"   Unrealized P&L: ${(51000.0 - 50000.0):,.2f} (+{((51000.0/50000.0 - 1)*100):.2f}%)\n")
    
    # Trade 2: Buy ETH
    print("2. Buying 10 ETH...")
    order2 = engine.place_order(
        symbol="ETH",
        chain="ethereum",
        side=OrderSide.BUY,
        quantity=10.0,
    )
    engine.execute_order(order2, current_price=3000.0)
    
    # Update positions
    engine.update_positions({
        "BTC_ethereum": 52000.0,
        "ETH_ethereum": 3100.0,
    })
    
    print(f"   Portfolio value: ${engine.get_portfolio_value():,.2f}")
    print(f"   Cash remaining: ${engine.cash:,.2f}\n")
    
    # Trade 3: Sell BTC
    print("3. Selling 1 BTC...")
    order3 = engine.place_order(
        symbol="BTC",
        chain="ethereum",
        side=OrderSide.SELL,
        quantity=1.0,
    )
    engine.execute_order(order3, current_price=52000.0)
    
    print(f"   Portfolio value: ${engine.get_portfolio_value():,.2f}")
    print(f"   Cash after sale: ${engine.cash:,.2f}\n")
    
    # Show performance
    engine.record_portfolio_value(datetime.now(), {"ETH_ethereum": 3100.0})
    metrics = engine.get_performance_metrics()
    
    print("📈 Performance Metrics:")
    print(f"   Total Trades: {metrics['total_trades']}")
    print(f"   Winning Trades: {metrics['winning_trades']}")
    print(f"   Win Rate: {metrics['win_rate_pct']:.2f}%")
    print(f"   Total Return: {metrics['total_return_pct']:.2f}%\n")


def demo_historical_data_fetcher():
    """Demo the historical data fetcher."""
    from src.data_ingestion.historical_crypto_data import HistoricalCryptoDataFetcher
    
    print("\n" + "="*70)
    print(" "*20 + "HISTORICAL DATA FETCHER DEMO")
    print("="*70 + "\n")
    
    fetcher = HistoricalCryptoDataFetcher()
    
    # Fetch single asset data
    print("📊 Fetching BTC historical data (30 days)...")
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    btc_data = fetcher.fetch_historical_data(
        symbol='BTCUSDT',
        start_date=start_date,
        end_date=end_date,
        interval='1h',
        source='simulated',
    )
    
    print(f"✅ Fetched {len(btc_data)} hourly candles")
    print(f"   Price range: ${btc_data['Close'].min():,.2f} - ${btc_data['Close'].max():,.2f}")
    print(f"   Average volume: {btc_data['Volume'].mean():,.0f}\n")
    
    # Fetch multi-asset data
    print("📊 Fetching multiple assets...")
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    multi_data = fetcher.fetch_multi_asset_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval='1d',
        source='simulated',
    )
    
    print(f"✅ Fetched data for {len(multi_data)} assets:")
    for symbol, data in multi_data.items():
        returns = data['Close'].pct_change().dropna()
        print(f"   {symbol}: {len(data)} days, "
              f"Return: {((data['Close'].iloc[-1]/data['Close'].iloc[0] - 1)*100):.2f}%, "
              f"Volatility: {(returns.std()*100):.2f}%")
    print()
    
    # Add technical indicators
    print("📈 Adding technical indicators...")
    btc_with_indicators = fetcher.add_technical_indicators(btc_data)
    
    indicators = [col for col in btc_with_indicators.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    print(f"✅ Added {len(indicators)} indicators:")
    print(f"   {', '.join(indicators[:10])}")
    if len(indicators) > 10:
        print(f"   ... and {len(indicators) - 10} more\n")


def demo_full_backtest():
    """Demo the complete backtesting system."""
    from src.backtesting.crypto_backtester import (
        CryptoBacktester,
        simple_sma_crossover_strategy,
        rsi_strategy,
    )
    
    print("\n" + "="*70)
    print(" "*20 + "FULL BACKTEST DEMO")
    print("="*70 + "\n")
    
    # Initialize backtester
    backtester = CryptoBacktester(
        initial_capital=100000,
        commission_bps=10,
        slippage_bps=30,
        gas_cost_usd=5,
    )
    
    print("✅ Initialized backtester with $100,000 capital\n")
    
    # Load historical data
    print("📊 Loading historical data (90 days)...")
    start_date = datetime.now() - timedelta(days=90)
    end_date = datetime.now()
    
    backtester.load_historical_data(
        symbols=['BTCUSDT', 'ETHUSDT'],
        start_date=start_date,
        end_date=end_date,
        interval='1d',
        chain='ethereum',
    )
    
    print(f"✅ Loaded data for {len(backtester.historical_data)} assets\n")
    
    # Run backtest with SMA crossover strategy
    print("🔄 Running backtest with SMA Crossover strategy...")
    results = backtester.run_backtest(simple_sma_crossover_strategy)
    
    metrics = results['metrics']
    print(f"\n📈 Strategy Performance:")
    print(f"   Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"   Total Trades: {metrics['total_trades']}")
    print(f"   Win Rate: {metrics['win_rate_pct']:.2f}%\n")
    
    # Compare multiple strategies
    print("🔄 Comparing multiple strategies...")
    strategies = {
        'SMA Crossover': simple_sma_crossover_strategy,
        'RSI Mean Reversion': rsi_strategy,
    }
    
    comparison = backtester.compare_strategies(strategies)
    
    print("\n📊 Strategy Comparison:")
    print("-" * 70)
    print(f"{'Strategy':<20} {'Return %':<12} {'Sharpe':<10} {'Max DD %':<12} {'Win Rate %':<12}")
    print("-" * 70)
    
    for strategy in comparison.index:
        row = comparison.loc[strategy]
        print(f"{strategy:<20} {row['total_return_pct']:>10.2f}% "
              f"{row['sharpe_ratio']:>9.2f} "
              f"{row['max_drawdown_pct']:>10.2f}% "
              f"{row['win_rate_pct']:>10.2f}%")
    print("-" * 70 + "\n")
    
    # Generate full report
    print("📄 Generating detailed report...\n")
    report = backtester.generate_report()
    print(report)
    
    return backtester


def demo_strategy_suggestions():
    """Demonstrate suggested strategy enhancements."""
    print("\n" + "="*70)
    print(" "*20 + "STRATEGY ENHANCEMENT SUGGESTIONS")
    print("="*70 + "\n")
    
    suggestions = [
        {
            'name': 'Multi-Timeframe Confirmation',
            'description': 'Use signals from multiple timeframes (1h, 4h, 1d) for stronger confirmation',
            'benefit': 'Reduces false signals and improves win rate',
        },
        {
            'name': 'Volatility-Based Position Sizing',
            'description': 'Adjust position sizes based on recent volatility (ATR)',
            'benefit': 'Better risk management and drawdown control',
        },
        {
            'name': 'Correlation-Aware Portfolio',
            'description': 'Consider asset correlations when building portfolio',
            'benefit': 'Improved diversification and risk-adjusted returns',
        },
        {
            'name': 'Dynamic Stop-Loss/Take-Profit',
            'description': 'Adjust stops based on volatility and support/resistance levels',
            'benefit': 'Better capital preservation and profit capture',
        },
        {
            'name': 'Funding Rate Integration',
            'description': 'Use perpetual funding rates as sentiment indicator',
            'benefit': 'Additional edge from derivatives market sentiment',
        },
        {
            'name': 'On-Chain Metrics',
            'description': 'Incorporate whale movements and exchange flows',
            'benefit': 'Early signals from smart money activity',
        },
    ]
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion['name']}")
        print(f"   Description: {suggestion['description']}")
        print(f"   Benefit: {suggestion['benefit']}")
        print()


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*10 + "CRYPTO PAPER TRADING - COMPREHENSIVE DEMO" + " "*16 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        # Demo 1: Paper Trading Engine
        demo_paper_trading_engine()
        
        # Demo 2: Historical Data Fetcher
        demo_historical_data_fetcher()
        
        # Demo 3: Full Backtest
        backtester = demo_full_backtest()
        
        # Demo 4: Strategy Suggestions
        demo_strategy_suggestions()
        
        # Summary
        print("\n" + "="*70)
        print(" "*20 + "DEMO SUMMARY")
        print("="*70 + "\n")
        
        print("✅ Successfully demonstrated:")
        print("   1. Paper trading engine with realistic costs")
        print("   2. Historical data fetching and preparation")
        print("   3. Complete backtesting framework")
        print("   4. Strategy comparison and evaluation")
        print("   5. Performance metrics and reporting")
        print()
        
        print("📚 Next Steps:")
        print("   1. Implement suggested strategy enhancements")
        print("   2. Connect to real data sources (Binance, CoinGecko)")
        print("   3. Add more sophisticated strategies")
        print("   4. Integrate with live paper trading")
        print("   5. Build dashboard for monitoring")
        print()
        
        print("📁 Generated Files:")
        print("   • Backtest results available in backtester object")
        print("   • Use backtester.plot_results() to visualize")
        print("   • Use backtester.generate_report('report.txt') to save")
        print()
        
        # Save report to file
        output_dir = Path("backtests")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"backtest_report_{timestamp}.txt"
        
        backtester.generate_report(str(report_file))
        print(f"💾 Report saved to: {report_file}")
        
        # Try to save plot
        try:
            plot_file = output_dir / f"backtest_plot_{timestamp}.png"
            backtester.plot_results(str(plot_file))
            print(f"📊 Plot saved to: {plot_file}")
        except Exception as e:
            print(f"⚠️  Could not save plot: {e}")
        
        print()
        print("="*70)
        print(" "*20 + "✅ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have all dependencies installed:")
        print("  pip install -r requirements.txt")
        print("  pip install -r requirements-crypto.txt")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
