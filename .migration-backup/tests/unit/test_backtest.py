"""
Simple script to test and run backtests.
"""
import sys
import os
sys.path.append('./src')

from backtesting.backtest import run_backtest, run_portfolio_backtest

def main():
    print("🧪 Testing Comprehensive Backtesting System...")
    
    # Test single ticker backtest
    print("\n📊 Running AAPL Backtest...")
    try:
        results = run_backtest('./signals/AAPL_signals.csv', initial_capital=10000)
        
        if 'error' not in results:
            print("✅ AAPL Backtest Results:")
            print(f"   Initial Capital: ${results['initial_capital']:,.2f}")
            print(f"   Final Value: ${results['final_value']:,.2f}")
            print(f"   Total Return: {results['total_return_pct']:.2f}%")
            print(f"   Buy & Hold Return: {results['buy_hold_return_pct']:.2f}%")
            print(f"   Excess Return: {results['excess_return']*100:.2f}%")
            print(f"   Sharpe Ratio: {results['sharpe_ratio']:.3f}")
            print(f"   Max Drawdown: {results['max_drawdown']*100:.2f}%")
            print(f"   Win Rate: {results['win_rate_pct']:.1f}%")
            print(f"   Total Trades: {results['num_trades']}")
            print(f"   Volatility: {results['volatility']*100:.2f}%")
        else:
            print(f"❌ AAPL Backtest Error: {results['error']}")
            
    except Exception as e:
        print(f"❌ Exception in AAPL backtest: {str(e)}")
    
    # Test portfolio backtest
    print("\n🎯 Running Portfolio Backtest...")
    try:
        portfolio_results = run_portfolio_backtest(['AAPL', 'MSFT', 'SPY'], initial_capital=30000)
        
        print("✅ Portfolio Backtest Results:")
        for ticker, results in portfolio_results.items():
            if ticker != 'portfolio_summary' and 'error' not in results:
                print(f"   {ticker}: {results['total_return_pct']:.2f}% return, {results['num_trades']} trades")
            elif ticker != 'portfolio_summary':
                print(f"   {ticker}: Error - {results.get('error', 'Unknown')}")
        
        summary = portfolio_results.get('portfolio_summary', {})
        if summary:
            print(f"\n📈 Portfolio Summary:")
            print(f"   Total Tickers: {summary.get('total_tickers', 0)}")
            print(f"   Successful Tests: {summary.get('successful_tests', 0)}")
            print(f"   Average Return: {summary.get('avg_return', 0)*100:.2f}%")
            
    except Exception as e:
        print(f"❌ Exception in portfolio backtest: {str(e)}")
    
    print("\n🎉 Backtesting complete! Check ./backtests/ for detailed reports.")

if __name__ == "__main__":
    main()
