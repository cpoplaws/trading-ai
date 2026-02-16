"""
Manual backtest runner to verify the system works.
"""
import pandas as pd
import numpy as np
import os

def simple_backtest_demo():
    """Run a simple demonstration backtest."""
    print("🧪 Manual Backtest Demonstration")
    print("=" * 50)
    
    # Load AAPL signals
    signals_file = './signals/AAPL_signals.csv'
    if not os.path.exists(signals_file):
        print("❌ Signals file not found")
        return
    
    # Load signals
    signals = pd.read_csv(signals_file, index_col=0, parse_dates=True)
    print(f"✅ Loaded {len(signals)} signals")
    
    # Load price data
    price_file = './data/processed/AAPL.csv'
    if not os.path.exists(price_file):
        print("❌ Price file not found")
        return
    
    # Skip problematic header rows
    prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
    print(f"✅ Loaded {len(prices)} price points")
    
    # Ensure we have numeric close prices
    if 'close' in prices.columns:
        prices['close'] = pd.to_numeric(prices['close'], errors='coerce')
        prices = prices.dropna()
    
    # Align dates
    common_dates = signals.index.intersection(prices.index)
    signals = signals.loc[common_dates]
    prices = prices.loc[common_dates]
    
    print(f"✅ Aligned {len(common_dates)} trading days")
    
    # Simple backtest simulation
    initial_capital = 10000
    cash = initial_capital
    shares = 0
    portfolio_values = []
    trades = 0
    
    print("\\n📊 Running Simple Backtest...")
    
    for date in common_dates[:100]:  # Test first 100 days
        signal = signals.loc[date, 'Signal']
        price = prices.loc[date, 'close']
        confidence = signals.loc[date, 'Confidence']
        
        # Simple trading logic
        if signal == 'BUY' and cash > price * 10:  # Buy 10 shares if we can afford it
            shares_to_buy = min(10, int(cash / price))
            cost = shares_to_buy * price
            cash -= cost
            shares += shares_to_buy
            trades += 1
            
        elif signal == 'SELL' and shares > 0:  # Sell half our shares
            shares_to_sell = max(1, shares // 2)
            proceeds = shares_to_sell * price
            cash += proceeds
            shares -= shares_to_sell
            trades += 1
        
        # Calculate portfolio value
        portfolio_value = cash + (shares * price)
        portfolio_values.append(portfolio_value)
    
    # Calculate results
    final_value = portfolio_values[-1] if portfolio_values else initial_capital
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    # Buy and hold comparison
    initial_price = prices.loc[common_dates[0], 'close']
    final_price = prices.loc[common_dates[99], 'close']  # Day 100
    buy_hold_return = (final_price - initial_price) / initial_price * 100
    
    print("\\n📈 Backtest Results:")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print(f"   Final Value: ${final_value:,.2f}")
    print(f"   Total Return: {total_return:.2f}%")
    print(f"   Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"   Excess Return: {total_return - buy_hold_return:.2f}%")
    print(f"   Total Trades: {trades}")
    print(f"   Final Cash: ${cash:,.2f}")
    print(f"   Final Shares: {shares}")
    
    if total_return > buy_hold_return:
        print("\\n🎉 Strategy OUTPERFORMED buy & hold!")
    else:
        print("\\n📉 Strategy underperformed buy & hold")
    
    print("\\n✅ Simple backtest completed successfully!")

if __name__ == "__main__":
    simple_backtest_demo()
