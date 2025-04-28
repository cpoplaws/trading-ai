# backtest.py - Simple backtest engine
import pandas as pd

def backtest(signal_file):
    df = pd.read_csv(signal_file, index_col=0, parse_dates=True)

    df['Returns'] = df['Signal'].map({'BUY': 0.01, 'SELL': -0.005})  # Simulate rough returns
    df['Equity Curve'] = (1 + df['Returns']).cumprod()

    print(f"Cumulative return: {df['Equity Curve'].iloc[-1]:.2f}x")
    df.to_csv(signal_file.replace('_signals.csv', '_backtest.csv'))
    print(f"Backtest results saved.")

if __name__ == "__main__":
    backtest('./signals/AAPL_signals.csv')