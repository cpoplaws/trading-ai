import pandas as pd
import os
from datetime import datetime

def backtest(signal_file, log_path='./logs/backtest.log'):
    df = pd.read_csv(signal_file, index_col=0, parse_dates=True)

    df['Returns'] = df['Signal'].map({'BUY': 0.01, 'SELL': -0.005})
    df['Equity Curve'] = (1 + df['Returns']).cumprod()

    summary = {
        "Final Equity": df['Equity Curve'].iloc[-1],
        "Max Drawdown": (df['Equity Curve'].cummax() - df['Equity Curve']).max(),
        "Sharpe Approx": df['Returns'].mean() / df['Returns'].std() if df['Returns'].std() != 0 else float('nan')
    }

    # Save full backtest CSV
    result_file = signal_file.replace('_signals.csv', '_backtest.csv')
    df.to_csv(result_file)
    print(f"Backtest saved to {result_file}")

    # Log performance
    with open(log_path, 'a') as f:
        f.write(f"{datetime.now()} | {os.path.basename(signal_file)} | "
                f"Equity: {summary['Final Equity']:.2f}, "
                f"Sharpe: {summary['Sharpe Approx']:.2f}, "
                f"Drawdown: {summary['Max Drawdown']:.2f}\n")