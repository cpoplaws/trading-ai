import pandas as pd
import os
from datetime import datetime

def backtest(signal_file, log_path='./logs/backtest.log'):
    df = pd.read_csv(signal_file, index_col=0, parse_dates=True)

    df['Returns'] = df['Signal'].map({'BUY': 0.01, 'SELL': -0.005})
    df['Equity Curve'] = (1 + df['Returns']).cumprod()

    returns = df['Returns']
    if returns.std() != 0:
        sharpe = returns.mean() / returns.std()
    else:
        sharpe = float('nan')
    summary = {
        "Final Equity": df['Equity Curve'].iloc[-1],
        "Max Drawdown": (df['Equity Curve'].cummax() - df['Equity Curve']).max(),
        "Sharpe Approx": sharpe
    }

    # Save full backtest CSV
    result_file = signal_file.replace('_signals.csv', '_backtest.csv')
    df.to_csv(result_file)
    print(f"Backtest saved to {result_file}")

    # Ensure the directory for log_path exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Log performance
    with open(log_path, 'a') as f:
        f.write(f"{datetime.now()} | {os.path.basename(signal_file)} | "
                f"Equity: {summary['Final Equity']:.2f}, "
                f"Sharpe: {summary['Sharpe Approx']:.2f}, "
                f"Drawdown: {summary['Max Drawdown']:.2f}\n")