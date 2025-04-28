import joblib
import pandas as pd
import os

def generate_signals(model_path, data_path, save_path='./signals/'):
    os.makedirs(save_path, exist_ok=True)
    model = joblib.load(model_path)
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    features = ['SMA_10', 'SMA_30', 'RSI_14', 'Volatility_20']

    df['Prediction'] = model.predict(df[features])
    df['Signal'] = df['Prediction'].map({1: 'BUY', 0: 'SELL'})

    filename = os.path.basename(data_path).replace('.csv', '_signals.csv')
    df[['Signal']].to_csv(os.path.join(save_path, filename))
    print(f"Signals saved to {os.path.join(save_path, filename)}")

if __name__ == "__main__":
    generate_signals('./models/model_AAPL.joblib', './data/processed/AAPL.csv')
