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

    # Optional: Add prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        df['Confidence'] = model.predict_proba(df[features])[:, 1]

    # Keep only necessary columns for signal log
    df_out = df[['Signal']].copy()
    if 'Confidence' in df:
        df_out['Confidence'] = df['Confidence']

    filename = os.path.basename(data_path).replace('.csv', '_signals.csv')
    output_path = os.path.join(save_path, filename)
    df_out.to_csv(output_path)
    print(f"Signals saved to {output_path}")

if __name__ == "__main__":
    generate_signals('./models/model_AAPL.joblib', './data/processed/AAPL.csv')
