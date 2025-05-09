import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model(df=None, file_path=None, save_path='./models/'):
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Load data if not provided
    if df is None:
        if file_path is None:
            raise ValueError("Must provide either a DataFrame or file_path")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Create the target column
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    features = ['SMA_10', 'SMA_30', 'RSI_14', 'Volatility_20']
    X = df[features].dropna()
    y = df.loc[X.index, 'Target']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model using file_path if provided, else fallback
    filename = file_path if file_path else "in_memory"
    model_filename = os.path.join(save_path, f"model_{os.path.basename(filename).split('.')[0]}.joblib")
    joblib.dump(model, model_filename)
    print(f"Model trained and saved to {model_filename}")

if __name__ == "__main__":
    # Example usage
    train_model(file_path='./data/processed/AAPL.csv')
