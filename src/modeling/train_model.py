import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Train the Random Forest model to predict UP/DOWN movement
def train_model(file_path, save_path='./models/'):
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Load the dataset
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Create the target column (1 if next day's Close > today's Close, else 0)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Define the features to use for training
    features = ['SMA_10', 'SMA_30', 'RSI_14', 'Volatility_20']
    X = df[features]
    y = df['Target']
    
    # Drop rows with missing values
    X = X.dropna()
    y = y.loc[X.index]
    
    # Initialize and train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the trained model
    model_filename = os.path.join(save_path, f"model_{os.path.basename(file_path).split('.')[0]}.joblib")
    joblib.dump(model, model_filename)
    print(f"Model trained and saved to {model_filename}")

if __name__ == "__main__":
    # Example usage
    train_model('./data/processed/AAPL.csv')
