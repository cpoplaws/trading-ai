import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from typing import Optional, Tuple
import logging

# Set up logging
logger = logging.getLogger(__name__)

def train_model(df: Optional[pd.DataFrame] = None, file_path: Optional[str] = None, 
                save_path: str = './models/', test_size: float = 0.2) -> Tuple[bool, dict]:
    """
    Train a machine learning model for trading signals.
    
    Args:
        df: DataFrame with features and target
        file_path: Path to CSV file with processed data
        save_path: Directory to save the trained model
        test_size: Fraction of data to use for testing
        
    Returns:
        Tuple of (success: bool, metrics: dict)
    """
    try:
        # Ensure the save directory exists
        os.makedirs(save_path, exist_ok=True)

        # Load data if not provided
        if df is None:
            if file_path is None:
                raise ValueError("Must provide either a DataFrame or file_path")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Ensure column names are standardized
        df.columns = df.columns.str.replace(' ', '_')
        
        # Create the target column if it doesn't exist
        if 'Target' not in df.columns and 'target' not in df.columns:
            # Try both uppercase and lowercase close column
            close_col = 'Close' if 'Close' in df.columns else 'close'
            if close_col in df.columns:
                df['Target'] = (df[close_col].shift(-1) > df[close_col]).astype(int)
            else:
                raise ValueError("No 'Close' or 'close' column found for target creation")

        # Define features (use all available technical indicators)
        feature_columns = ['SMA_10', 'SMA_30', 'RSI_14', 'Volatility_20']
        available_features = [col for col in feature_columns if col in df.columns]
        
        if not available_features:
            # Fallback to any numeric columns except target and price columns
            exclude_cols = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'open', 'high', 'low', 'close', 'volume']
            available_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                                if col not in exclude_cols]
        
        if not available_features:
            raise ValueError("No suitable features found for training")

        logger.info(f"Using features: {available_features}")
        
        # Prepare data
        X = df[available_features].dropna()
        y = df.loc[X.index, 'Target']
        
        if len(X) == 0:
            raise ValueError("No valid samples after removing NaN values")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Train model
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': available_features,
            'feature_importance': dict(zip(available_features, model.feature_importances_))
        }
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")

        # Save model
        if file_path:
            filename = os.path.basename(file_path).split('.')[0]
        else:
            filename = "in_memory_model"
            
        model_filename = os.path.join(save_path, f"model_{filename}.joblib")
        joblib.dump(model, model_filename)
        
        # Also save feature list for inference
        feature_filename = os.path.join(save_path, f"features_{filename}.joblib")
        joblib.dump(available_features, feature_filename)
        
        logger.info(f"Model saved to {model_filename}")
        logger.info(f"Features saved to {feature_filename}")
        
        return True, metrics
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return False, {'error': str(e)}

def load_model_and_features(model_path: str) -> Tuple[Optional[object], Optional[list]]:
    """
    Load a trained model and its feature list.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Tuple of (model, feature_list) or (None, None) on error
    """
    try:
        model = joblib.load(model_path)
        
        # Try to load corresponding features
        feature_path = model_path.replace('model_', 'features_')
        try:
            features = joblib.load(feature_path)
        except FileNotFoundError:
            logger.warning(f"Feature file not found: {feature_path}")
            features = None
            
        return model, features
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Example usage
    success, metrics = train_model(file_path='./data/processed/AAPL.csv')
    if success:
        logger.info(f"Training completed successfully. Metrics: {metrics}")
    else:
        logger.error(f"Training failed: {metrics}")
