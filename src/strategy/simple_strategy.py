"""Trading strategy module for signal generation and analysis."""
import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger(__name__)

def generate_signals(model_path: str, data_path: str, save_path: str = './signals/') -> bool:
    """
    Generate trading signals using a trained model.
    
    Args:
        model_path: Path to the trained model file
        data_path: Path to the processed data file
        save_path: Directory to save signal files
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(save_path, exist_ok=True)
        
        # Load model
        model = joblib.load(model_path)
        
        # Try to load features list
        feature_path = model_path.replace('model_', 'features_')
        try:
            features = joblib.load(feature_path)
            logger.info(f"Using saved feature list: {features}")
        except FileNotFoundError:
            # Fallback to default features
            features = ['SMA_10', 'SMA_30', 'RSI_14', 'Volatility_20']
            logger.warning(f"Feature file not found, using default: {features}")
        
        # Load data
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Ensure we have the required features
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing features in data: {missing_features}")
            return False
        
        # Generate predictions
        feature_data = df[features].dropna()
        if len(feature_data) == 0:
            logger.error("No valid data points for prediction")
            return False
            
        predictions = model.predict(feature_data)
        
        # Create signals DataFrame
        signals_df = pd.DataFrame(index=feature_data.index)
        signals_df['Prediction'] = predictions
        signals_df['Signal'] = signals_df['Prediction'].map({1: 'BUY', 0: 'SELL'})

        # Add prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(feature_data)
            signals_df['Confidence'] = probabilities[:, 1]  # Probability of positive class
            signals_df['Buy_Confidence'] = probabilities[:, 1]
            signals_df['Sell_Confidence'] = probabilities[:, 0]
        
        # Add current price for reference
        if 'Close' in df.columns:
            signals_df['Price'] = df.loc[feature_data.index, 'Close']
        elif 'close' in df.columns:
            signals_df['Price'] = df.loc[feature_data.index, 'close']
            
        # Add additional signal metadata
        signals_df['Timestamp'] = signals_df.index
        signals_df['Signal_Strength'] = 'MEDIUM'  # Default, can be enhanced
        
        # Enhance signal strength based on confidence if available
        if 'Confidence' in signals_df.columns:
            signals_df.loc[signals_df['Confidence'] > 0.8, 'Signal_Strength'] = 'STRONG'
            signals_df.loc[signals_df['Confidence'] < 0.6, 'Signal_Strength'] = 'WEAK'

        # Save signals
        filename = os.path.basename(data_path).replace('.csv', '_signals.csv')
        output_path = os.path.join(save_path, filename)
        signals_df.to_csv(output_path)
        
        logger.info(f"Generated {len(signals_df)} signals")
        logger.info(f"Signal distribution: {signals_df['Signal'].value_counts().to_dict()}")
        logger.info(f"Signals saved to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        return False

def analyze_signals(signal_path: str) -> Dict[str, Any]:
    """
    Analyze generated signals for basic statistics.
    
    Args:
        signal_path: Path to the signals CSV file
        
    Returns:
        Dictionary with signal analysis
    """
    try:
        df = pd.read_csv(signal_path, index_col=0, parse_dates=True)
        
        analysis = {
            'total_signals': len(df),
            'buy_signals': len(df[df['Signal'] == 'BUY']),
            'sell_signals': len(df[df['Signal'] == 'SELL']),
            'buy_percentage': (len(df[df['Signal'] == 'BUY']) / len(df)) * 100,
        }
        
        if 'Confidence' in df.columns:
            analysis.update({
                'avg_confidence': df['Confidence'].mean(),
                'high_confidence_signals': len(df[df['Confidence'] > 0.8]),
                'low_confidence_signals': len(df[df['Confidence'] < 0.6]),
            })
            
        if 'Signal_Strength' in df.columns:
            analysis['signal_strength_dist'] = df['Signal_Strength'].value_counts().to_dict()
            
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing signals: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    # Example usage
    model_file = './models/model_AAPL.joblib'
    data_file = './data/processed/AAPL.csv'
    
    success = generate_signals(model_file, data_file)
    if success:
        analysis = analyze_signals('./signals/AAPL_signals.csv')
        logger.info(f"Signal analysis: {analysis}")
    else:
        logger.error("Failed to generate signals")
