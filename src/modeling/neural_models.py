"""
Advanced Neural Network Models for Trading AI

This module implements LSTM and other neural network architectures for 
time series prediction and trading signal generation.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)

class LSTMTradingModel:
    """
    LSTM Neural Network for trading signal prediction.
    
    This model uses sequences of price and technical indicators to predict
    future price movements and generate trading signals.
    """
    
    def __init__(self, sequence_length: int = 60, features: List[str] = None):
        """
        Initialize the LSTM Trading Model.
        
        Args:
            sequence_length: Number of time steps to look back for prediction
            features: List of feature column names to use
        """
        self.sequence_length = sequence_length
        self.features = features or ['close', 'volume', 'sma_10', 'sma_30', 'rsi_14', 'volatility_20']
        self.model = None
        self.scaler = None
        self.target_scaler = None
        self.is_fitted = False
        
    def _prepare_sequences(self, data: pd.DataFrame, target_col: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        
        Args:
            data: DataFrame with features and target
            target_col: Name of the target column
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        # Ensure we have the required features
        available_features = [f for f in self.features if f in data.columns]
        if not available_features:
            raise ValueError(f"None of the required features {self.features} found in data")
        
        logger.info(f"Using features: {available_features}")
        self.features = available_features  # Update to only available features
        
        # Scale the features
        self.scaler = MinMaxScaler()
        scaled_features = self.scaler.fit_transform(data[self.features])
        
        # Prepare target
        target_data = data[target_col].values
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(target_data[i])
        
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)
            
        Returns:
            Compiled LSTM model
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(25, return_sequences=False),
            Dropout(0.2),
            
            Dense(25, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, data: pd.DataFrame, target_col: str = 'target', 
              test_size: float = 0.2, epochs: int = 100, batch_size: int = 32,
              validation_split: float = 0.1) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            data: DataFrame with features and target
            target_col: Name of the target column
            test_size: Fraction of data for testing
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of training data for validation
            
        Returns:
            Dictionary with training metrics
        """
        try:
            logger.info("Preparing sequences for LSTM training...")
            
            # Prepare sequences
            X, y = self._prepare_sequences(data, target_col)
            
            if len(X) == 0:
                raise ValueError("No sequences created. Check data length and sequence_length.")
            
            logger.info(f"Created {len(X)} sequences with shape {X.shape}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Build model
            self.model = self._build_model((self.sequence_length, len(self.features)))
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
            ]
            
            logger.info("Training LSTM model...")
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            test_predictions = self.model.predict(X_test)
            test_pred_binary = (test_predictions > 0.5).astype(int).flatten()
            
            accuracy = accuracy_score(y_test, test_pred_binary)
            
            self.is_fitted = True
            
            metrics = {
                'accuracy': accuracy,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features_used': self.features,
                'sequence_length': self.sequence_length,
                'final_train_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'best_val_accuracy': max(history.history['val_accuracy'])
            }
            
            logger.info(f"LSTM Model accuracy: {accuracy:.4f}")
            logger.info(f"Best validation accuracy: {metrics['best_val_accuracy']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}")
            return {'error': str(e)}
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            data: DataFrame with features
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare sequences
        available_features = [f for f in self.features if f in data.columns]
        if not available_features:
            raise ValueError(f"Required features not found in data: {self.features}")
        
        scaled_features = self.scaler.transform(data[available_features])
        
        # Create sequences
        X = []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
        
        if len(X) == 0:
            return np.array([])
        
        X = np.array(X)
        predictions = self.model.predict(X)
        
        return predictions.flatten()
    
    def save(self, save_path: str, model_name: str = "lstm_model"):
        """
        Save the trained model and scaler.
        
        Args:
            save_path: Directory to save the model
            model_name: Name for the model files
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save the Keras model
        model_path = os.path.join(save_path, f"{model_name}.h5")
        self.model.save(model_path)
        
        # Save the scaler and metadata
        metadata = {
            'scaler': self.scaler,
            'features': self.features,
            'sequence_length': self.sequence_length
        }
        metadata_path = os.path.join(save_path, f"{model_name}_metadata.joblib")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"LSTM model saved to {model_path}")
        logger.info(f"Model metadata saved to {metadata_path}")
    
    def load(self, save_path: str, model_name: str = "lstm_model"):
        """
        Load a trained model and scaler.
        
        Args:
            save_path: Directory containing the model
            model_name: Name of the model files
        """
        try:
            # Load the Keras model
            model_path = os.path.join(save_path, f"{model_name}.h5")
            self.model = tf.keras.models.load_model(model_path)
            
            # Load the scaler and metadata
            metadata_path = os.path.join(save_path, f"{model_name}_metadata.joblib")
            metadata = joblib.load(metadata_path)
            
            self.scaler = metadata['scaler']
            self.features = metadata['features']
            self.sequence_length = metadata['sequence_length']
            self.is_fitted = True
            
            logger.info(f"LSTM model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading LSTM model: {str(e)}")
            raise


class HybridTradingModel:
    """
    Hybrid model combining LSTM and traditional ML approaches.
    """
    
    def __init__(self, sequence_length: int = 60):
        """
        Initialize the hybrid model.
        
        Args:
            sequence_length: Sequence length for LSTM component
        """
        self.lstm_model = LSTMTradingModel(sequence_length=sequence_length)
        self.traditional_model = None  # Will be RandomForest from existing code
        self.ensemble_weights = {'lstm': 0.6, 'traditional': 0.4}
        self.is_fitted = False
    
    def train(self, data: pd.DataFrame, target_col: str = 'target') -> Dict:
        """
        Train both LSTM and traditional models.
        
        Args:
            data: Training data
            target_col: Target column name
            
        Returns:
            Training metrics
        """
        try:
            # Train LSTM model
            lstm_metrics = self.lstm_model.train(data, target_col)
            
            # Train traditional model (simplified Random Forest)
            from sklearn.ensemble import RandomForestClassifier
            
            feature_columns = ['sma_10', 'sma_30', 'rsi_14', 'volatility_20']
            available_features = [col for col in feature_columns if col in data.columns]
            
            if available_features:
                X = data[available_features].dropna()
                y = data.loc[X.index, target_col]
                
                self.traditional_model = RandomForestClassifier(
                    n_estimators=100, random_state=42, max_depth=10
                )
                self.traditional_model.fit(X, y)
                
                # Quick evaluation
                rf_pred = self.traditional_model.predict(X)
                rf_accuracy = accuracy_score(y, rf_pred)
                
                self.is_fitted = True
                
                combined_metrics = {
                    'lstm_metrics': lstm_metrics,
                    'rf_accuracy': rf_accuracy,
                    'ensemble_weights': self.ensemble_weights
                }
                
                logger.info(f"Hybrid model trained. LSTM: {lstm_metrics.get('accuracy', 0):.4f}, RF: {rf_accuracy:.4f}")
                
                return combined_metrics
            else:
                logger.warning("No features available for traditional model")
                return lstm_metrics
                
        except Exception as e:
            logger.error(f"Error training hybrid model: {str(e)}")
            return {'error': str(e)}
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            data: Input data
            
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        
        # LSTM predictions
        try:
            lstm_pred = self.lstm_model.predict(data)
            if len(lstm_pred) > 0:
                predictions.append(('lstm', lstm_pred))
        except Exception as e:
            logger.warning(f"LSTM prediction failed: {str(e)}")
        
        # Traditional model predictions
        try:
            if self.traditional_model is not None:
                feature_columns = ['sma_10', 'sma_30', 'rsi_14', 'volatility_20']
                available_features = [col for col in feature_columns if col in data.columns]
                
                if available_features:
                    X = data[available_features].ffill().fillna(0)
                    rf_pred = self.traditional_model.predict_proba(X)[:, 1]
                    
                    # Align with LSTM predictions if needed
                    if len(predictions) > 0:
                        lstm_len = len(predictions[0][1])
                        if len(rf_pred) > lstm_len:
                            rf_pred = rf_pred[-lstm_len:]
                        elif len(rf_pred) < lstm_len:
                            # Pad with last known value
                            padding = np.full(lstm_len - len(rf_pred), rf_pred[-1] if len(rf_pred) > 0 else 0.5)
                            rf_pred = np.concatenate([padding, rf_pred])
                    
                    predictions.append(('traditional', rf_pred))
        except Exception as e:
            logger.warning(f"Traditional model prediction failed: {str(e)}")
        
        # Ensemble predictions
        if len(predictions) == 0:
            return np.array([])
        elif len(predictions) == 1:
            return predictions[0][1]
        else:
            # Weighted ensemble
            lstm_pred = predictions[0][1] if predictions[0][0] == 'lstm' else predictions[1][1]
            rf_pred = predictions[1][1] if predictions[1][0] == 'traditional' else predictions[0][1]
            
            ensemble_pred = (self.ensemble_weights['lstm'] * lstm_pred + 
                           self.ensemble_weights['traditional'] * rf_pred)
            
            return ensemble_pred
    
    def save(self, save_path: str, model_name: str = "hybrid_model"):
        """
        Save the hybrid model components.
        
        Args:
            save_path: Directory to save the model
            model_name: Name for the model files
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save LSTM component
        lstm_model_name = f"{model_name}_lstm"
        self.lstm_model.save(save_path, lstm_model_name)
        
        # Save traditional model component using joblib
        if self.traditional_model is not None:
            traditional_model_path = os.path.join(save_path, f"{model_name}_traditional.joblib")
            joblib.dump(self.traditional_model, traditional_model_path)
            logger.info(f"Traditional model saved to {traditional_model_path}")
        
        # Save hybrid model metadata
        hybrid_metadata = {
            'ensemble_weights': self.ensemble_weights,
            'model_name': model_name,
            'has_traditional_model': self.traditional_model is not None
        }
        
        metadata_path = os.path.join(save_path, f"{model_name}_hybrid_metadata.joblib")
        joblib.dump(hybrid_metadata, metadata_path)
        
        logger.info(f"Hybrid model saved to {save_path}")
        logger.info(f"Hybrid metadata saved to {metadata_path}")
    
    def load(self, save_path: str, model_name: str = "hybrid_model"):
        """
        Load the hybrid model components.
        
        Args:
            save_path: Directory containing the model
            model_name: Name of the model files
        """
        try:
            # Load LSTM component
            lstm_model_name = f"{model_name}_lstm"
            self.lstm_model.load(save_path, lstm_model_name)
            
            # Load traditional model component
            traditional_model_path = os.path.join(save_path, f"{model_name}_traditional.joblib")
            if os.path.exists(traditional_model_path):
                self.traditional_model = joblib.load(traditional_model_path)
                logger.info(f"Traditional model loaded from {traditional_model_path}")
            else:
                self.traditional_model = None
                logger.info("No traditional model component found")
            
            # Load hybrid metadata
            metadata_path = os.path.join(save_path, f"{model_name}_hybrid_metadata.joblib")
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.ensemble_weights = metadata.get('ensemble_weights', {'lstm': 0.6, 'traditional': 0.4})
            
            self.is_fitted = True
            logger.info(f"Hybrid model loaded from {save_path}")
            
        except Exception as e:
            logger.error(f"Error loading hybrid model: {str(e)}")
            raise


def train_neural_model(df: Optional[pd.DataFrame] = None, file_path: Optional[str] = None,
                      save_path: str = './models/', model_type: str = 'lstm',
                      sequence_length: int = 60) -> Tuple[bool, dict]:
    """
    Train a neural network model for trading signals.
    
    Args:
        df: DataFrame with features and target
        file_path: Path to CSV file with processed data
        save_path: Directory to save the trained model
        model_type: Type of model ('lstm', 'hybrid')
        sequence_length: Sequence length for LSTM
        
    Returns:
        Tuple of (success: bool, metrics: dict)
    """
    try:
        # Load data if not provided
        if df is None:
            if file_path is None:
                raise ValueError("Must provide either a DataFrame or file_path")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Create target if needed
        if 'target' not in df.columns:
            if 'close' in df.columns:
                df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            else:
                raise ValueError("No 'close' column found for target creation")
        
        # Remove rows with NaN target
        df = df.dropna(subset=['target'])
        
        if len(df) < sequence_length + 100:  # Minimum data requirement
            raise ValueError(f"Insufficient data: need at least {sequence_length + 100} rows")
        
        # Initialize and train model
        if model_type == 'lstm':
            model = LSTMTradingModel(sequence_length=sequence_length)
        elif model_type == 'hybrid':
            model = HybridTradingModel(sequence_length=sequence_length)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        metrics = model.train(df, 'target')
        
        if 'error' not in metrics:
            # Save model
            if file_path:
                filename = os.path.basename(file_path).split('.')[0]
            else:
                filename = "in_memory_model"
            
            model_name = f"{model_type}_{filename}"
            model.save(save_path, model_name)
            
            metrics['model_name'] = model_name
            metrics['model_type'] = model_type
            
            return True, metrics
        else:
            return False, metrics
            
    except Exception as e:
        logger.error(f"Error training neural model: {str(e)}")
        return False, {'error': str(e)}


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('/workspaces/trading-ai/src')
    
    # Test with sample data
    success, metrics = train_neural_model(
        file_path='/workspaces/trading-ai/data/processed/AAPL.csv',
        model_type='lstm'
    )
    
    if success:
        logger.info(f"Neural model training completed successfully. Metrics: {metrics}")
    else:
        logger.error(f"Neural model training failed: {metrics}")
