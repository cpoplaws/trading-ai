"""
Advanced ML Ensemble Trading Example
====================================

Demonstrates how to use multiple advanced ML models together for robust trading predictions:

1. Ensemble Model (XGBoost, LightGBM, RF, MLP) - Base predictions
2. GRU Model - Temporal patterns with attention
3. CNN-LSTM Hybrid - Pattern recognition + temporal
4. VAE Anomaly Detector - Risk management

Strategy:
- Use all models to generate predictions
- Weight predictions by confidence
- Filter trades if VAE detects anomalies
- Combine pattern detection with price forecasting
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ml.advanced_ensemble import AdvancedEnsemble, EnsembleConfig, EnsembleMethod, PredictionTask
from src.ml.gru_predictor import GRUTradingPredictor, GRUConfig
from src.ml.cnn_lstm_hybrid import CNNLSTMPredictor, CNNLSTMConfig
from src.ml.vae_anomaly_detector import VAEAnomalyDetector, VAEConfig, AnomalyLevel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedMLTradingSystem:
    """
    Advanced ML Trading System combining multiple models.

    Models:
    - Ensemble: Robust baseline predictions
    - GRU: Attention-based sequence modeling
    - CNN-LSTM: Pattern recognition
    - VAE: Anomaly detection for risk management
    """

    def __init__(self):
        """Initialize all models."""
        logger.info("Initializing Advanced ML Trading System...")

        self.ensemble = None
        self.gru = None
        self.cnn_lstm = None
        self.vae = None

        self.is_trained = False

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features from OHLCV data.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            Feature array
        """
        df = df.copy()

        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Volatility
        df['volatility_20'] = df['returns'].rolling(20).std()

        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()

        # Technical indicators
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        # Bollinger Bands
        bb_window = 20
        bb_std = df['close'].rolling(bb_window).std()
        bb_mean = df['close'].rolling(bb_window).mean()
        df['bb_upper'] = bb_mean + 2 * bb_std
        df['bb_lower'] = bb_mean - 2 * bb_std
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Volume features
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # Drop NaN rows
        df = df.dropna()

        # Select features for modeling
        feature_cols = [
            'close', 'returns', 'log_returns', 'volatility_20',
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'rsi', 'macd', 'macd_signal', 'bb_position',
            'volume', 'volume_ratio'
        ]

        return df[feature_cols].values

    def train(
        self,
        historical_data: pd.DataFrame,
        train_split: float = 0.7,
        val_split: float = 0.15
    ):
        """
        Train all models.

        Args:
            historical_data: Historical OHLCV data
            train_split: Training data fraction
            val_split: Validation data fraction
        """
        logger.info(f"Training on {len(historical_data)} samples")

        # Prepare features
        features = self.prepare_features(historical_data)
        n_samples = len(features)

        # Split data
        train_end = int(n_samples * train_split)
        val_end = int(n_samples * (train_split + val_split))

        train_data = features[:train_end]
        val_data = features[train_end:val_end]
        test_data = features[val_end:]

        logger.info(f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

        # 1. Train Ensemble Model
        logger.info("\n" + "="*60)
        logger.info("Training Ensemble Model...")
        logger.info("="*60)

        ensemble_config = EnsembleConfig(
            ensemble_method=EnsembleMethod.STACKING,
            prediction_task=PredictionTask.REGRESSION,
            use_feature_engineering=False  # Already engineered
        )

        self.ensemble = AdvancedEnsemble(ensemble_config)

        # Prepare X, y for ensemble
        sequence_length = 1  # Ensemble doesn't need sequences
        X_train = train_data[:, :-1]  # All features except last (target)
        y_train = train_data[:, 0]  # Close price as target

        X_val = val_data[:, :-1]
        y_val = val_data[:, 0]

        ensemble_metrics = self.ensemble.train(X_train, y_train, X_val, y_val)
        logger.info(f"Ensemble metrics: {ensemble_metrics}")

        # 2. Train GRU Model
        logger.info("\n" + "="*60)
        logger.info("Training GRU Model...")
        logger.info("="*60)

        gru_config = GRUConfig(
            hidden_size=64,
            num_layers=2,
            use_attention=True,
            sequence_length=30,
            forecast_horizon=1,
            num_epochs=30
        )

        self.gru = GRUTradingPredictor(input_size=features.shape[1], config=gru_config)
        gru_history = self.gru.train(train_data, val_data, verbose=False)
        logger.info(f"GRU final val loss: {gru_history['val_loss'][-1]:.6f}")

        # 3. Train CNN-LSTM Model
        logger.info("\n" + "="*60)
        logger.info("Training CNN-LSTM Model...")
        logger.info("="*60)

        cnn_lstm_config = CNNLSTMConfig(
            cnn_filters=[32, 64],
            kernel_sizes=[3, 3],
            pool_sizes=[2, 1],
            lstm_hidden_size=64,
            lstm_num_layers=2,
            sequence_length=30,
            num_epochs=30
        )

        self.cnn_lstm = CNNLSTMPredictor(input_size=features.shape[1], config=cnn_lstm_config)
        cnn_lstm_history = self.cnn_lstm.train(train_data, val_data, verbose=False)
        logger.info(f"CNN-LSTM final val loss: {cnn_lstm_history['val_loss'][-1]:.6f}")

        # 4. Train VAE Anomaly Detector (only on normal training data)
        logger.info("\n" + "="*60)
        logger.info("Training VAE Anomaly Detector...")
        logger.info("="*60)

        vae_config = VAEConfig(
            encoder_hidden_dims=[64, 32],
            latent_dim=8,
            decoder_hidden_dims=[32, 64],
            num_epochs=30,
            anomaly_threshold_percentile=95.0
        )

        self.vae = VAEAnomalyDetector(input_dim=features.shape[1], config=vae_config)
        vae_history = self.vae.train(train_data, verbose=False)
        logger.info(f"VAE final recon loss: {vae_history['recon_loss'][-1]:.6f}")

        self.is_trained = True
        logger.info("\n✅ All models trained successfully!")

    def predict(self, recent_data: np.ndarray) -> dict:
        """
        Make comprehensive prediction using all models.

        Args:
            recent_data: Recent market data (sequence_length, features)

        Returns:
            Prediction dictionary with all model outputs
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Call train() first.")

        predictions = {}

        # 1. Ensemble prediction
        ensemble_result = self.ensemble.predict(recent_data[-1])
        predictions['ensemble'] = {
            'price': ensemble_result.predicted_value,
            'confidence': ensemble_result.confidence,
            'individual_models': ensemble_result.individual_predictions
        }

        # 2. GRU prediction
        gru_result = self.gru.predict(recent_data[-self.gru.config.sequence_length:])
        predictions['gru'] = {
            'price': gru_result.predicted_prices[0],
            'confidence': gru_result.confidence
        }

        # 3. CNN-LSTM prediction with patterns
        cnn_lstm_result = self.cnn_lstm.predict(recent_data[-self.cnn_lstm.config.sequence_length:])
        predictions['cnn_lstm'] = {
            'price': cnn_lstm_result.predicted_price,
            'direction': cnn_lstm_result.predicted_direction,
            'confidence': cnn_lstm_result.confidence,
            'patterns': cnn_lstm_result.detected_patterns
        }

        # 4. VAE anomaly detection
        vae_result = self.vae.detect_anomaly(recent_data[-1])
        predictions['vae'] = {
            'is_anomaly': vae_result.is_anomaly,
            'anomaly_level': vae_result.anomaly_level,
            'anomaly_score': vae_result.anomaly_score,
            'reconstruction_error': vae_result.reconstruction_error
        }

        # 5. Meta prediction (weighted average by confidence)
        if not vae_result.is_anomaly:
            weights = {
                'ensemble': predictions['ensemble']['confidence'],
                'gru': predictions['gru']['confidence'],
                'cnn_lstm': predictions['cnn_lstm']['confidence']
            }
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}

            meta_prediction = (
                weights['ensemble'] * predictions['ensemble']['price'] +
                weights['gru'] * predictions['gru']['price'] +
                weights['cnn_lstm'] * predictions['cnn_lstm']['price']
            )

            predictions['meta'] = {
                'price': meta_prediction,
                'weights': weights,
                'is_tradeable': True
            }
        else:
            predictions['meta'] = {
                'price': None,
                'weights': None,
                'is_tradeable': False,
                'reason': f"Anomaly detected: {vae_result.anomaly_level}"
            }

        return predictions

    def generate_trading_signal(self, predictions: dict, current_price: float) -> dict:
        """
        Generate trading signal from predictions.

        Args:
            predictions: Prediction dictionary from predict()
            current_price: Current market price

        Returns:
            Trading signal dictionary
        """
        signal = {
            'action': 'HOLD',
            'confidence': 0.0,
            'reason': '',
            'target_price': None,
            'stop_loss': None
        }

        # Don't trade if anomaly detected
        if not predictions['meta']['is_tradeable']:
            signal['reason'] = predictions['meta']['reason']
            return signal

        meta_price = predictions['meta']['price']
        expected_return = (meta_price - current_price) / current_price

        # Decision thresholds
        BUY_THRESHOLD = 0.02  # 2% expected gain
        SELL_THRESHOLD = -0.02  # 2% expected loss

        # Average confidence
        avg_confidence = np.mean([
            predictions['ensemble']['confidence'],
            predictions['gru']['confidence'],
            predictions['cnn_lstm']['confidence']
        ])

        # Check CNN-LSTM direction
        cnn_direction = predictions['cnn_lstm']['direction']

        if expected_return > BUY_THRESHOLD and cnn_direction == 'up':
            signal['action'] = 'BUY'
            signal['confidence'] = avg_confidence
            signal['reason'] = f"Expected return: {expected_return:.2%}, Direction: {cnn_direction}"
            signal['target_price'] = meta_price
            signal['stop_loss'] = current_price * 0.98  # 2% stop loss

        elif expected_return < SELL_THRESHOLD and cnn_direction == 'down':
            signal['action'] = 'SELL'
            signal['confidence'] = avg_confidence
            signal['reason'] = f"Expected return: {expected_return:.2%}, Direction: {cnn_direction}"
            signal['target_price'] = meta_price
            signal['stop_loss'] = current_price * 1.02  # 2% stop loss

        else:
            signal['action'] = 'HOLD'
            signal['confidence'] = avg_confidence
            signal['reason'] = f"Expected return {expected_return:.2%} below threshold"

        return signal


def main():
    """Main example function."""
    print("\n" + "="*70)
    print("Advanced ML Ensemble Trading System")
    print("="*70)

    # Generate synthetic OHLCV data
    np.random.seed(42)
    n_samples = 2000

    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='1h')

    # Trend + seasonality + noise
    t = np.linspace(0, 100, n_samples)
    trend = 100 + 10 * np.sin(t / 20)
    noise = np.cumsum(np.random.randn(n_samples) * 0.5)
    closes = trend + noise

    # OHLCV
    opens = closes + np.random.randn(n_samples) * 0.2
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n_samples) * 0.3)
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n_samples) * 0.3)
    volumes = np.random.lognormal(10, 0.3, n_samples)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })

    print(f"\nGenerated {len(df)} samples of synthetic OHLCV data")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Create and train system
    system = AdvancedMLTradingSystem()

    print("\nTraining all models (this may take a few minutes)...")
    system.train(df, train_split=0.7, val_split=0.15)

    # Make predictions on recent data
    print("\n" + "="*70)
    print("Making Predictions")
    print("="*70)

    recent_data = system.prepare_features(df.tail(100))
    current_price = recent_data[-1, 0]

    print(f"\nCurrent Price: ${current_price:.2f}")

    predictions = system.predict(recent_data)

    print(f"\n--- Model Predictions ---")
    print(f"Ensemble: ${predictions['ensemble']['price']:.2f} (conf: {predictions['ensemble']['confidence']:.4f})")
    print(f"GRU: ${predictions['gru']['price']:.2f} (conf: {predictions['gru']['confidence']:.4f})")
    print(f"CNN-LSTM: ${predictions['cnn_lstm']['price']:.2f} (dir: {predictions['cnn_lstm']['direction']}, conf: {predictions['cnn_lstm']['confidence']:.4f})")

    if predictions['meta']['is_tradeable']:
        print(f"\nMeta Prediction: ${predictions['meta']['price']:.2f}")
        print(f"Model Weights: {predictions['meta']['weights']}")
    else:
        print(f"\n⚠️  Trading Disabled: {predictions['meta']['reason']}")

    print(f"\n--- Anomaly Detection ---")
    print(f"Is Anomaly: {predictions['vae']['is_anomaly']}")
    print(f"Anomaly Level: {predictions['vae']['anomaly_level']}")
    print(f"Anomaly Score: {predictions['vae']['anomaly_score']:.4f}")

    print(f"\n--- Detected Patterns ---")
    top_patterns = sorted(
        predictions['cnn_lstm']['patterns'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    for pattern, score in top_patterns:
        print(f"{pattern}: {score:.4f}")

    # Generate trading signal
    signal = system.generate_trading_signal(predictions, current_price)

    print(f"\n" + "="*70)
    print(f"Trading Signal")
    print(f"="*70)
    print(f"Action: {signal['action']}")
    print(f"Confidence: {signal['confidence']:.4f}")
    print(f"Reason: {signal['reason']}")
    if signal['target_price']:
        print(f"Target Price: ${signal['target_price']:.2f}")
        print(f"Stop Loss: ${signal['stop_loss']:.2f}")

    print("\n✅ Advanced ML Ensemble Trading Example Complete!")


if __name__ == '__main__':
    main()
