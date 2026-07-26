"""
ML Model Server
Loads and serves predictions from trained ML models
"""
import os
import logging
import pickle
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class ModelServer:
    """Serves ML model predictions with caching"""

    def __init__(self, models_dir: str = "../../models"):
        self.models_dir = models_dir
        self.models = {}
        self.cache = {}
        self.cache_ttl = 60  # Cache predictions for 60 seconds

        logger.info(f"ModelServer initialized with models_dir: {models_dir}")

    def load_model(self, model_name: str):
        """Load a trained model from disk"""
        try:
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")

            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                return None

            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            self.models[model_name] = model
            logger.info(f"✅ Loaded model: {model_name}")
            return model

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None

    def load_all_models(self):
        """Load all available models"""
        model_names = [
            "lstm_model",
            "gru_model",
            "transformer_model",
            "dqn_model",
            "ensemble_model"
        ]

        for model_name in model_names:
            self.load_model(model_name)

        logger.info(f"Loaded {len(self.models)} models")

    def predict(self, model_name: str, features: np.ndarray) -> Optional[float]:
        """
        Get prediction from a model

        Args:
            model_name: Name of the model
            features: Input features

        Returns:
            Prediction value or None
        """
        # Check cache first
        cache_key = f"{model_name}_{hash(features.tobytes())}"
        if cache_key in self.cache:
            cached_pred, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_pred

        # Load model if not loaded
        if model_name not in self.models:
            model = self.load_model(model_name)
            if model is None:
                return None
        else:
            model = self.models[model_name]

        try:
            # Make prediction
            prediction = model.predict(features)

            # Cache the result
            self.cache[cache_key] = (prediction, datetime.now())

            return prediction

        except Exception as e:
            logger.error(f"Error predicting with {model_name}: {e}")
            return None

    def predict_ensemble(
        self,
        features: np.ndarray,
        models: List[str],
        weights: Optional[List[float]] = None
    ) -> Optional[float]:
        """
        Get ensemble prediction from multiple models

        Args:
            features: Input features
            models: List of model names
            weights: Optional weights for each model (must sum to 1.0)

        Returns:
            Weighted average prediction or None
        """
        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        if len(weights) != len(models):
            logger.error("Weights length must match models length")
            return None

        predictions = []
        for model_name in models:
            pred = self.predict(model_name, features)
            if pred is not None:
                predictions.append(pred)
            else:
                logger.warning(f"Failed to get prediction from {model_name}")

        if not predictions:
            return None

        # Weighted average
        ensemble_pred = sum(p * w for p, w in zip(predictions, weights[:len(predictions)]))

        return ensemble_pred

    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            "loaded_models": list(self.models.keys()),
            "total_models": len(self.models),
            "cache_size": len(self.cache),
            "models_dir": self.models_dir
        }


class FeatureExtractor:
    """Extract features from market data for ML models"""

    @staticmethod
    def extract_features(
        prices: List[float],
        volumes: Optional[List[float]] = None,
        lookback: int = 20
    ) -> np.ndarray:
        """
        Extract technical features from price data

        Args:
            prices: List of historical prices
            volumes: Optional list of volumes
            lookback: Number of periods to look back

        Returns:
            Feature array
        """
        if len(prices) < lookback:
            # Pad with zeros if not enough history
            prices = [prices[0]] * (lookback - len(prices)) + prices

        prices_array = np.array(prices[-lookback:])

        features = []

        # 1. Returns (log returns)
        returns = np.diff(np.log(prices_array + 1e-10))
        features.extend(returns[-10:])  # Last 10 returns

        # 2. Moving averages
        sma_5 = np.mean(prices_array[-5:])
        sma_10 = np.mean(prices_array[-10:])
        sma_20 = np.mean(prices_array[-20:]) if len(prices_array) >= 20 else np.mean(prices_array)
        features.extend([sma_5, sma_10, sma_20])

        # 3. Price relative to SMA
        current_price = prices_array[-1]
        features.append((current_price - sma_20) / sma_20)

        # 4. Volatility (std of returns)
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        features.append(volatility)

        # 5. RSI-like feature
        gains = returns.copy()
        losses = returns.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
        avg_loss = abs(np.mean(losses[-14:])) if len(losses) >= 14 else abs(np.mean(losses))
        rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))
        features.append(rsi / 100.0)  # Normalize to 0-1

        # 6. Momentum
        momentum = (prices_array[-1] - prices_array[-10]) / prices_array[-10] if len(prices_array) >= 10 else 0
        features.append(momentum)

        # 7. Trend (linear regression slope)
        x = np.arange(len(prices_array))
        trend = np.polyfit(x, prices_array, 1)[0]
        features.append(trend)

        # 8. Bollinger Band position
        bb_std = np.std(prices_array[-20:]) if len(prices_array) >= 20 else np.std(prices_array)
        bb_upper = sma_20 + 2 * bb_std
        bb_lower = sma_20 - 2 * bb_std
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower + 1e-10)
        features.append(bb_position)

        # Volume features (if available)
        if volumes and len(volumes) >= lookback:
            volumes_array = np.array(volumes[-lookback:])
            avg_volume = np.mean(volumes_array[-10:])
            volume_ratio = volumes_array[-1] / (avg_volume + 1e-10)
            features.append(volume_ratio)
        else:
            features.append(1.0)  # Neutral volume

        return np.array(features, dtype=np.float32)


# Global model server instance
model_server: Optional[ModelServer] = None


def get_model_server() -> ModelServer:
    """Get or create global model server instance"""
    global model_server
    if model_server is None:
        model_server = ModelServer()
    return model_server
