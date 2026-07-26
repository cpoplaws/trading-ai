"""
ML Ensemble Strategy
Combines predictions from LSTM, GRU, and Transformer models
"""
import numpy as np
from typing import Dict, List
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from strategies.base_strategy import BaseStrategy, Signal
from ml.model_server import get_model_server, FeatureExtractor

logger = logging.getLogger(__name__)


class MLEnsembleStrategy(BaseStrategy):
    """
    ML Ensemble Strategy using multiple neural network models

    Combines predictions from:
    - LSTM (60% weight)
    - GRU (20% weight)
    - Transformer (20% weight)
    """

    def __init__(self, symbols: List[str]):
        super().__init__(symbols)
        self.price_history = []
        self.volume_history = []
        self.model_server = get_model_server()
        self.feature_extractor = FeatureExtractor()

        # Model weights (higher weight = more trust)
        self.model_weights = {
            "lstm_model": 0.60,
            "gru_model": 0.20,
            "transformer_model": 0.20
        }

        # Confidence thresholds
        self.buy_threshold = 0.6  # Predict > 60% price increase
        self.sell_threshold = 0.4  # Predict < 40% price increase (bearish)

        logger.info("ML Ensemble Strategy initialized")

    def generate_signal(self, market_data: Dict) -> Signal:
        """
        Generate trading signal using ML ensemble

        Process:
        1. Extract features from price history
        2. Get predictions from LSTM, GRU, Transformer
        3. Combine using weighted voting
        4. Generate BUY/SELL/HOLD signal
        """
        symbol = self.symbols[0]
        if symbol not in market_data:
            return Signal.HOLD

        current_price = market_data[symbol]["price"]
        current_volume = market_data[symbol].get("volume", 0)

        # Update history
        self.price_history.append(current_price)
        self.volume_history.append(current_volume)

        # Keep only recent history
        if len(self.price_history) > 200:
            self.price_history = self.price_history[-200:]
            self.volume_history = self.volume_history[-200:]

        # Need minimum history for features
        if len(self.price_history) < 20:
            return Signal.HOLD

        # Check for stop loss/take profit on existing position
        if self.position > 0:
            if self.check_stop_loss(current_price):
                logger.info(f"ML Ensemble: Stop loss triggered at ${current_price:.2f}")
                self.position = 0
                self.entry_price = 0
                return Signal.SELL
            if self.check_take_profit(current_price):
                logger.info(f"ML Ensemble: Take profit triggered at ${current_price:.2f}")
                self.position = 0
                self.entry_price = 0
                return Signal.SELL

        # Extract features
        try:
            features = self.feature_extractor.extract_features(
                self.price_history,
                self.volume_history
            )
            features = features.reshape(1, -1)  # Reshape for model input
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return Signal.HOLD

        # Get ensemble prediction
        try:
            # Get predictions from each model
            models = list(self.model_weights.keys())
            weights = list(self.model_weights.values())

            prediction = self.model_server.predict_ensemble(
                features,
                models,
                weights
            )

            if prediction is None:
                # Fallback: try individual models
                logger.warning("Ensemble prediction failed, trying individual models")
                predictions = []

                for model_name in models:
                    pred = self.model_server.predict(model_name, features)
                    if pred is not None:
                        predictions.append(pred)

                if predictions:
                    prediction = np.mean(predictions)
                else:
                    logger.warning("All ML models failed - using HOLD signal")
                    return Signal.HOLD

        except Exception as e:
            logger.error(f"Error getting ML prediction: {e}")
            return Signal.HOLD

        # Interpret prediction
        # Prediction is expected to be a probability or normalized score (0-1)
        # > 0.6 = bullish, < 0.4 = bearish, else neutral

        logger.info(f"ML Ensemble prediction: {prediction:.3f}")

        # Generate signal based on prediction
        if self.position == 0:
            # Not in position - look for entry
            if prediction > self.buy_threshold:
                logger.info(f"ML Ensemble: BUY signal (prediction: {prediction:.3f})")
                self.entry_price = current_price
                self.position = 1
                return Signal.BUY

            elif prediction < self.sell_threshold:
                logger.info(f"ML Ensemble: SELL signal (prediction: {prediction:.3f})")
                self.entry_price = current_price
                self.position = -1
                return Signal.SELL

        else:
            # In position - look for exit
            if self.position > 0 and prediction < 0.5:
                # Exit long if prediction turns bearish
                logger.info(f"ML Ensemble: Exit LONG (prediction turned bearish: {prediction:.3f})")
                self.position = 0
                self.entry_price = 0
                return Signal.SELL

            elif self.position < 0 and prediction > 0.5:
                # Exit short if prediction turns bullish
                logger.info(f"ML Ensemble: Exit SHORT (prediction turned bullish: {prediction:.3f})")
                self.position = 0
                self.entry_price = 0
                return Signal.BUY

        return Signal.HOLD


class SimpleMLPredictor:
    """
    Simple ML predictor for when trained models are not available
    Uses basic statistical methods to simulate ML behavior
    """

    @staticmethod
    def predict(features: np.ndarray) -> float:
        """
        Simple prediction based on features

        Returns value between 0-1 representing bullish (>0.5) or bearish (<0.5) outlook
        """
        # Extract key features
        # Assuming features are: [returns..., sma_5, sma_10, sma_20, price_sma_ratio, volatility, rsi, momentum, trend, bb_position, volume_ratio]

        try:
            # Simple heuristic-based prediction
            price_sma_ratio = features[13] if len(features) > 13 else 0
            rsi = features[15] if len(features) > 15 else 0.5
            momentum = features[16] if len(features) > 16 else 0
            bb_position = features[18] if len(features) > 18 else 0.5

            # Combine signals
            score = 0.5  # Start neutral

            # RSI component
            if rsi < 0.3:
                score += 0.15  # Oversold = bullish
            elif rsi > 0.7:
                score -= 0.15  # Overbought = bearish

            # Momentum component
            score += momentum * 0.2

            # Bollinger Band position
            if bb_position < 0.2:
                score += 0.15  # Near lower band = bullish
            elif bb_position > 0.8:
                score -= 0.15  # Near upper band = bearish

            # Price vs SMA
            if price_sma_ratio > 0.02:
                score += 0.1  # Above SMA = bullish
            elif price_sma_ratio < -0.02:
                score -= 0.1  # Below SMA = bearish

            # Clip to 0-1 range
            score = np.clip(score, 0.0, 1.0)

            return score

        except Exception as e:
            logger.error(f"Error in simple ML predictor: {e}")
            return 0.5  # Neutral
