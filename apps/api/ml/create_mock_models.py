"""
Create Mock ML Models for Testing
Generates simple models that can be used when trained models are not available
"""
import os
import pickle
import numpy as np
from typing import Any


class MockModel:
    """Simple mock model that returns predictions"""

    def __init__(self, name: str, strategy: str = "random"):
        self.name = name
        self.strategy = strategy

    def predict(self, X: np.ndarray) -> float:
        """
        Generate prediction based on strategy

        Args:
            X: Input features

        Returns:
            Prediction value between 0-1
        """
        if self.strategy == "random":
            # Random prediction
            return np.random.uniform(0.4, 0.6)

        elif self.strategy == "bullish":
            # Slightly bullish bias
            return np.random.uniform(0.5, 0.7)

        elif self.strategy == "bearish":
            # Slightly bearish bias
            return np.random.uniform(0.3, 0.5)

        elif self.strategy == "feature_based":
            # Use features to make simple prediction
            try:
                # Assume last few features are RSI, momentum, etc.
                if len(X.shape) > 1:
                    features = X[0]
                else:
                    features = X

                # Simple heuristic
                if len(features) > 15:
                    rsi = features[15]
                    momentum = features[16] if len(features) > 16 else 0

                    # Combine signals
                    score = 0.5
                    if rsi < 0.3:
                        score += 0.2
                    elif rsi > 0.7:
                        score -= 0.2

                    if momentum > 0:
                        score += 0.1
                    elif momentum < 0:
                        score -= 0.1

                    return np.clip(score, 0.0, 1.0)

                return 0.5  # Neutral if not enough features

            except Exception:
                return 0.5


class MockDQNModel:
    """Mock DQN model that returns action values"""

    def __init__(self, name: str):
        self.name = name

    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict Q-values for each action

        Returns:
            Array of Q-values for [HOLD, BUY, SELL]
        """
        # Simple heuristic based on state
        try:
            if len(state.shape) > 1:
                state_features = state[0]
            else:
                state_features = state

            # Check position state
            in_position = state_features[-6] if len(state_features) > 6 else 0
            pnl = state_features[-4] if len(state_features) > 4 else 0

            # Default Q-values
            q_values = np.array([0.5, 0.3, 0.2])  # [HOLD, BUY, SELL]

            # If in position with profit, favor SELL
            if in_position > 0 and pnl > 0.05:
                q_values = np.array([0.2, 0.1, 0.7])

            # If in position with loss, favor SELL (stop loss)
            elif in_position > 0 and pnl < -0.03:
                q_values = np.array([0.1, 0.1, 0.8])

            # If not in position, favor BUY or HOLD based on features
            elif in_position == 0:
                # Use RSI and momentum
                if len(state_features) > 16:
                    rsi = state_features[15]
                    momentum = state_features[16]

                    if rsi < 0.3 and momentum > 0:
                        q_values = np.array([0.2, 0.7, 0.1])  # Favor BUY
                    elif rsi > 0.7:
                        q_values = np.array([0.3, 0.1, 0.6])  # Favor SELL
                    else:
                        q_values = np.array([0.6, 0.2, 0.2])  # Favor HOLD

            return q_values

        except Exception:
            return np.array([0.5, 0.3, 0.2])  # Default


def create_mock_models(models_dir: str = "../../models"):
    """
    Create mock models for testing

    Args:
        models_dir: Directory to save models
    """
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    models = {
        "lstm_model": MockModel("LSTM", strategy="feature_based"),
        "gru_model": MockModel("GRU", strategy="bullish"),
        "transformer_model": MockModel("Transformer", strategy="feature_based"),
        "dqn_model": MockDQNModel("DQN"),
        "ensemble_model": MockModel("Ensemble", strategy="feature_based"),
    }

    # Save each model
    for model_name, model in models.items():
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Created mock model: {model_path}")

    print(f"\n✅ Created {len(models)} mock models in {models_dir}")
    print("These models can be used for testing until real trained models are available")


if __name__ == "__main__":
    create_mock_models()
