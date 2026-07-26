"""
GRU (Gated Recurrent Unit) Price Predictor
===========================================

GRU is an alternative to LSTM with fewer parameters and often faster training.
Good for time series prediction with less computational cost than LSTM.

Features:
- Bidirectional GRU for better context
- Attention mechanism
- Multi-step ahead forecasting
- Uncertainty estimation
"""

import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")


@dataclass
class GRUConfig:
    """GRU model configuration."""
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    use_attention: bool = True
    sequence_length: int = 60
    forecast_horizon: int = 1  # Number of steps ahead to predict
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10


@dataclass
class PredictionResult:
    """GRU prediction result."""
    predicted_prices: List[float]  # Multi-step predictions
    confidence: float
    attention_weights: Optional[np.ndarray]
    timestamp: datetime


class Attention(nn.Module):
    """
    Attention mechanism for GRU.
    Learns which time steps are most important for prediction.
    """

    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, gru_output):
        """
        Args:
            gru_output: (batch, seq_len, hidden_size)

        Returns:
            context: (batch, hidden_size)
            attention_weights: (batch, seq_len)
        """
        # Calculate attention scores
        scores = self.attention(gru_output)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)

        # Apply attention weights
        context = torch.sum(attention_weights * gru_output, dim=1)  # (batch, hidden_size)

        return context, attention_weights.squeeze(-1)


class GRUPredictor(nn.Module):
    """
    GRU-based price predictor with attention.

    Architecture:
    - Input: (batch, sequence_length, features)
    - Bidirectional GRU layers
    - Optional attention mechanism
    - Fully connected layers
    - Output: (batch, forecast_horizon)
    """

    def __init__(self, input_size: int, config: GRUConfig):
        super(GRUPredictor, self).__init__()

        self.config = config
        self.hidden_size = config.hidden_size * (2 if config.bidirectional else 1)

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True
        )

        # Attention
        if config.use_attention:
            self.attention = Attention(self.hidden_size)
        else:
            self.attention = None

        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, config.forecast_horizon)

        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, sequence_length, features)

        Returns:
            predictions: (batch, forecast_horizon)
            attention_weights: (batch, sequence_length) or None
        """
        # GRU
        gru_output, hidden = self.gru(x)  # gru_output: (batch, seq_len, hidden_size)

        # Attention or use last hidden state
        if self.attention:
            context, attention_weights = self.attention(gru_output)
        else:
            # Use last time step output
            context = gru_output[:, -1, :]
            attention_weights = None

        # Fully connected layers
        x = self.relu(self.fc1(context))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        predictions = self.fc3(x)

        return predictions, attention_weights


class TimeSeriesDataset(Dataset):
    """Dataset for time series data."""

    def __init__(self, data: np.ndarray, sequence_length: int, forecast_horizon: int):
        self.data = data
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.data) - self.sequence_length - self.forecast_horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length:idx + self.sequence_length + self.forecast_horizon, 0]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class GRUTradingPredictor:
    """
    High-level GRU predictor for trading.

    Handles training, prediction, and uncertainty estimation.
    """

    def __init__(self, input_size: int, config: Optional[GRUConfig] = None):
        """
        Initialize GRU predictor.

        Args:
            input_size: Number of input features
            config: GRU configuration
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.config = config or GRUConfig()
        self.input_size = input_size
        self.model = GRUPredictor(input_size, self.config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.criterion = nn.MSELoss()

        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.is_trained = False

        logger.info(f"Initialized GRU predictor with {input_size} features")

    def _normalize(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize data using z-score normalization."""
        if fit:
            self.scaler_mean = np.mean(data, axis=0)
            self.scaler_std = np.std(data, axis=0) + 1e-8

        return (data - self.scaler_mean) / self.scaler_std

    def _denormalize(self, data: np.ndarray, feature_idx: int = 0) -> np.ndarray:
        """Denormalize predictions."""
        return data * self.scaler_std[feature_idx] + self.scaler_mean[feature_idx]

    def train(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train GRU model.

        Args:
            train_data: Training data (n_samples, n_features)
            val_data: Validation data (n_samples, n_features)
            verbose: Print training progress

        Returns:
            Training history
        """
        logger.info(f"Training GRU with {train_data.shape[0]} samples")

        # Normalize data
        train_data_norm = self._normalize(train_data, fit=True)
        if val_data is not None:
            val_data_norm = self._normalize(val_data, fit=False)

        # Create datasets
        train_dataset = TimeSeriesDataset(
            train_data_norm,
            self.config.sequence_length,
            self.config.forecast_horizon
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        if val_data is not None:
            val_dataset = TimeSeriesDataset(
                val_data_norm,
                self.config.sequence_length,
                self.config.forecast_horizon
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )

        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            train_losses = []

            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()

                predictions, _ = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)

                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)

            # Validation
            if val_data is not None:
                self.model.eval()
                val_losses = []

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        predictions, _ = self.model(batch_x)
                        loss = self.criterion(predictions, batch_y)
                        val_losses.append(loss.item())

                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)

                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.config.num_epochs} - "
                        f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                    )

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.config.num_epochs} - "
                        f"Train Loss: {avg_train_loss:.6f}"
                    )

        self.is_trained = True
        logger.info("Training complete")
        return history

    def predict(
        self,
        sequence: np.ndarray,
        return_attention: bool = False
    ) -> PredictionResult:
        """
        Make prediction.

        Args:
            sequence: Input sequence (sequence_length, features)
            return_attention: Return attention weights

        Returns:
            Prediction result
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()

        # Normalize input
        sequence_norm = self._normalize(sequence, fit=False)

        # Convert to tensor
        x = torch.FloatTensor(sequence_norm).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            predictions_norm, attention_weights = self.model(x)

        # Denormalize predictions
        predictions = self._denormalize(predictions_norm.numpy()[0], feature_idx=0)

        # Calculate confidence from attention if available
        if attention_weights is not None:
            # Higher confidence if attention is focused (low entropy)
            attn_entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-10)
            ).item()
            # Normalize entropy to [0, 1] confidence
            max_entropy = np.log(self.config.sequence_length)
            confidence = 1.0 - (attn_entropy / max_entropy)
        else:
            confidence = 0.7  # Default confidence

        return PredictionResult(
            predicted_prices=predictions.tolist(),
            confidence=float(confidence),
            attention_weights=attention_weights.numpy()[0] if return_attention else None,
            timestamp=datetime.utcnow()
        )

    def predict_with_uncertainty(
        self,
        sequence: np.ndarray,
        num_samples: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation using dropout at test time (MC Dropout).

        Args:
            sequence: Input sequence
            num_samples: Number of forward passes

        Returns:
            mean_predictions: Mean prediction
            std_predictions: Standard deviation (uncertainty)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Enable dropout at test time
        self.model.train()

        predictions = []
        for _ in range(num_samples):
            result = self.predict(sequence)
            predictions.append(result.predicted_prices)

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        self.model.eval()

        return mean_pred, std_pred

    def save(self, path: str):
        """Save model to disk."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'input_size': self.input_size
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'GRUTradingPredictor':
        """Load model from disk."""
        checkpoint = torch.load(path)

        predictor = cls(checkpoint['input_size'], checkpoint['config'])
        predictor.model.load_state_dict(checkpoint['model_state'])
        predictor.optimizer.load_state_dict(checkpoint['optimizer_state'])
        predictor.scaler_mean = checkpoint['scaler_mean']
        predictor.scaler_std = checkpoint['scaler_std']
        predictor.is_trained = True

        logger.info(f"Model loaded from {path}")
        return predictor


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("GRU Price Predictor Example")
    print("=" * 60)

    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch not available. Please install: pip install torch")
        exit(1)

    # Generate synthetic price data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5

    # Create synthetic trend + noise
    t = np.linspace(0, 100, n_samples)
    prices = 100 + 10 * np.sin(t / 10) + np.cumsum(np.random.randn(n_samples) * 0.5)

    # Additional features
    returns = np.diff(prices, prepend=prices[0])
    volume = np.random.lognormal(10, 0.5, n_samples)
    volatility = np.array([np.std(returns[max(0, i-20):i+1]) for i in range(n_samples)])
    rsi = 50 + 25 * np.sin(t / 15)

    data = np.column_stack([prices, returns, volume, volatility, rsi])

    # Split data
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    # Create and train model
    config = GRUConfig(
        hidden_size=64,
        num_layers=2,
        use_attention=True,
        sequence_length=30,
        forecast_horizon=3,  # Predict 3 steps ahead
        num_epochs=50
    )

    predictor = GRUTradingPredictor(input_size=n_features, config=config)

    print("\nTraining GRU model...")
    history = predictor.train(train_data, val_data)

    print(f"\nFinal train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")

    # Make predictions
    print("\nMaking predictions...")
    test_sequence = val_data[-config.sequence_length:]
    result = predictor.predict(test_sequence, return_attention=True)

    print(f"\nPrediction Result:")
    print(f"  Predicted prices (next {config.forecast_horizon} steps): {result.predicted_prices}")
    print(f"  Confidence: {result.confidence:.4f}")
    if result.attention_weights is not None:
        print(f"  Top 3 attended time steps: {np.argsort(result.attention_weights)[-3:]}")

    # Prediction with uncertainty
    print("\nPredicting with uncertainty...")
    mean_pred, std_pred = predictor.predict_with_uncertainty(test_sequence, num_samples=20)
    print(f"  Mean prediction: {mean_pred}")
    print(f"  Std deviation (uncertainty): {std_pred}")

    print("\n✅ GRU Predictor Example Complete!")
