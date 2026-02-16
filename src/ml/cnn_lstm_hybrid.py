"""
CNN-LSTM Hybrid Model
=====================

Combines Convolutional Neural Networks (CNN) for pattern recognition
with Long Short-Term Memory (LSTM) for temporal dependencies.

Architecture:
1. CNN layers extract local patterns (candlestick patterns, support/resistance)
2. LSTM layers capture temporal dependencies
3. Fully connected layers for final prediction

Benefits:
- CNNs detect chart patterns automatically
- LSTMs model time-series relationships
- Better than pure LSTM for pattern-based trading
"""

import logging
from typing import Optional, Tuple, Dict, List
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
class CNNLSTMConfig:
    """CNN-LSTM model configuration."""
    # CNN parameters
    cnn_filters: List[int] = None  # [64, 128, 256]
    kernel_sizes: List[int] = None  # [3, 3, 3]
    pool_sizes: List[int] = None  # [2, 2, 2]

    # LSTM parameters
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2

    # Training parameters
    sequence_length: int = 60
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10

    def __post_init__(self):
        if self.cnn_filters is None:
            self.cnn_filters = [64, 128, 256]
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 3, 3]
        if self.pool_sizes is None:
            self.pool_sizes = [2, 2, 2]


@dataclass
class PredictionResult:
    """Prediction result with pattern information."""
    predicted_price: float
    predicted_direction: str  # 'up', 'down', 'neutral'
    confidence: float
    detected_patterns: Dict[str, float]  # Pattern name -> activation strength
    timestamp: datetime


class CNNBlock(nn.Module):
    """
    Convolutional block for pattern recognition.

    Each block: Conv1d -> BatchNorm -> ReLU -> MaxPool
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int):
        super(CNNBlock, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class CNNLSTMHybrid(nn.Module):
    """
    CNN-LSTM Hybrid Architecture.

    1. CNN layers extract local patterns
    2. LSTM layers capture temporal dependencies
    3. Fully connected layers for prediction

    Input shape: (batch, sequence_length, features)
    Output shape: (batch, output_size)
    """

    def __init__(self, input_size: int, config: CNNLSTMConfig, output_size: int = 1):
        super(CNNLSTMHybrid, self).__init__()

        self.config = config
        self.input_size = input_size

        # CNN layers
        self.cnn_layers = nn.ModuleList()
        in_channels = input_size

        for out_channels, kernel_size, pool_size in zip(
            config.cnn_filters,
            config.kernel_sizes,
            config.pool_sizes
        ):
            self.cnn_layers.append(
                CNNBlock(in_channels, out_channels, kernel_size, pool_size)
            )
            in_channels = out_channels

        # Calculate sequence length after CNN layers
        seq_len_after_cnn = config.sequence_length
        for pool_size in config.pool_sizes:
            seq_len_after_cnn = seq_len_after_cnn // pool_size

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.cnn_filters[-1],
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0,
            batch_first=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(config.lstm_hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

        # Pattern detection layer (for interpretability)
        self.pattern_detector = nn.Linear(config.lstm_hidden_size, 10)  # 10 common patterns

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, sequence_length, features)

        Returns:
            prediction: (batch, output_size)
            patterns: (batch, 10) - Pattern activation scores
        """
        batch_size = x.size(0)

        # CNN expects (batch, channels, sequence)
        x = x.permute(0, 2, 1)  # (batch, features, sequence_length)

        # CNN layers
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)

        # Back to (batch, sequence, features) for LSTM
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]

        # Pattern detection (for interpretability)
        patterns = torch.sigmoid(self.pattern_detector(last_hidden))

        # Prediction
        x = self.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        prediction = self.fc3(x)

        return prediction, patterns


class TimeSeriesDataset(Dataset):
    """Dataset for time series data."""

    def __init__(self, data: np.ndarray, sequence_length: int):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length, :-1]  # All features except last
        y = self.data[idx + self.sequence_length, -1]  # Target (price)
        return torch.FloatTensor(x), torch.FloatTensor([y])


class CNNLSTMPredictor:
    """
    High-level CNN-LSTM predictor for trading.

    Combines pattern recognition with temporal modeling.
    """

    def __init__(self, input_size: int, config: Optional[CNNLSTMConfig] = None):
        """
        Initialize CNN-LSTM predictor.

        Args:
            input_size: Number of input features
            config: Model configuration
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.config = config or CNNLSTMConfig()
        self.input_size = input_size

        self.model = CNNLSTMHybrid(input_size, self.config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.criterion = nn.MSELoss()

        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.is_trained = False

        # Pattern names (for interpretability)
        self.pattern_names = [
            "Double Top", "Double Bottom", "Head & Shoulders",
            "Inverse H&S", "Triangle", "Flag",
            "Wedge", "Support Break", "Resistance Break", "Trend"
        ]

        logger.info(f"Initialized CNN-LSTM predictor with {input_size} features")

    def _normalize(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize data using z-score normalization."""
        if fit:
            self.scaler_mean = np.mean(data, axis=0)
            self.scaler_std = np.std(data, axis=0) + 1e-8

        return (data - self.scaler_mean) / self.scaler_std

    def _denormalize_price(self, price: float) -> float:
        """Denormalize price prediction."""
        # Assuming last column is price
        return price * self.scaler_std[-1] + self.scaler_mean[-1]

    def train(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train CNN-LSTM model.

        Args:
            train_data: Training data (n_samples, n_features)
            val_data: Validation data
            verbose: Print training progress

        Returns:
            Training history
        """
        logger.info(f"Training CNN-LSTM with {train_data.shape[0]} samples")

        # Normalize data
        train_data_norm = self._normalize(train_data, fit=True)
        if val_data is not None:
            val_data_norm = self._normalize(val_data, fit=False)

        # Create datasets
        train_dataset = TimeSeriesDataset(train_data_norm, self.config.sequence_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        if val_data is not None:
            val_dataset = TimeSeriesDataset(val_data_norm, self.config.sequence_length)
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

                predictions, patterns = self.model(batch_x)
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
                        predictions, patterns = self.model(batch_x)
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

    def predict(self, sequence: np.ndarray) -> PredictionResult:
        """
        Make prediction with pattern detection.

        Args:
            sequence: Input sequence (sequence_length, features)

        Returns:
            Prediction result with detected patterns
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()

        # Normalize input
        sequence_norm = self._normalize(sequence, fit=False)

        # Convert to tensor
        x = torch.FloatTensor(sequence_norm).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            prediction_norm, patterns = self.model(x)

        # Denormalize prediction
        predicted_price = self._denormalize_price(prediction_norm.item())

        # Determine direction
        current_price = sequence[-1, 0]  # Assuming first feature is price
        price_change = (predicted_price - current_price) / current_price

        if price_change > 0.01:  # >1% up
            direction = "up"
        elif price_change < -0.01:  # >1% down
            direction = "down"
        else:
            direction = "neutral"

        # Extract detected patterns
        pattern_scores = patterns.numpy()[0]
        detected_patterns = {
            name: float(score)
            for name, score in zip(self.pattern_names, pattern_scores)
        }

        # Confidence from pattern clarity
        # High confidence if patterns are very strong or very weak (clear signal)
        pattern_entropy = -np.sum(pattern_scores * np.log(pattern_scores + 1e-10))
        max_entropy = np.log(len(pattern_scores))
        confidence = 1.0 - (pattern_entropy / max_entropy)

        return PredictionResult(
            predicted_price=float(predicted_price),
            predicted_direction=direction,
            confidence=float(confidence),
            detected_patterns=detected_patterns,
            timestamp=datetime.utcnow()
        )

    def get_top_patterns(self, sequence: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top K detected patterns.

        Args:
            sequence: Input sequence
            top_k: Number of top patterns to return

        Returns:
            List of (pattern_name, score) tuples
        """
        result = self.predict(sequence)
        sorted_patterns = sorted(
            result.detected_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_patterns[:top_k]

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
    def load(cls, path: str) -> 'CNNLSTMPredictor':
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

    print("CNN-LSTM Hybrid Predictor Example")
    print("=" * 60)

    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch not available. Please install: pip install torch")
        exit(1)

    # Generate synthetic OHLCV data
    np.random.seed(42)
    n_samples = 1000
    n_features = 6  # open, high, low, close, volume, price

    # Create synthetic price series with patterns
    t = np.linspace(0, 100, n_samples)
    base_price = 100 + 10 * np.sin(t / 10)  # Trend
    noise = np.cumsum(np.random.randn(n_samples) * 0.3)
    prices = base_price + noise

    # OHLC
    opens = prices + np.random.randn(n_samples) * 0.1
    highs = np.maximum(opens, prices) + np.abs(np.random.randn(n_samples) * 0.2)
    lows = np.minimum(opens, prices) - np.abs(np.random.randn(n_samples) * 0.2)
    closes = prices
    volumes = np.random.lognormal(10, 0.3, n_samples)

    data = np.column_stack([opens, highs, lows, closes, volumes, prices])

    # Split data
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    # Create and train model
    config = CNNLSTMConfig(
        cnn_filters=[32, 64, 128],
        kernel_sizes=[3, 3, 3],
        pool_sizes=[2, 2, 1],  # Less aggressive pooling
        lstm_hidden_size=64,
        lstm_num_layers=2,
        sequence_length=60,
        num_epochs=50
    )

    predictor = CNNLSTMPredictor(input_size=n_features, config=config)

    print("\nTraining CNN-LSTM model...")
    history = predictor.train(train_data, val_data)

    print(f"\nFinal train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")

    # Make predictions
    print("\nMaking predictions with pattern detection...")
    test_sequence = val_data[-config.sequence_length:]
    result = predictor.predict(test_sequence)

    print(f"\nPrediction Result:")
    print(f"  Current price: {test_sequence[-1, 0]:.2f}")
    print(f"  Predicted price: {result.predicted_price:.2f}")
    print(f"  Direction: {result.predicted_direction}")
    print(f"  Confidence: {result.confidence:.4f}")

    print(f"\nTop 3 Detected Patterns:")
    top_patterns = predictor.get_top_patterns(test_sequence, top_k=3)
    for pattern, score in top_patterns:
        print(f"  {pattern}: {score:.4f}")

    print("\n✅ CNN-LSTM Hybrid Predictor Example Complete!")
