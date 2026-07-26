"""
Advanced LSTM Price Prediction with PyTorch
Full implementation with proper training, validation, and testing.
"""
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# Try importing PyTorch, fallback to warning if not available
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
class TrainingConfig:
    """LSTM training configuration."""
    sequence_length: int = 60  # Look back 60 time steps
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10
    train_split: float = 0.7
    val_split: float = 0.15
    # test_split will be 1 - train - val = 0.15


@dataclass
class PredictionResult:
    """Prediction result with confidence."""
    predicted_price: float
    predicted_direction: str
    confidence: float
    current_price: float
    price_change_pct: float
    timestamp: datetime


class PriceDataset(Dataset):
    """PyTorch dataset for price sequences."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Initialize dataset.

        Args:
            sequences: Input sequences (N, seq_len, features)
            targets: Target values (N,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


class LSTMPricePredictor(nn.Module):
    """
    Advanced LSTM model for price prediction.

    Architecture:
    - Input layer
    - Multiple LSTM layers with dropout
    - Fully connected layers
    - Output layer
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMPricePredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, features)

        Returns:
            Predictions (batch, 1)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Take last time step
        lstm_out = lstm_out[:, -1, :]

        # Fully connected layers
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out


class AdvancedLSTMTrainer:
    """
    LSTM trainer with proper training loop, validation, and checkpointing.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self.config = config or TrainingConfig()
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        logger.info(f"LSTM trainer initialized on device: {self.device}")

    def prepare_sequences(
        self,
        prices: np.ndarray,
        features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for training.

        Args:
            prices: Price array (N,)
            features: Feature array (N, num_features)

        Returns:
            (sequences, targets) tuple
        """
        seq_len = self.config.sequence_length

        sequences = []
        targets = []

        for i in range(len(prices) - seq_len):
            # Input: features for seq_len time steps
            seq = features[i:i+seq_len]
            # Target: next price (normalized)
            target = prices[i+seq_len]

            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def split_data(
        self,
        sequences: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Split data into train/val/test sets.

        Args:
            sequences: Input sequences
            targets: Target values

        Returns:
            (train_loader, val_loader, test_loader) tuple
        """
        n = len(sequences)
        train_size = int(n * self.config.train_split)
        val_size = int(n * self.config.val_split)

        # Split
        train_seq = sequences[:train_size]
        train_targets = targets[:train_size]

        val_seq = sequences[train_size:train_size+val_size]
        val_targets = targets[train_size:train_size+val_size]

        test_seq = sequences[train_size+val_size:]
        test_targets = targets[train_size+val_size:]

        # Create datasets
        train_dataset = PriceDataset(train_seq, train_targets)
        val_dataset = PriceDataset(val_seq, val_targets)
        test_dataset = PriceDataset(test_seq, test_targets)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        logger.info(f"Data split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for sequences, targets in train_loader:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device).unsqueeze(1)

            # Forward pass
            predictions = self.model(sequences)
            loss = self.criterion(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                predictions = self.model(sequences)
                loss = self.criterion(predictions, targets)

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(
        self,
        prices: List[float],
        features: np.ndarray
    ) -> Dict:
        """
        Train the LSTM model.

        Args:
            prices: Price history
            features: Feature matrix (normalized)

        Returns:
            Training metrics
        """
        logger.info("Starting LSTM training...")

        # Prepare data
        prices_array = np.array(prices)
        sequences, targets = self.prepare_sequences(prices_array, features)
        train_loader, val_loader, test_loader = self.split_data(sequences, targets)

        # Initialize model
        input_size = features.shape[1]
        self.model = LSTMPricePredictor(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        # Training loop
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                # Save best model
                self.save_checkpoint('best_model.pth')
            else:
                self.epochs_without_improvement += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f}"
                )

            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Test evaluation
        test_loss = self.validate(test_loader)

        logger.info(f"Training complete! Best val loss: {self.best_val_loss:.6f}, Test loss: {test_loss:.6f}")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'test_loss': test_loss,
            'epochs_trained': len(self.train_losses)
        }

    def predict(
        self,
        prices: List[float],
        features: np.ndarray
    ) -> PredictionResult:
        """
        Make prediction for next price.

        Args:
            prices: Recent price history
            features: Recent features (normalized)

        Returns:
            Prediction result
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()

        # Prepare sequence
        seq_len = self.config.sequence_length
        if len(prices) < seq_len:
            raise ValueError(f"Need at least {seq_len} prices for prediction")

        sequence = features[-seq_len:]
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            prediction = self.model(sequence_tensor)
            predicted_price = prediction.item()

        # Calculate metrics
        current_price = prices[-1]
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100

        if price_change_pct > 1.0:
            direction = "up"
        elif price_change_pct < -1.0:
            direction = "down"
        else:
            direction = "neutral"

        # Confidence based on validation loss
        confidence = max(0.0, min(1.0, 1.0 - (self.best_val_loss * 10)))

        return PredictionResult(
            predicted_price=predicted_price,
            predicted_direction=direction,
            confidence=confidence,
            current_price=current_price,
            price_change_pct=price_change_pct,
            timestamp=datetime.now()
        )

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        if self.model is None:
            return

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }, path)

        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str, input_size: int):
        """Load model checkpoint."""
        checkpoint = torch.load(path)

        self.config = checkpoint['config']
        self.best_val_loss = checkpoint['best_val_loss']

        self.model = LSTMPricePredictor(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Checkpoint loaded from {path}")


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch not installed!")
        print("Install with: pip install torch")
        print("\nThis is the advanced LSTM implementation with:")
        print("- Proper neural network architecture")
        print("- Training/validation/test splits")
        print("- Early stopping")
        print("- Model checkpointing")
        print("- GPU support")
        exit(1)

    print("🤖 Advanced PyTorch LSTM Demo")
    print("=" * 60)

    # Generate sample data
    print("\n1. Generating sample data...")
    np.random.seed(42)
    n_samples = 1000

    # Generate price data with trend + noise
    prices = [2000.0]
    for i in range(n_samples - 1):
        trend = 0.0001 * i  # Slight uptrend
        noise = np.random.normal(0, 20)
        prices.append(prices[-1] + trend + noise)

    # Generate features (price + returns + volatility)
    features = []
    for i in range(len(prices)):
        if i < 10:
            features.append([prices[i], 0.0, 0.0])
        else:
            returns = (prices[i] - prices[i-1]) / prices[i-1]
            volatility = np.std(prices[i-10:i])
            features.append([prices[i], returns, volatility])

    features = np.array(features)

    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    print(f"Generated {len(prices)} prices, {features.shape[1]} features")

    # Train model
    print("\n2. Training LSTM model...")
    config = TrainingConfig(
        sequence_length=60,
        hidden_size=64,
        num_layers=2,
        num_epochs=50,
        batch_size=32
    )

    trainer = AdvancedLSTMTrainer(config)
    metrics = trainer.train(prices, features_normalized)

    print(f"\n3. Training Results:")
    print(f"   Epochs trained: {metrics['epochs_trained']}")
    print(f"   Best val loss: {metrics['best_val_loss']:.6f}")
    print(f"   Test loss: {metrics['test_loss']:.6f}")

    # Make prediction
    print("\n4. Making prediction...")
    prediction = trainer.predict(prices, features_normalized)

    print(f"\n   Current Price: ${prediction.current_price:.2f}")
    print(f"   Predicted Price: ${prediction.predicted_price:.2f}")
    print(f"   Direction: {prediction.predicted_direction}")
    print(f"   Change: {prediction.price_change_pct:+.2f}%")
    print(f"   Confidence: {prediction.confidence*100:.1f}%")

    print("\n✅ Advanced LSTM demo complete!")
    print("\nThis is a production-ready LSTM with:")
    print("- Proper PyTorch implementation")
    print("- Training/validation/test splits")
    print("- Early stopping (patience=10)")
    print("- Model checkpointing")
    print("- GPU support (if available)")
    print("- Normalized features")
    print("- MSE loss function")
