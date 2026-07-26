"""
Transformer-Based Price Prediction
State-of-the-art architecture for time series forecasting.
"""
import logging
from typing import List, Optional, Tuple
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
    import math
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")


@dataclass
class TransformerConfig:
    """Transformer configuration."""
    d_model: int = 128  # Model dimension
    nhead: int = 8  # Number of attention heads
    num_layers: int = 4  # Number of transformer layers
    dim_feedforward: int = 512  # Feedforward dimension
    dropout: float = 0.1
    sequence_length: int = 60
    learning_rate: float = 0.0001
    batch_size: int = 32
    num_epochs: int = 100


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    Adds position information to sequence embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerPricePredictor(nn.Module):
    """
    Transformer model for price prediction.

    Architecture:
    - Input embedding
    - Positional encoding
    - Transformer encoder layers
    - Output projection
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super(TransformerPricePredictor, self).__init__()

        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.fc_out = nn.Linear(d_model, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_size)

        Returns:
            Predictions (batch, 1)
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply dropout
        x = self.dropout(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Take last time step
        x = x[:, -1, :]  # (batch, d_model)

        # Output projection
        x = self.fc_out(x)  # (batch, 1)

        return x


class TransformerTrainer:
    """
    Transformer trainer for price prediction.
    """

    def __init__(self, config: Optional[TransformerConfig] = None):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.config = config or TransformerConfig()
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

        logger.info(f"Transformer trainer initialized on device: {self.device}")

    def train(self, prices: List[float], features: np.ndarray):
        """Train transformer model."""
        logger.info("Training Transformer model...")

        # Prepare data (same as LSTM)
        from src.ml.advanced_lstm import PriceDataset

        seq_len = self.config.sequence_length
        sequences = []
        targets = []

        for i in range(len(prices) - seq_len):
            seq = features[i:i+seq_len]
            target = prices[i+seq_len]
            sequences.append(seq)
            targets.append(target)

        sequences = np.array(sequences)
        targets = np.array(targets)

        # Split data
        n = len(sequences)
        train_size = int(n * 0.7)
        val_size = int(n * 0.15)

        train_seq = sequences[:train_size]
        train_targets = targets[:train_size]
        val_seq = sequences[train_size:train_size+val_size]
        val_targets = targets[train_size:train_size+val_size]

        train_dataset = PriceDataset(train_seq, train_targets)
        val_dataset = PriceDataset(val_seq, val_targets)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        # Initialize model
        input_size = features.shape[1]
        self.model = TransformerPricePredictor(
            input_size=input_size,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_layers=self.config.num_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Training loop
        for epoch in range(self.config.num_epochs):
            # Train
            self.model.train()
            train_loss = 0
            for sequences_batch, targets_batch in train_loader:
                sequences_batch = sequences_batch.to(self.device)
                targets_batch = targets_batch.to(self.device).unsqueeze(1)

                predictions = self.model(sequences_batch)
                loss = self.criterion(predictions, targets_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for sequences_batch, targets_batch in val_loader:
                    sequences_batch = sequences_batch.to(self.device)
                    targets_batch = targets_batch.to(self.device).unsqueeze(1)

                    predictions = self.model(sequences_batch)
                    loss = self.criterion(predictions, targets_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} | "
                           f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        logger.info(f"Training complete! Best val loss: {self.best_val_loss:.6f}")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }

    def predict(self, prices: List[float], features: np.ndarray) -> dict:
        """Make prediction."""
        if self.model is None:
            raise ValueError("Model not trained")

        self.model.eval()

        seq_len = self.config.sequence_length
        sequence = features[-seq_len:]
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(sequence_tensor)
            predicted_price = prediction.item()

        current_price = prices[-1]
        price_change_pct = ((predicted_price - current_price) / current_price) * 100

        return {
            'predicted_price': predicted_price,
            'current_price': current_price,
            'price_change_pct': price_change_pct,
            'direction': 'up' if price_change_pct > 0 else 'down',
            'confidence': max(0.0, min(1.0, 1.0 - self.best_val_loss * 10))
        }


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch not installed!")
        print("Install with: pip install torch")
        exit(1)

    print("🤖 Transformer Price Predictor Demo")
    print("=" * 60)

    # Generate sample data
    print("\n1. Generating sample data...")
    np.random.seed(42)
    prices = [2000.0]
    for i in range(999):
        prices.append(prices[-1] + np.random.normal(0, 20))

    # Features
    features = []
    for i in range(len(prices)):
        if i < 10:
            features.append([prices[i], 0.0, 0.0])
        else:
            returns = (prices[i] - prices[i-1]) / prices[i-1]
            volatility = np.std(prices[i-10:i])
            features.append([prices[i], returns, volatility])

    features = np.array(features)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    print(f"Generated {len(prices)} prices")

    # Train
    print("\n2. Training Transformer model...")
    config = TransformerConfig(
        d_model=64,
        nhead=4,
        num_layers=2,
        num_epochs=30
    )

    trainer = TransformerTrainer(config)
    metrics = trainer.train(prices, features_normalized)

    print(f"\n3. Training Results:")
    print(f"   Best val loss: {metrics['best_val_loss']:.6f}")

    # Predict
    print("\n4. Making prediction...")
    prediction = trainer.predict(prices, features_normalized)

    print(f"\n   Current: ${prediction['current_price']:.2f}")
    print(f"   Predicted: ${prediction['predicted_price']:.2f}")
    print(f"   Direction: {prediction['direction']}")
    print(f"   Change: {prediction['price_change_pct']:+.2f}%")
    print(f"   Confidence: {prediction['confidence']*100:.1f}%")

    print("\n✅ Transformer demo complete!")
    print("\nFeatures:")
    print("- Multi-head self-attention")
    print("- Positional encoding")
    print("- 4 transformer encoder layers")
    print("- State-of-the-art architecture")
