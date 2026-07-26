"""
Variational AutoEncoder (VAE) for Anomaly Detection
====================================================

Uses VAE to detect anomalous market conditions and regime changes.

Applications:
- Detect flash crashes, black swan events
- Identify market regime changes (bull -> bear)
- Detect unusual trading patterns
- Risk management (avoid trading in anomalous conditions)

How it works:
1. VAE learns normal market patterns during training
2. Reconstruction error indicates how "normal" new data is
3. High reconstruction error = anomaly = potential risk
"""

import logging
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")


class AnomalyLevel(str, Enum):
    """Anomaly severity levels."""
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class VAEConfig:
    """VAE configuration."""
    # Architecture
    encoder_hidden_dims: list = None  # [128, 64]
    latent_dim: int = 32
    decoder_hidden_dims: list = None  # [64, 128]

    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    beta: float = 1.0  # Weight for KL divergence (β-VAE)

    # Anomaly detection
    anomaly_threshold_percentile: float = 95.0  # 95th percentile of reconstruction error

    def __post_init__(self):
        if self.encoder_hidden_dims is None:
            self.encoder_hidden_dims = [128, 64]
        if self.decoder_hidden_dims is None:
            self.decoder_hidden_dims = [64, 128]


@dataclass
class AnomalyResult:
    """Anomaly detection result."""
    is_anomaly: bool
    anomaly_level: AnomalyLevel
    reconstruction_error: float
    anomaly_score: float  # 0-1, higher = more anomalous
    latent_representation: np.ndarray
    timestamp: datetime


class Encoder(nn.Module):
    """
    VAE Encoder: Maps input to latent distribution parameters.

    Output: mean and log_variance of latent distribution
    """

    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super(Encoder, self).__init__()

        # Build encoder layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Latent distribution parameters
        self.fc_mean = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        """
        Encode input to latent distribution.

        Returns:
            mean: Latent mean
            log_var: Latent log variance
        """
        h = self.encoder(x)
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mean, log_var


class Decoder(nn.Module):
    """
    VAE Decoder: Maps latent representation back to input space.
    """

    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int):
        super(Decoder, self).__init__()

        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        """
        Decode latent representation to reconstruction.

        Args:
            z: Latent representation

        Returns:
            Reconstructed input
        """
        return self.decoder(z)


class VAE(nn.Module):
    """
    Variational AutoEncoder.

    Learns compressed latent representation of normal market patterns.
    Anomalies have high reconstruction error.
    """

    def __init__(self, input_dim: int, config: VAEConfig):
        super(VAE, self).__init__()

        self.config = config
        self.input_dim = input_dim

        self.encoder = Encoder(
            input_dim,
            config.encoder_hidden_dims,
            config.latent_dim
        )

        self.decoder = Decoder(
            config.latent_dim,
            config.decoder_hidden_dims,
            input_dim
        )

    def reparameterize(self, mean, log_var):
        """
        Reparameterization trick: z = mean + std * epsilon
        where epsilon ~ N(0, 1)

        This allows backpropagation through the stochastic sampling.
        """
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def forward(self, x):
        """
        Forward pass through VAE.

        Returns:
            reconstruction: Reconstructed input
            mean: Latent mean
            log_var: Latent log variance
        """
        # Encode
        mean, log_var = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mean, log_var)

        # Decode
        reconstruction = self.decoder(z)

        return reconstruction, mean, log_var

    def encode(self, x):
        """Encode input to latent representation (deterministic)."""
        mean, _ = self.encoder(x)
        return mean


def vae_loss(x, reconstruction, mean, log_var, beta: float = 1.0):
    """
    VAE loss = Reconstruction Loss + β * KL Divergence

    Args:
        x: Original input
        reconstruction: Reconstructed input
        mean: Latent mean
        log_var: Latent log variance
        beta: Weight for KL divergence (β-VAE)

    Returns:
        Total loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstruction, x, reduction='mean')

    # KL divergence
    # KL(N(mean, var) || N(0, 1))
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    kl_loss = kl_loss / x.size(0)  # Average over batch

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


class VAEAnomalyDetector:
    """
    High-level VAE anomaly detector for trading.

    Detects unusual market conditions for risk management.
    """

    def __init__(self, input_dim: int, config: Optional[VAEConfig] = None):
        """
        Initialize VAE anomaly detector.

        Args:
            input_dim: Number of input features
            config: VAE configuration
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.config = config or VAEConfig()
        self.input_dim = input_dim

        self.model = VAE(input_dim, self.config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.anomaly_threshold: Optional[float] = None
        self.is_trained = False

        logger.info(f"Initialized VAE anomaly detector with {input_dim} features")

    def _normalize(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize data using z-score normalization."""
        if fit:
            self.scaler_mean = np.mean(data, axis=0)
            self.scaler_std = np.std(data, axis=0) + 1e-8

        return (data - self.scaler_mean) / self.scaler_std

    def train(
        self,
        normal_data: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train VAE on normal market data.

        IMPORTANT: Training data should only contain normal, non-anomalous data.

        Args:
            normal_data: Normal market data (n_samples, n_features)
            verbose: Print training progress

        Returns:
            Training history
        """
        logger.info(f"Training VAE on {normal_data.shape[0]} normal samples")

        # Normalize data
        data_norm = self._normalize(normal_data, fit=True)

        # Create dataset
        dataset = torch.FloatTensor(data_norm)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        # Training loop
        history = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_losses = {'total': [], 'recon': [], 'kl': []}

            for batch in dataloader:
                self.optimizer.zero_grad()

                # Forward pass
                reconstruction, mean, log_var = self.model(batch)

                # Calculate loss
                total_loss, recon_loss, kl_loss = vae_loss(
                    batch, reconstruction, mean, log_var, self.config.beta
                )

                # Backward pass
                total_loss.backward()
                self.optimizer.step()

                epoch_losses['total'].append(total_loss.item())
                epoch_losses['recon'].append(recon_loss.item())
                epoch_losses['kl'].append(kl_loss.item())

            # Record history
            history['total_loss'].append(np.mean(epoch_losses['total']))
            history['recon_loss'].append(np.mean(epoch_losses['recon']))
            history['kl_loss'].append(np.mean(epoch_losses['kl']))

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs} - "
                    f"Loss: {history['total_loss'][-1]:.6f} "
                    f"(Recon: {history['recon_loss'][-1]:.6f}, "
                    f"KL: {history['kl_loss'][-1]:.6f})"
                )

        # Calculate anomaly threshold from training data
        self.model.eval()
        with torch.no_grad():
            reconstruction_errors = []
            for batch in dataloader:
                recon, _, _ = self.model(batch)
                errors = torch.mean((batch - recon) ** 2, dim=1)
                reconstruction_errors.extend(errors.numpy())

        # Set threshold at specified percentile
        self.anomaly_threshold = np.percentile(
            reconstruction_errors,
            self.config.anomaly_threshold_percentile
        )

        logger.info(f"Anomaly threshold set to: {self.anomaly_threshold:.6f}")

        self.is_trained = True
        return history

    def detect_anomaly(self, data: np.ndarray) -> AnomalyResult:
        """
        Detect if data is anomalous.

        Args:
            data: Input features (n_features,)

        Returns:
            Anomaly detection result
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()

        # Normalize
        data_norm = self._normalize(data.reshape(1, -1), fit=False)

        # Convert to tensor
        x = torch.FloatTensor(data_norm)

        with torch.no_grad():
            # Reconstruct
            reconstruction, mean, log_var = self.model(x)

            # Calculate reconstruction error
            recon_error = torch.mean((x - reconstruction) ** 2).item()

            # Get latent representation
            latent = mean.numpy()[0]

        # Determine if anomaly
        is_anomaly = recon_error > self.anomaly_threshold

        # Calculate normalized anomaly score (0-1)
        # Score based on how much reconstruction error exceeds threshold
        if recon_error <= self.anomaly_threshold:
            anomaly_score = recon_error / self.anomaly_threshold
        else:
            # Exponential scaling for values above threshold
            anomaly_score = 1.0 - np.exp(-(recon_error - self.anomaly_threshold) / self.anomaly_threshold)

        # Determine anomaly level
        if recon_error <= self.anomaly_threshold:
            level = AnomalyLevel.NORMAL
        elif recon_error <= self.anomaly_threshold * 1.5:
            level = AnomalyLevel.MILD
        elif recon_error <= self.anomaly_threshold * 2.0:
            level = AnomalyLevel.MODERATE
        elif recon_error <= self.anomaly_threshold * 3.0:
            level = AnomalyLevel.SEVERE
        else:
            level = AnomalyLevel.CRITICAL

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_level=level,
            reconstruction_error=float(recon_error),
            anomaly_score=float(anomaly_score),
            latent_representation=latent,
            timestamp=datetime.utcnow()
        )

    def detect_batch(self, data: np.ndarray) -> list:
        """
        Detect anomalies in batch of samples.

        Args:
            data: Input features (n_samples, n_features)

        Returns:
            List of anomaly results
        """
        results = []
        for sample in data:
            result = self.detect_anomaly(sample)
            results.append(result)
        return results

    def save(self, path: str):
        """Save model to disk."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'anomaly_threshold': self.anomaly_threshold,
            'input_dim': self.input_dim
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'VAEAnomalyDetector':
        """Load model from disk."""
        checkpoint = torch.load(path)

        detector = cls(checkpoint['input_dim'], checkpoint['config'])
        detector.model.load_state_dict(checkpoint['model_state'])
        detector.optimizer.load_state_dict(checkpoint['optimizer_state'])
        detector.scaler_mean = checkpoint['scaler_mean']
        detector.scaler_std = checkpoint['scaler_std']
        detector.anomaly_threshold = checkpoint['anomaly_threshold']
        detector.is_trained = True

        logger.info(f"Model loaded from {path}")
        return detector


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("VAE Anomaly Detector Example")
    print("=" * 60)

    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch not available. Please install: pip install torch")
        exit(1)

    # Generate synthetic market data
    np.random.seed(42)
    n_normal_samples = 1000
    n_features = 10

    # Normal market data (no extreme movements)
    normal_data = np.random.randn(n_normal_samples, n_features) * 0.02  # 2% volatility

    # Create and train VAE
    config = VAEConfig(
        encoder_hidden_dims=[64, 32],
        latent_dim=8,
        decoder_hidden_dims=[32, 64],
        num_epochs=50,
        beta=1.0
    )

    detector = VAEAnomalyDetector(input_dim=n_features, config=config)

    print("\nTraining VAE on normal market data...")
    history = detector.train(normal_data)

    print(f"\nFinal losses:")
    print(f"  Total: {history['total_loss'][-1]:.6f}")
    print(f"  Reconstruction: {history['recon_loss'][-1]:.6f}")
    print(f"  KL: {history['kl_loss'][-1]:.6f}")
    print(f"  Anomaly threshold: {detector.anomaly_threshold:.6f}")

    # Test on normal data
    print("\nTesting on normal data...")
    normal_sample = normal_data[-1]
    result = detector.detect_anomaly(normal_sample)
    print(f"  Is anomaly: {result.is_anomaly}")
    print(f"  Level: {result.anomaly_level}")
    print(f"  Recon error: {result.reconstruction_error:.6f}")
    print(f"  Anomaly score: {result.anomaly_score:.4f}")

    # Test on anomalous data (flash crash simulation)
    print("\nTesting on anomalous data (simulated flash crash)...")
    anomaly_sample = np.random.randn(n_features) * 0.15  # 15% volatility - very unusual
    result = detector.detect_anomaly(anomaly_sample)
    print(f"  Is anomaly: {result.is_anomaly}")
    print(f"  Level: {result.anomaly_level}")
    print(f"  Recon error: {result.reconstruction_error:.6f}")
    print(f"  Anomaly score: {result.anomaly_score:.4f}")

    # Test batch detection
    print("\nTesting batch detection...")
    test_batch = np.vstack([
        normal_data[-5:],  # 5 normal samples
        np.random.randn(5, n_features) * 0.12  # 5 anomalous samples
    ])

    results = detector.detect_batch(test_batch)
    anomaly_count = sum(1 for r in results if r.is_anomaly)
    print(f"  Detected {anomaly_count}/{len(results)} anomalies")

    print("\n✅ VAE Anomaly Detector Example Complete!")
