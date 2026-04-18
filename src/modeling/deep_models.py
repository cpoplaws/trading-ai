"""
Deep learning time series models for trading.

Implements lightweight TimesNet and Autoformer inspired architectures in PyTorch
with a reusable training loop that supports early stopping and checkpointing.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """Simple sliding-window dataset for supervised forecasting/classification."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.targets[idx]


def _compatible_heads(hidden_size: int, requested_heads: int) -> int:
    """Ensure the transformer head count divides the hidden size."""
    if hidden_size % requested_heads == 0:
        return requested_heads
    for heads in range(requested_heads, 0, -1):
        if hidden_size % heads == 0:
            return heads
    return 1


class TimesNetModel(nn.Module):
    """
    Lightweight TimesNet-inspired model.

    Uses temporal convolutions followed by a Transformer encoder and a simple head
    to produce binary trading signals.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        heads = _compatible_heads(hidden_size, max(1, n_heads))
        self.temporal_proj = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        x = self.temporal_proj(x)  # (batch, hidden, seq_len)
        x = x.permute(2, 0, 1)  # (seq_len, batch, hidden)
        encoded = self.encoder(x)  # (seq_len, batch, hidden)
        last_token = encoded[-1]  # (batch, hidden)
        logits = self.head(last_token)
        return logits.squeeze(-1)


class AutoformerModel(nn.Module):
    """
    Simplified Autoformer-style encoder-decoder model for binary prediction.

    This is intentionally lightweight for quick experimentation and CPU training.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        heads = _compatible_heads(hidden_size, max(1, n_heads))
        self.input_projection = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = self.input_projection(x)  # (batch, seq_len, hidden)
        x = x.permute(1, 0, 2)  # (seq_len, batch, hidden)
        memory = self.encoder(x)  # (seq_len, batch, hidden)
        tgt = x[-1:].clone()  # use last observed step as start token
        decoded = self.decoder(tgt, memory)  # (1, batch, hidden)
        logits = self.head(decoded.squeeze(0))
        return logits.squeeze(-1)


class DeepModelTrainer:
    """
    Helper to train deep time-series models with early stopping and checkpointing.
    """

    def __init__(
        self,
        sequence_length: int = 32,
        feature_columns: Optional[List[str]] = None,
        model_type: str = "timesnet",
        device: Optional[torch.device] = None,
    ):
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns
        self.model_type = model_type.lower()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[nn.Module] = None
        self.is_fitted = False

    def _prepare_sequences(
        self, df: pd.DataFrame, target_col: str, fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        data = df.copy()
        data.columns = data.columns.str.lower()
        target_col = target_col.lower()

        if target_col not in data.columns:
            if "close" not in data.columns:
                raise ValueError("Data must include a target column or 'close' price.")
            data[target_col] = (data["close"].shift(-1) > data["close"]).astype(int)
            data = data.dropna(subset=[target_col])

        if self.feature_columns:
            features = [f.lower() for f in self.feature_columns if f.lower() in data.columns]
            if not fit_scaler and len(features) < len(self.feature_columns):
                missing = set(self.feature_columns) - set(data.columns)
                raise ValueError(f"Missing required features for inference: {missing}")
        else:
            exclude = {target_col}
            features = [
                col
                for col in data.columns
                if col not in exclude and pd.api.types.is_numeric_dtype(data[col])
            ]

        if not features:
            raise ValueError("No usable feature columns found for deep model training.")

        if fit_scaler or self.scaler is None:
            self.scaler = StandardScaler()
            feature_values = self.scaler.fit_transform(data[features])
        else:
            feature_values = self.scaler.transform(data[features])

        self.feature_columns = features
        targets = data[target_col].values

        sequences, labels = [], []
        for i in range(self.sequence_length, len(feature_values)):
            sequences.append(feature_values[i - self.sequence_length : i])
            labels.append(targets[i])

        sequences_np = np.asarray(sequences, dtype=np.float32)
        labels_np = np.asarray(labels, dtype=np.float32)
        return sequences_np, labels_np, features

    def _build_model(self, input_size: int) -> nn.Module:
        if self.model_type == "timesnet":
            return TimesNetModel(input_size=input_size)
        if self.model_type == "autoformer":
            return AutoformerModel(input_size=input_size)
        raise ValueError(f"Unsupported model type: {self.model_type}")

    @torch.no_grad()
    def _evaluate(
        self, data_loader: DataLoader, criterion: nn.Module, model: nn.Module
    ) -> float:
        model.eval()
        total_loss = 0.0
        total_samples = 0
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)
        return total_loss / max(1, total_samples)

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        checkpoint_interval: int = 5,
        patience: int = 5,
        save_dir: str = "./models",
        model_name: str = "deep_model",
        validation_size: float = 0.2,
    ) -> Dict[str, object]:
        """
        Train the selected deep model with early stopping and checkpointing.
        """
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        sequences, labels, features = self._prepare_sequences(df, target_col)
        if len(sequences) < 8:
            raise ValueError("Insufficient sequences for training deep model.")

        X_train, X_val, y_train, y_val = train_test_split(
            sequences,
            labels,
            test_size=validation_size,
            random_state=42,
            stratify=labels if len(np.unique(labels)) > 1 else None,
        )

        train_loader = DataLoader(
            SequenceDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            SequenceDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        self.model = self._build_model(input_size=len(features)).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        best_state = None
        best_val_loss = float("inf")
        no_improve_epochs = 0
        saved_checkpoints: List[str] = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            samples = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_x.size(0)
                samples += batch_x.size(0)

            avg_train_loss = running_loss / max(1, samples)
            val_loss = self._evaluate(val_loader, criterion, self.model)

            if val_loss + 1e-5 < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if checkpoint_interval and epoch % checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"{model_name}_epoch{epoch}.pt"
                )
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "scaler": self.scaler,
                        "feature_columns": features,
                        "sequence_length": self.sequence_length,
                        "model_type": self.model_type,
                    },
                    checkpoint_path,
                )
                saved_checkpoints.append(checkpoint_path)

            logger.info(
                "Epoch %s/%s - train_loss=%.4f val_loss=%.4f",
                epoch,
                epochs,
                avg_train_loss,
                val_loss,
            )

            if no_improve_epochs >= patience:
                logger.info("Early stopping triggered after %s epochs", epoch)
                break

        if best_state:
            self.model.load_state_dict(best_state)

        payload = {
            "model_state_dict": self.model.state_dict(),
            "scaler": self.scaler,
            "feature_columns": features,
            "sequence_length": self.sequence_length,
            "model_type": self.model_type,
        }
        final_model_path = os.path.join(save_dir, f"{model_name}.pt")
        torch.save(payload, final_model_path)

        self.is_fitted = True
        return {
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch,
            "final_model_path": final_model_path,
            "checkpoints": saved_checkpoints,
            "features_used": features,
        }

    def predict(self, df: pd.DataFrame, target_col: str = "target") -> np.ndarray:
        """
        Generate predictions for a new dataframe.
        """
        if not self.is_fitted or self.model is None or self.scaler is None:
            raise ValueError("Model must be trained or loaded before prediction.")

        sequences, _, _ = self._prepare_sequences(df, target_col, fit_scaler=False)
        dataset = SequenceDataset(sequences, np.zeros(len(sequences)))
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        self.model.eval()
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                preds.append(torch.sigmoid(logits).cpu().numpy())

        return np.concatenate(preds, axis=0)

    def load(self, model_path: str):
        """
        Load a saved model checkpoint (.pt file).
        """
        payload = torch.load(model_path, map_location=self.device)
        self.sequence_length = payload.get("sequence_length", self.sequence_length)
        self.feature_columns = payload.get("feature_columns", self.feature_columns)
        self.scaler = payload.get("scaler")
        self.model_type = payload.get("model_type", self.model_type)
        self.model = self._build_model(input_size=len(self.feature_columns or []))
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device)
        self.is_fitted = True
