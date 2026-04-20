"""Federated Averaging (FedAvg) prototype across simulated price-data nodes.

Each virtual client holds a shard of synthetic OHLC-derived features and trains
a logistic-regression head for next-bar direction. A coordinator averages the
client weights every round. Raw features never leave the client — only gradients
/ weight deltas transit — illustrating the privacy posture FL exists to enable.

This is a simulation (no network), PyTorch-only, no TensorFlow-Federated. The
goal is to exercise the protocol, not to win accuracy benchmarks.

Run directly:

    python -m research.federated_learning.federated_trainer
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


def _lazy_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "PyTorch is not installed. Install the research extra: pip install '.[research]'"
        ) from exc
    return torch, nn


@dataclass
class ClientShard:
    features: np.ndarray  # (n_samples, n_features)
    labels: np.ndarray    # (n_samples,) in {0, 1}

    @property
    def size(self) -> int:
        return int(self.features.shape[0])


def generate_synthetic_shards(
    n_clients: int = 3,
    samples_per_client: int = 512,
    n_features: int = 6,
    noise: float = 0.8,
    seed: int = 0,
) -> List[ClientShard]:
    """Build per-client datasets with slight distribution skew between clients."""
    rng = np.random.default_rng(seed)
    true_w = rng.normal(size=n_features)
    shards = []
    for client in range(n_clients):
        client_shift = rng.normal(scale=0.3, size=n_features)
        x = rng.normal(size=(samples_per_client, n_features)) + client_shift
        logits = x @ true_w + rng.normal(scale=noise, size=samples_per_client)
        y = (logits > 0).astype(np.int64)
        shards.append(ClientShard(features=x, labels=y))
    return shards


def _make_model(n_features: int):
    _, nn = _lazy_torch()
    model = nn.Sequential(
        nn.Linear(n_features, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )
    return model


def _client_update(
    global_state: dict,
    shard: ClientShard,
    *,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict:
    torch, nn = _lazy_torch()
    model = _make_model(shard.features.shape[1])
    model.load_state_dict(global_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    features = torch.from_numpy(shard.features).float()
    labels = torch.from_numpy(shard.labels).long()

    for _ in range(local_epochs):
        indices = torch.randperm(shard.size)
        for start in range(0, shard.size, batch_size):
            batch = indices[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(features[batch])
            loss = loss_fn(logits, labels[batch])
            loss.backward()
            optimizer.step()

    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _aggregate(states: Sequence[dict], weights: Sequence[float]) -> dict:
    torch, _ = _lazy_torch()
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("aggregation weights must sum to a positive value")
    averaged: dict = {}
    for key in states[0].keys():
        stacked = torch.stack([state[key].float() * (w / total) for state, w in zip(states, weights)])
        averaged[key] = stacked.sum(dim=0)
    return averaged


def _evaluate(global_state: dict, shard: ClientShard) -> float:
    torch, _ = _lazy_torch()
    model = _make_model(shard.features.shape[1])
    model.load_state_dict(global_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(shard.features).float())
        preds = logits.argmax(dim=1).numpy()
    return float((preds == shard.labels).mean())


def federated_average(
    shards: Sequence[ClientShard],
    *,
    rounds: int = 5,
    local_epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.05,
) -> Tuple[dict, List[float]]:
    if not shards:
        raise ValueError("at least one client shard is required")

    torch, _ = _lazy_torch()
    torch.manual_seed(0)

    n_features = shards[0].features.shape[1]
    global_model = _make_model(n_features)
    global_state = {k: v.detach().clone() for k, v in global_model.state_dict().items()}

    history: List[float] = []
    for _ in range(rounds):
        client_states = [
            _client_update(
                global_state,
                shard,
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
            for shard in shards
        ]
        global_state = _aggregate(client_states, [shard.size for shard in shards])
        avg_accuracy = float(np.mean([_evaluate(global_state, shard) for shard in shards]))
        history.append(avg_accuracy)

    return global_state, history


def _demo() -> None:
    shards = generate_synthetic_shards(n_clients=3, samples_per_client=600)
    _, history = federated_average(shards, rounds=6, local_epochs=2)
    print("Federated accuracy by round (simulated 3-client FedAvg):")
    for round_idx, accuracy in enumerate(history, start=1):
        print(f"  round {round_idx}: avg_client_acc={accuracy:.3f}")


if __name__ == "__main__":
    _demo()
