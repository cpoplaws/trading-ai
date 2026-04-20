# Research — Phase 8 frontier prototypes

Two timeboxed prototypes live here. Both are **simulations**; neither is wired
into the live trading path and neither is part of the default install.

```
pip install '.[research]'
python -m research.quantum_ml.quantum_portfolio
python -m research.federated_learning.federated_trainer
```

## `quantum_ml/quantum_portfolio.py`

QAOA portfolio selection on PennyLane's `default.qubit` simulator. Encodes a
mean-variance objective (with an optional cardinality penalty) as an Ising
Hamiltonian, trains the variational parameters with gradient descent, and
samples the most-probable bitstring as the selected subset. The brute-force
baseline for `n ≤ ~20` is included so you can see when QAOA finds the optimum
and when it does not.

Why PennyLane over Qiskit: lighter install, no IBM Quantum account needed, and
the simulator is fast enough for the prototype range (`n ≤ 16` wires comfortably
on a laptop). For real quantum hardware you would swap the device.

Limits:

- Runtime scales exponentially in `n` (simulator). Past ~22 wires you need
  hardware or tensor-network backends.
- Barren-plateau risk grows with layer count — keep `layers ≤ 4` for toy sizes.
- No proven quantum advantage on this problem class yet; this is a pipeline
  exercise, not a claim.

## `federated_learning/federated_trainer.py`

FedAvg across `N` simulated clients, PyTorch only. Generates three synthetic
OHLC-style shards with per-client distribution skew, trains a small MLP locally
on each, then averages the state dicts weighted by client dataset size. Accuracy
is reported per round as a sanity check that the aggregated model is actually
learning.

Why not TensorFlow-Federated: TFF's install story is routinely broken across
Python / TF pairings and pulls a heavy dependency tree. For a *protocol*
demonstration, plain PyTorch + numpy is honest and reproducible.

Limits:

- Single-process simulation — no actual network layer, no secure aggregation,
  no differential privacy. All of those sit on top of FedAvg and are out of
  scope here.
- Synthetic data. Swapping in real per-venue feature shards is the obvious
  next step if we decide to productionize.
- Clients are homogeneous in model architecture. Heterogeneous FL (FedProx,
  FedMA) would need a different aggregation rule.

## Where this goes next

Either prototype graduating to production requires (in order):
1. Replace synthetic data with a real per-client / per-asset feature pipeline.
2. Benchmark against the classical solver (`cvxpy` mean-variance, centralized
   MLP) under the same compute budget — if classical wins flat, shelve it.
3. Decide whether the privacy or optimization property is actually worth the
   infra cost. "Cool tech" alone is not a shipping criterion.
