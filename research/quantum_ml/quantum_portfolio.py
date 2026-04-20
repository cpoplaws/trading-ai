"""Quantum portfolio optimization prototype (Phase 8).

Uses PennyLane's QAOA to solve a binary asset-selection problem: given a pool of
candidate assets with expected returns and a covariance matrix, pick the subset
of size ``k`` that maximizes ``expected_return - gamma * risk``.

This is a simulator-only prototype. It proves the pipeline end-to-end on toy
data; it does not claim a quantum speed-up on any real portfolio task. See
``research/README.md`` for the rationale and limits.

Run directly:

    python -m research.quantum_ml.quantum_portfolio
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class PortfolioProblem:
    expected_returns: np.ndarray
    covariance: np.ndarray
    risk_aversion: float = 0.5
    cardinality: int | None = None
    cardinality_penalty: float = 1.0

    def __post_init__(self) -> None:
        self.expected_returns = np.asarray(self.expected_returns, dtype=float)
        self.covariance = np.asarray(self.covariance, dtype=float)
        if self.covariance.shape != (self.n_assets, self.n_assets):
            raise ValueError("covariance must be (n, n) matching expected_returns length")

    @property
    def n_assets(self) -> int:
        return self.expected_returns.shape[0]

    def objective(self, weights: np.ndarray) -> float:
        ret = float(self.expected_returns @ weights)
        risk = float(weights @ self.covariance @ weights)
        penalty = 0.0
        if self.cardinality is not None:
            penalty = self.cardinality_penalty * (weights.sum() - self.cardinality) ** 2
        return ret - self.risk_aversion * risk - penalty


def brute_force_best(problem: PortfolioProblem) -> tuple[np.ndarray, float]:
    """Exhaustively scan all 2^n subsets to obtain a reference optimum."""
    n = problem.n_assets
    best_weights = np.zeros(n)
    best_score = -np.inf
    for mask in range(1, 1 << n):
        weights = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
        score = problem.objective(weights)
        if score > best_score:
            best_score = score
            best_weights = weights
    return best_weights, best_score


def solve_qaoa(
    problem: PortfolioProblem,
    *,
    layers: int = 2,
    steps: int = 60,
    learning_rate: float = 0.2,
    seed: int = 7,
) -> dict[str, object]:
    """Run QAOA on the portfolio Ising Hamiltonian and sample the best bitstring."""
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "PennyLane is not installed. Install the research extra: pip install '.[research]'"
        ) from exc

    n = problem.n_assets
    # Map x_i in {0, 1} to s_i = (1 - z_i) / 2 where z_i is a Pauli-Z eigenvalue.
    # The portfolio objective becomes a sum of Z and ZZ terms plus a constant.
    # We negate because QAOA minimizes.
    returns = problem.expected_returns
    cov = problem.covariance
    gamma = problem.risk_aversion

    coeffs: list[float] = []
    observables: list[qml.operation.Observable] = []

    constant = 0.0
    for i in range(n):
        ret_i = returns[i]
        var_i = cov[i, i]
        # Contribution from -r_i * s_i + gamma * var_i * s_i^2 (s_i^2 = s_i for binary).
        linear_z = 0.5 * (ret_i - gamma * var_i)
        constant += -0.5 * (ret_i - gamma * var_i)
        coeffs.append(linear_z)
        observables.append(qml.PauliZ(i))

    for i in range(n):
        for j in range(i + 1, n):
            cov_ij = cov[i, j] + cov[j, i]
            if cov_ij == 0.0:
                continue
            # s_i * s_j expands to (1 - z_i - z_j + z_i z_j) / 4.
            zz_coeff = 0.25 * gamma * cov_ij
            coeffs.append(zz_coeff)
            observables.append(qml.PauliZ(i) @ qml.PauliZ(j))
            constant += 0.25 * gamma * cov_ij
            coeffs.append(-0.25 * gamma * cov_ij)
            observables.append(qml.PauliZ(i))
            coeffs.append(-0.25 * gamma * cov_ij)
            observables.append(qml.PauliZ(j))

    cost_h = qml.Hamiltonian(coeffs, observables)
    mixer_h = qml.Hamiltonian([1.0] * n, [qml.PauliX(i) for i in range(n)])

    dev = qml.device("default.qubit", wires=n)

    def qaoa_layer(params):
        qml.templates.ApproxTimeEvolution(cost_h, params[0], 1)
        qml.templates.ApproxTimeEvolution(mixer_h, params[1], 1)

    @qml.qnode(dev, interface="autograd")
    def circuit(params):
        for i in range(n):
            qml.Hadamard(wires=i)
        for layer in range(layers):
            qaoa_layer(params[layer])
        return qml.expval(cost_h)

    @qml.qnode(dev, interface="autograd")
    def sampler(params):
        for i in range(n):
            qml.Hadamard(wires=i)
        for layer in range(layers):
            qaoa_layer(params[layer])
        return qml.probs(wires=range(n))

    rng = np.random.default_rng(seed)
    params = pnp.array(rng.uniform(0, np.pi, size=(layers, 2)), requires_grad=True)
    optimizer = qml.GradientDescentOptimizer(stepsize=learning_rate)

    energies = []
    for _ in range(steps):
        params, energy = optimizer.step_and_cost(circuit, params)
        energies.append(float(energy))

    probs = np.asarray(sampler(params))
    best_index = int(np.argmax(probs))
    bits = np.array([(best_index >> i) & 1 for i in range(n)], dtype=float)
    # Flip to big-endian-from-wire-0 convention so bits[i] matches asset i.
    bits = bits[::-1]

    return {
        "bitstring": bits,
        "score": problem.objective(bits),
        "probability": float(probs[best_index]),
        "energy_history": energies,
        "constant_offset": constant,
    }


def _demo() -> None:
    rng = np.random.default_rng(42)
    n = 5
    returns = rng.uniform(0.05, 0.18, size=n)
    base = rng.uniform(-0.02, 0.05, size=(n, n))
    cov = base @ base.T / n + np.eye(n) * 0.02
    problem = PortfolioProblem(returns, cov, risk_aversion=1.0, cardinality=3)

    print("Brute-force reference (2^n scan):")
    ref_weights, ref_score = brute_force_best(problem)
    print(f"  picked: {ref_weights.astype(int).tolist()}  score={ref_score:.4f}")

    print("\nQAOA (PennyLane default.qubit):")
    result = solve_qaoa(problem, layers=2, steps=80)
    print(f"  picked: {result['bitstring'].astype(int).tolist()}  score={result['score']:.4f}")
    print(f"  sampled_probability={result['probability']:.3f}")


if __name__ == "__main__":
    _demo()
