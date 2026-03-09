"""Metrics — extract physical quantities from PauliState."""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

from aces.runtime.pauli_state import PauliState


def compute_bloch_vectors(state: PauliState) -> Dict[int, Tuple[float, float, float]]:
    """Extract Bloch vectors for all qubits."""
    return {q: state.bloch_vector(q) for q in range(state.num_qubits)}


def compute_purity(state: PauliState) -> float:
    """Compute state purity: Tr(ρ²) = (1/2^n) Σ c_P².

    For a pure state, purity = 1. For maximally mixed, purity = 1/2^n.
    """
    sum_sq = sum(c ** 2 for c in state.coeffs.values())
    return sum_sq / (2 ** state.num_qubits)


def entanglement_matrix(state: PauliState) -> np.ndarray:
    """Build n×n matrix of pairwise entanglement strength.

    Strength(i,j) = Σ |c_P| for all P with non-identity on both i and j.
    """
    n = state.num_qubits
    mat = np.zeros((n, n))
    for pauli_str, coeff in state.coeffs.items():
        # Find which qubits are non-identity
        active = [i for i in range(n) if pauli_str[i] != "I"]
        for idx_a in range(len(active)):
            for idx_b in range(idx_a + 1, len(active)):
                i, j = active[idx_a], active[idx_b]
                mat[i, j] += abs(coeff)
                mat[j, i] += abs(coeff)
    return mat


def compression_stats(state: PauliState) -> Dict[str, float]:
    """Compute compression statistics."""
    naive = state.naive_size
    actual = state.num_terms
    return {
        "naive_size": float(naive),
        "actual_terms": float(actual),
        "compression_ratio": actual / naive if naive > 0 else 0,
        "savings_pct": 100.0 * (1 - actual / naive) if naive > 0 else 0,
    }
