"""Entanglement Tracker — measures cross-qubit correlations from Pauli dictionary."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from aces.runtime.pauli_state import PauliState


@dataclass
class EntanglementSnapshot:
    """Snapshot of entanglement metrics at one time step."""
    step: int
    gate_name: str
    num_terms: int
    pair_strength: Dict[Tuple[int, int], float] = field(default_factory=dict)
    bloch_radii: Dict[int, float] = field(default_factory=dict)


class EntanglementTracker:
    """Tracks entanglement by measuring cross-qubit Pauli term magnitudes.

    Entanglement between qubits i and j is estimated by summing the magnitudes
    of all Pauli terms that have non-identity operators on BOTH qubits i and j.
    """

    def __init__(self):
        self._snapshots: List[EntanglementSnapshot] = []

    def record(self, step: int, gate_name: str, state: PauliState) -> EntanglementSnapshot:
        """Take a snapshot of the current entanglement structure."""
        n = state.num_qubits
        snap = EntanglementSnapshot(step=step, gate_name=gate_name, num_terms=state.num_terms)

        # Bloch radii
        for q in range(n):
            bv = state.bloch_vector(q)
            snap.bloch_radii[q] = float(np.linalg.norm(bv))

        # Pairwise entanglement strength
        for i in range(n):
            for j in range(i + 1, n):
                strength = 0.0
                for pauli_str, coeff in state.coeffs.items():
                    if pauli_str[i] != "I" and pauli_str[j] != "I":
                        strength += abs(coeff)
                snap.pair_strength[(i, j)] = strength

        self._snapshots.append(snap)
        return snap

    def entanglement_matrix(self, state: PauliState) -> np.ndarray:
        """Build an n×n entanglement strength matrix."""
        n = state.num_qubits
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                strength = 0.0
                for pauli_str, coeff in state.coeffs.items():
                    if pauli_str[i] != "I" and pauli_str[j] != "I":
                        strength += abs(coeff)
                mat[i, j] = strength
                mat[j, i] = strength
        return mat

    def history(self) -> List[Dict]:
        """Return history as list of dicts."""
        return [
            {
                "step": s.step,
                "gate": s.gate_name,
                "num_terms": s.num_terms,
                "pair_strength": {f"{a}-{b}": v for (a, b), v in s.pair_strength.items()},
                "bloch_radii": dict(s.bloch_radii),
            }
            for s in self._snapshots
        ]

    def strength_over_time(self, qubit_a: int, qubit_b: int) -> List[float]:
        """Get entanglement strength for a pair across all recorded steps."""
        pair = (min(qubit_a, qubit_b), max(qubit_a, qubit_b))
        return [s.pair_strength.get(pair, 0.0) for s in self._snapshots]

    @property
    def num_snapshots(self) -> int:
        return len(self._snapshots)
