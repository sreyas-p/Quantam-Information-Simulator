"""Pruning Engine — removes weak Pauli terms for compression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from aces.runtime.pauli_state import PauliState


@dataclass
class PruneResult:
    """Result of a single pruning pass."""
    terms_before: int
    terms_after: int
    terms_pruned: int
    total_dropped_magnitude: float

    @property
    def compression_ratio(self) -> float:
        if self.terms_before == 0:
            return 1.0
        return self.terms_after / self.terms_before

    @property
    def error_estimate(self) -> float:
        """Dropped magnitude is an upper bound on the error introduced."""
        return self.total_dropped_magnitude

    def __repr__(self) -> str:
        return (
            f"PruneResult(pruned={self.terms_pruned}, "
            f"ratio={self.compression_ratio:.2%}, "
            f"error≤{self.error_estimate:.6f})"
        )


class PruningEngine:
    """Prunes Pauli terms whose coefficient magnitude falls below threshold.

    When |c_P| < threshold, remove P from the dictionary.

    Usage:
        engine = PruningEngine(threshold=0.01)
        result = engine.prune(state)
    """

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self._history: List[PruneResult] = []

    def prune(self, state: PauliState) -> PruneResult:
        """Run one pruning pass. Modifies state in-place.

        Never prunes the all-identity string (normalization).
        """
        identity = "I" * state.num_qubits
        terms_before = len(state.coeffs)
        dropped_mag = 0.0
        pruned_count = 0

        keys_to_remove = []
        for pauli_str, coeff in state.coeffs.items():
            if pauli_str == identity:
                continue  # Never prune identity
            if abs(coeff) < self.threshold:
                keys_to_remove.append(pauli_str)
                dropped_mag += abs(coeff)
                pruned_count += 1

        for key in keys_to_remove:
            del state.coeffs[key]

        result = PruneResult(
            terms_before=terms_before,
            terms_after=len(state.coeffs),
            terms_pruned=pruned_count,
            total_dropped_magnitude=dropped_mag,
        )
        self._history.append(result)
        return result

    @property
    def history(self) -> List[PruneResult]:
        return list(self._history)

    @property
    def total_pruned(self) -> int:
        return sum(r.terms_pruned for r in self._history)

    @property
    def cumulative_error(self) -> float:
        return sum(r.total_dropped_magnitude for r in self._history)
