"""Noise Models — coefficient scaling for noise channels."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List

from aces.runtime.pauli_state import PauliState


@dataclass
class NoiseResult:
    """Result of applying noise."""
    channel: str
    avg_shrinkage: float
    terms_affected: int


class NoiseModel:
    """Base class for noise models operating on Pauli dictionaries."""

    name: str = "base"

    def apply(self, state: PauliState, params: Dict[str, float]) -> NoiseResult:
        raise NotImplementedError


class DepolarizingNoise(NoiseModel):
    """Depolarizing noise: every non-identity coefficient *= (1 - p).

    This is the Pauli-basis equivalent of ρ → (1-p)ρ + p·I/2^n.
    Elegant: noise is just scaling.
    """

    name = "depolarizing"

    def apply(self, state: PauliState, params: Dict[str, float]) -> NoiseResult:
        p = params.get("p", 0.0)
        factor = 1.0 - p
        identity = "I" * state.num_qubits
        affected = 0

        for pauli_str in list(state.coeffs.keys()):
            if pauli_str == identity:
                continue
            state.coeffs[pauli_str] *= factor
            affected += 1

        return NoiseResult(channel=self.name, avg_shrinkage=factor, terms_affected=affected)


class DephasingNoise(NoiseModel):
    """Dephasing noise: coefficients with X or Y on target qubit get scaled.

    Simulates loss of coherence — Z information preserved, X/Y information decays.
    """

    name = "dephasing"

    def apply(
        self,
        state: PauliState,
        params: Dict[str, float],
        target_qubit: int = -1,
    ) -> NoiseResult:
        p = params.get("p", 0.0)
        factor = 1.0 - p
        affected = 0

        for pauli_str in list(state.coeffs.keys()):
            should_scale = False
            if target_qubit >= 0:
                # Scale only terms with X or Y on target qubit
                if pauli_str[target_qubit] in ("X", "Y"):
                    should_scale = True
            else:
                # Apply to all qubits: scale any term with X or Y anywhere
                if any(c in ("X", "Y") for c in pauli_str):
                    should_scale = True

            if should_scale:
                state.coeffs[pauli_str] *= factor
                affected += 1

        return NoiseResult(channel=self.name, avg_shrinkage=factor, terms_affected=affected)


class GlobalDepolarizingNoise(NoiseModel):
    """Per-qubit depolarizing noise applied to every qubit.

    For each qubit, non-identity Pauli on that qubit gets scaled by (1-p).
    A term with k non-identity Paulis gets scaled by (1-p)^k.
    """

    name = "global_depolarizing"

    def apply(self, state: PauliState, params: Dict[str, float]) -> NoiseResult:
        p = params.get("p", 0.0)
        factor = 1.0 - p
        affected = 0

        for pauli_str in list(state.coeffs.keys()):
            # Count non-identity characters
            weight = sum(1 for c in pauli_str if c != "I")
            if weight > 0:
                state.coeffs[pauli_str] *= factor ** weight
                affected += 1

        avg_shrink = factor if affected > 0 else 1.0
        return NoiseResult(channel=self.name, avg_shrinkage=avg_shrink, terms_affected=affected)
