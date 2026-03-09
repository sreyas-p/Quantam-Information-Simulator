"""Pauli State — the core state representation for ACES.

The quantum state is stored as a dictionary:
    {pauli_string: coefficient}

where each key is a string like "IXZY" and each value is a real float.

The density matrix is reconstructed as:
    ρ = (1/2^n) Σ_P c_P · P
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from aces.compiler.gate_rules import transform_pauli_string
from aces.compiler.pauli_transform import CompiledCircuit, CompiledStep


class PauliState:
    """Dictionary-based Pauli state representation.

    Usage:
        state = PauliState.zero_state(3)  # |000⟩
        state.apply_step(CompiledStep("H", (0,)))
        print(state.bloch_vector(0))
    """

    def __init__(self, num_qubits: int, coeffs: Optional[Dict[str, float]] = None):
        self.num_qubits = num_qubits
        self.coeffs: Dict[str, float] = coeffs if coeffs is not None else {}

    @classmethod
    def zero_state(cls, num_qubits: int) -> PauliState:
        """Initialize |0⟩^⊗n state.

        For |0⟩ = (I+Z)/2 per qubit, the nonzero Pauli terms are all
        strings composed of only I and Z, each with coefficient 1.0.
        """
        coeffs: Dict[str, float] = {}
        for combo in itertools.product("IZ", repeat=num_qubits):
            coeffs["".join(combo)] = 1.0
        return cls(num_qubits=num_qubits, coeffs=coeffs)

    def apply_step(self, step: CompiledStep) -> None:
        """Apply a single compiled gate step by rewriting Pauli terms."""
        new_coeffs: Dict[str, float] = defaultdict(float)

        for pauli_str, coeff in self.coeffs.items():
            transformed = transform_pauli_string(
                pauli_str=pauli_str,
                gate_name=step.gate_name,
                qubits=step.qubits,
                num_qubits=self.num_qubits,
                params=step.params,
            )
            for phase, new_str in transformed:
                new_coeffs[new_str] += coeff * phase

        self.coeffs = dict(new_coeffs)

    def run_circuit(self, compiled: CompiledCircuit) -> None:
        """Execute all steps of a compiled circuit."""
        for step in compiled.steps:
            self.apply_step(step)

    def expectation(self, pauli_str: str) -> float:
        """Get the expectation value ⟨P⟩ = c_P."""
        return self.coeffs.get(pauli_str, 0.0)

    def bloch_vector(self, qubit: int) -> Tuple[float, float, float]:
        """Extract the Bloch vector (⟨X⟩, ⟨Y⟩, ⟨Z⟩) for a single qubit.

        ⟨σ⟩ for qubit q = sum of coefficients where σ appears at position q
        and I appears at all other positions.
        """
        rx, ry, rz = 0.0, 0.0, 0.0
        identity_mask = "I" * self.num_qubits

        for pauli_str, coeff in self.coeffs.items():
            # Check: all positions except `qubit` must be I
            is_single_qubit = all(
                pauli_str[i] == "I" for i in range(self.num_qubits) if i != qubit
            )
            if not is_single_qubit:
                continue

            p = pauli_str[qubit]
            if p == "X":
                rx += coeff
            elif p == "Y":
                ry += coeff
            elif p == "Z":
                rz += coeff

        return (rx, ry, rz)

    def measurement_probabilities(self, qubit: int) -> Dict[str, float]:
        """Compute P(0) and P(1) for a single qubit."""
        _, _, rz = self.bloch_vector(qubit)
        p0 = (1 + rz) / 2
        p1 = (1 - rz) / 2
        return {"0": max(0.0, p0), "1": max(0.0, p1)}

    @property
    def num_terms(self) -> int:
        """Number of active Pauli terms in the dictionary."""
        return sum(1 for v in self.coeffs.values() if abs(v) > 1e-15)

    @property
    def naive_size(self) -> int:
        """Naive full state size: 4^n Pauli strings."""
        return 4 ** self.num_qubits

    @property
    def compression_ratio(self) -> float:
        """Ratio of actual terms to naive size."""
        return self.num_terms / self.naive_size if self.naive_size > 0 else 0

    def copy(self) -> PauliState:
        return PauliState(num_qubits=self.num_qubits, coeffs=dict(self.coeffs))

    def __repr__(self) -> str:
        return f"PauliState(n={self.num_qubits}, terms={self.num_terms})"
