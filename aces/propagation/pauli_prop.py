"""Pauli Propagation — Heisenberg-picture back-propagation simulator.

Computes ⟨0|U† O U|0⟩ by evolving the observable O backward through the
circuit using adjoint conjugation U_i† P U_i at each step.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

from aces.compiler.gate_rules import transform_pauli_string
from aces.compiler.pauli_transform import CompiledCircuit

_SELF_INVERSE = {"H", "X", "Y", "Z", "CX", "CNOT", "CZ"}


def _adjoint_gate_name(gate_name: str) -> str:
    """Return the adjoint gate name (S↔SDG, rotations keep name)."""
    upper = gate_name.upper()
    if upper in _SELF_INVERSE:
        return upper
    if upper == "S":
        return "SDG"
    if upper == "SDG":
        return "S"
    return upper


def _adjoint_params(gate_name: str, params: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Negate rotation angle for adjoint: R(θ)† = R(-θ)."""
    upper = gate_name.upper()
    if upper in ("RX", "RY", "RZ") and params is not None:
        return {"theta": -params.get("theta", 0.0)}
    return params


class PauliPropagator:
    """Heisenberg-picture Pauli back-propagation simulator."""

    def simulate(
        self,
        compiled_circuit: CompiledCircuit,
        observable_str: str,
        noise_rate: float = 0.0,
        coeff_threshold: float = 1e-6,
        weight_cutoff: Optional[int] = None,
    ) -> float:
        """Back-propagate an observable and return ⟨0|U† O U|0⟩."""
        num_qubits = len(observable_str)
        dictionary: Dict[str, float] = {observable_str: 1.0}

        for step in reversed(compiled_circuit.steps):
            new_dict: Dict[str, float] = defaultdict(float)
            adj_name = _adjoint_gate_name(step.gate_name)
            adj_params = _adjoint_params(step.gate_name, step.params)

            for pauli_str, coeff in dictionary.items():
                transformed = transform_pauli_string(
                    pauli_str=pauli_str,
                    gate_name=adj_name,
                    qubits=step.qubits,
                    num_qubits=num_qubits,
                    params=adj_params,
                )
                for phase, new_str in transformed:
                    new_dict[new_str] += coeff * phase

            if noise_rate > 0:
                factor = 1.0 - noise_rate
                for pauli_str in list(new_dict.keys()):
                    weight = sum(1 for c in pauli_str if c != "I")
                    if weight > 0:
                        new_dict[pauli_str] *= factor ** weight

            dictionary = {}
            for pauli_str, coeff in new_dict.items():
                if coeff_threshold > 0 and abs(coeff) < coeff_threshold:
                    continue
                if weight_cutoff is not None:
                    weight = sum(1 for c in pauli_str if c != "I")
                    if weight > weight_cutoff:
                        continue
                dictionary[pauli_str] = coeff

        expectation = 0.0
        for pauli_str, coeff in dictionary.items():
            if all(c in ("I", "Z") for c in pauli_str):
                expectation += coeff

        return expectation
