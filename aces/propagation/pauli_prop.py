"""Pauli Propagation — Heisenberg-picture back-propagation simulator.

Given an observable O and a circuit U = U_n ... U_1, computes
⟨0|U† O U|0⟩ by back-propagating O through the circuit gates in
reverse order, applying the ADJOINT conjugation U_i† P U_i at
each step.

This is the "dual" of ACES's Schrödinger picture: instead of evolving
the state forward (ρ → U ρ U†), we evolve the observable backward
(O → U† O U) and then overlap with the initial |0⟩ state.

Key detail: ACES uses UPU† to transform Pauli strings (Schrödinger).
Heisenberg back-propagation needs U†PU (the ADJOINT). For self-inverse
gates (H, X, CX, CZ) these are identical, but:
  - S → use SDG rules (S† = S^dagger)
  - RX/RY/RZ(θ) → negate the angle (R(θ)† = R(-θ))
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

from aces.compiler.gate_rules import transform_pauli_string
from aces.compiler.pauli_transform import CompiledCircuit


# Gates that are self-inverse: U† = U
_SELF_INVERSE = {"H", "X", "Y", "Z", "CX", "CNOT", "CZ"}


def _adjoint_gate_name(gate_name: str) -> str:
    """Return the gate name for the adjoint (inverse) gate.

    Self-inverse gates map to themselves.
    S maps to SDG, SDG maps to S.
    Rotation gates stay the same (angle is negated separately).
    """
    upper = gate_name.upper()
    if upper in _SELF_INVERSE:
        return upper
    if upper == "S":
        return "SDG"
    if upper == "SDG":
        return "S"
    # Rotation gates: same name, angle negated in params
    if upper in ("RX", "RY", "RZ"):
        return upper
    return upper


def _adjoint_params(gate_name: str, params: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Return the parameters for the adjoint gate.

    For rotation gates, negate the angle θ since R(θ)† = R(-θ).
    """
    upper = gate_name.upper()
    if upper in ("RX", "RY", "RZ") and params is not None:
        return {"theta": -params.get("theta", 0.0)}
    return params


class PauliPropagator:
    """Heisenberg-picture Pauli back-propagation simulator.

    For each observable, starts with {observable: 1.0} and propagates
    backward through the circuit using adjoint gate conjugation.
    The expectation value is extracted by overlapping the final
    dictionary with the |0⟩^⊗n state (all I/Z-only strings have
    coefficient 1.0 in the zero state).
    """

    def simulate(
        self,
        compiled_circuit: CompiledCircuit,
        observable_str: str,
        noise_rate: float = 0.0,
        coeff_threshold: float = 1e-6,
        weight_cutoff: Optional[int] = None,
    ) -> float:
        """Back-propagate a single observable through the circuit.

        Args:
            compiled_circuit: The compiled circuit to propagate through.
            observable_str: Pauli string observable, e.g. "ZIIIII".
            noise_rate: Per-qubit depolarizing noise rate (0 = noiseless).
            coeff_threshold: Drop terms with |coeff| below this value.
            weight_cutoff: If set, drop terms with Pauli weight above this.

        Returns:
            The expectation value ⟨0|U† O U|0⟩.
        """
        num_qubits = len(observable_str)

        # Start with the observable as a single-term dictionary
        dictionary: Dict[str, float] = {observable_str: 1.0}

        # Iterate through gates in REVERSE order (Heisenberg picture)
        # For each gate U_i, we apply the ADJOINT: U_i† P U_i
        for step in reversed(compiled_circuit.steps):
            new_dict: Dict[str, float] = defaultdict(float)

            # Compute adjoint gate
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

            # Apply noise: scale non-identity terms by (1 - noise_rate)^weight
            if noise_rate > 0:
                factor = 1.0 - noise_rate
                for pauli_str in list(new_dict.keys()):
                    weight = sum(1 for c in pauli_str if c != "I")
                    if weight > 0:
                        new_dict[pauli_str] *= factor ** weight

            # Coefficient truncation
            dictionary = {}
            for pauli_str, coeff in new_dict.items():
                if coeff_threshold > 0 and abs(coeff) < coeff_threshold:
                    continue
                # Weight cutoff
                if weight_cutoff is not None:
                    weight = sum(1 for c in pauli_str if c != "I")
                    if weight > weight_cutoff:
                        continue
                dictionary[pauli_str] = coeff

        # Extract expectation value: overlap with |0⟩^⊗n state.
        # The zero state has coefficient 1.0 for every I/Z-only string,
        # so the expectation is the sum of coefficients of all such strings.
        expectation = 0.0
        for pauli_str, coeff in dictionary.items():
            if all(c in ("I", "Z") for c in pauli_str):
                expectation += coeff

        return expectation

