"""Pauli conjugation rules for quantum gates.

Each gate defines how Pauli operators transform under conjugation: U P U†.
For Clifford gates, one Pauli maps to exactly one Pauli (with ±1 phase).
For rotation gates, one Pauli maps to a weighted sum of two Paulis.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

# -- Single-qubit Clifford rules: gate -> {input_pauli: (phase, output_pauli)} --

SINGLE_QUBIT_RULES: Dict[str, Dict[str, Tuple[float, str]]] = {
    "H": {"I": (1, "I"), "X": (1, "Z"), "Y": (-1, "Y"), "Z": (1, "X")},
    "X": {"I": (1, "I"), "X": (1, "X"), "Y": (-1, "Y"), "Z": (-1, "Z")},
    "Y": {"I": (1, "I"), "X": (-1, "X"), "Y": (1, "Y"), "Z": (-1, "Z")},
    "Z": {"I": (1, "I"), "X": (-1, "X"), "Y": (-1, "Y"), "Z": (1, "Z")},
    "S": {"I": (1, "I"), "X": (1, "Y"), "Y": (-1, "X"), "Z": (1, "Z")},
    "SDG": {"I": (1, "I"), "X": (-1, "Y"), "Y": (1, "X"), "Z": (1, "Z")},
}

# -- Two-qubit Clifford rules: gate -> {(ctrl_pauli, tgt_pauli): (phase, out_ctrl, out_tgt)} --

CNOT_RULES: Dict[Tuple[str, str], Tuple[float, str, str]] = {
    ("I", "I"): (1, "I", "I"), ("I", "X"): (1, "I", "X"),
    ("I", "Y"): (1, "Z", "Y"), ("I", "Z"): (1, "Z", "Z"),
    ("X", "I"): (1, "X", "X"), ("X", "X"): (1, "X", "I"),
    ("X", "Y"): (1, "Y", "Z"), ("X", "Z"): (-1, "Y", "Y"),
    ("Y", "I"): (1, "Y", "X"), ("Y", "X"): (1, "Y", "I"),
    ("Y", "Y"): (-1, "X", "Z"), ("Y", "Z"): (1, "X", "Y"),
    ("Z", "I"): (1, "Z", "I"), ("Z", "X"): (1, "Z", "X"),
    ("Z", "Y"): (1, "I", "Y"), ("Z", "Z"): (1, "I", "Z"),
}

CZ_RULES: Dict[Tuple[str, str], Tuple[float, str, str]] = {
    ("I", "I"): (1, "I", "I"), ("I", "X"): (1, "Z", "X"),
    ("I", "Y"): (1, "Z", "Y"), ("I", "Z"): (1, "I", "Z"),
    ("X", "I"): (1, "X", "Z"), ("X", "X"): (1, "Y", "Y"),
    ("X", "Y"): (-1, "Y", "X"), ("X", "Z"): (1, "X", "I"),
    ("Y", "I"): (1, "Y", "Z"), ("Y", "X"): (-1, "X", "Y"),
    ("Y", "Y"): (1, "X", "X"), ("Y", "Z"): (1, "Y", "I"),
    ("Z", "I"): (1, "Z", "I"), ("Z", "X"): (1, "I", "X"),
    ("Z", "Y"): (1, "I", "Y"), ("Z", "Z"): (1, "Z", "Z"),
}

TWO_QUBIT_RULES: Dict[str, Dict[Tuple[str, str], Tuple[float, str, str]]] = {
    "CX": CNOT_RULES,
    "CNOT": CNOT_RULES,
    "CZ": CZ_RULES,
}

# -- Rotation gates: return list of (coefficient, output_pauli) --


def _rotation_rule(
    pauli: str, axis: str, theta: float
) -> List[Tuple[float, str]]:
    """Transform a single-qubit Pauli under rotation about `axis` by angle `theta`.

    R_axis(θ) P R_axis(θ)†:
      - P == axis or P == I: unchanged
      - otherwise: cos(θ)P ± sin(θ)Q  (where Q is the third Pauli)
    """
    if pauli == "I" or pauli == axis:
        return [(1.0, pauli)]

    c, s = math.cos(theta), math.sin(theta)
    # The third Pauli and sign follow the cyclic rule X→Y→Z→X
    cycle = {"X": "Y", "Y": "Z", "Z": "X"}
    reverse = {"Y": "X", "Z": "Y", "X": "Z"}

    if cycle[axis] == pauli:
        # e.g. Rz: X → cos(θ)X + sin(θ)Y
        third = cycle[pauli]
        return [(c, pauli), (s, third)]
    else:
        # e.g. Rz: Y → cos(θ)Y - sin(θ)X
        third = reverse[pauli]
        return [(c, pauli), (-s, third)]


def transform_pauli_string(
    pauli_str: str,
    gate_name: str,
    qubits: Tuple[int, ...],
    num_qubits: int,
    params: Optional[Dict[str, float]] = None,
) -> List[Tuple[float, str]]:
    """Transform an n-qubit Pauli string under gate conjugation.

    Args:
        pauli_str: e.g. "IXZY" (length == num_qubits)
        gate_name: e.g. "H", "CX", "RZ"
        qubits: qubit indices the gate acts on
        num_qubits: total qubit count
        params: optional parameters for rotation gates (key "theta")

    Returns:
        List of (coefficient, new_pauli_string) pairs.
    """
    gate = gate_name.upper()

    # -- Single-qubit Clifford --
    if gate in SINGLE_QUBIT_RULES and len(qubits) == 1:
        q = qubits[0]
        p = pauli_str[q]
        phase, new_p = SINGLE_QUBIT_RULES[gate][p]
        new_str = pauli_str[:q] + new_p + pauli_str[q + 1:]
        return [(phase, new_str)]

    # -- Two-qubit Clifford --
    if gate in TWO_QUBIT_RULES and len(qubits) == 2:
        rules = TWO_QUBIT_RULES[gate]
        qa, qb = qubits
        pa, pb = pauli_str[qa], pauli_str[qb]
        phase, new_a, new_b = rules[(pa, pb)]
        chars = list(pauli_str)
        chars[qa] = new_a
        chars[qb] = new_b
        return [(phase, "".join(chars))]

    # -- Rotation gates --
    if gate in ("RX", "RY", "RZ") and len(qubits) == 1:
        axis = gate[1]  # "X", "Y", or "Z"
        theta = params.get("theta", 0.0) if params else 0.0
        q = qubits[0]
        p = pauli_str[q]
        terms = _rotation_rule(p, axis, theta)
        result = []
        for coeff, new_p in terms:
            new_str = pauli_str[:q] + new_p + pauli_str[q + 1:]
            result.append((coeff, new_str))
        return result

    raise ValueError(f"Unknown gate: {gate_name} on {len(qubits)} qubits")
