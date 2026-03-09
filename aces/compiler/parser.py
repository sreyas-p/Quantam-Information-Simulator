"""Circuit parser — converts circuit descriptions into typed GateOp objects."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GateOp:
    """A single gate operation in the circuit."""
    name: str
    qubits: Tuple[int, ...]
    params: Optional[Dict[str, float]] = None

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    def __repr__(self) -> str:
        p = f", params={self.params}" if self.params else ""
        return f"GateOp({self.name}, qubits={self.qubits}{p})"


def parse_circuit(gate_list: List[Dict[str, Any]]) -> List[GateOp]:
    """Parse a circuit description into a list of GateOps.

    Args:
        gate_list: List of dicts with "name", "qubits", and optional "params".
            Example: [{"name": "H", "qubits": [0]}, {"name": "CX", "qubits": [0, 1]}]

    Returns:
        List of GateOp objects.
    """
    ops = []
    for gate in gate_list:
        name = gate["name"].upper()
        qubits = tuple(gate["qubits"])
        params = gate.get("params", None)
        ops.append(GateOp(name=name, qubits=qubits, params=params))
    return ops
