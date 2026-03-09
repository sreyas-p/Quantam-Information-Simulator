"""Pauli transform compiler — compiles circuits into Pauli rewrite instructions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from aces.compiler.parser import GateOp, parse_circuit


@dataclass(frozen=True)
class CompiledStep:
    """A single compiled Pauli transformation instruction."""
    gate_name: str
    qubits: Tuple[int, ...]
    params: Optional[Dict[str, float]] = None


@dataclass
class CompiledCircuit:
    """Compiled circuit — a frozen list of Pauli transformation steps.

    Runtime receives only this object; it never sees the original circuit.
    """
    num_qubits: int
    measured_qubits: Set[int]
    steps: List[CompiledStep]

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return f"CompiledCircuit(qubits={self.num_qubits}, steps={self.num_steps})"


class CircuitCompiler:
    """Compiles a gate sequence into a CompiledCircuit.

    Usage:
        compiler = CircuitCompiler()
        compiled = compiler.compile(
            [{"name": "H", "qubits": [0]}, {"name": "CX", "qubits": [0, 1]}],
            num_qubits=2,
            measured_qubits={0, 1},
        )
    """

    def compile(
        self,
        gate_sequence: List[Dict[str, Any]],
        num_qubits: int,
        measured_qubits: Optional[Set[int]] = None,
    ) -> CompiledCircuit:
        """Compile gate sequence into frozen Pauli transformation steps."""
        if measured_qubits is None:
            measured_qubits = set(range(num_qubits))

        ops = parse_circuit(gate_sequence)
        steps = [
            CompiledStep(
                gate_name=op.name,
                qubits=op.qubits,
                params=op.params,
            )
            for op in ops
        ]

        return CompiledCircuit(
            num_qubits=num_qubits,
            measured_qubits=measured_qubits,
            steps=steps,
        )
