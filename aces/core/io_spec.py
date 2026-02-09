"""Input/Output specifications for ACES compilation and runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, FrozenSet
from enum import Enum


class MeasurementBasis(Enum):
    COMPUTATIONAL = "Z"
    X_BASIS = "X"
    Y_BASIS = "Y"


@dataclass(frozen=True)
class MeasurementSpec:
    """Specification of what to measure."""
    qubits: FrozenSet[int]
    basis: MeasurementBasis = MeasurementBasis.COMPUTATIONAL
    observables: Optional[tuple] = None
    
    @classmethod
    def computational(cls, qubits: Set[int], observables: Optional[List[str]] = None):
        return cls(
            qubits=frozenset(qubits),
            basis=MeasurementBasis.COMPUTATIONAL,
            observables=tuple(observables) if observables else None
        )


@dataclass(frozen=True)
class NoiseModelSpec:
    """Noise model parameters for compile-time decisions."""
    single_qubit_depolarizing: float = 0.0
    two_qubit_depolarizing: float = 0.0
    readout_error: float = 0.0
    t1: Optional[float] = None
    t2: Optional[float] = None
    
    def decay_per_gate(self, gate_type: str) -> float:
        if gate_type == "single":
            return 1.0 - self.single_qubit_depolarizing
        elif gate_type == "two":
            return 1.0 - self.two_qubit_depolarizing
        return 1.0


@dataclass(frozen=True)
class CompileTimeInput:
    """Compile-time inputs that determine ACES executable structure."""
    circuit: Any
    measurement_spec: MeasurementSpec
    noise_model: Optional[NoiseModelSpec] = None
    entanglement_budget: float = 100.0
    
    @classmethod
    def from_circuit(cls, circuit, measured_qubits: Set[int],
                     observables: Optional[List[str]] = None,
                     noise_model: Optional[NoiseModelSpec] = None) -> CompileTimeInput:
        return cls(
            circuit=circuit,
            measurement_spec=MeasurementSpec.computational(measured_qubits, observables),
            noise_model=noise_model
        )


@dataclass
class RuntimeInput:
    """Runtime inputs that don't require recompilation."""
    gate_params: Dict[str, float] = field(default_factory=dict)
    classical_flags: Dict[str, bool] = field(default_factory=dict)
    num_samples: int = 0
    observables: Optional[List[str]] = None
    
    @classmethod
    def with_params(cls, **params) -> RuntimeInput:
        return cls(gate_params=params)


@dataclass
class RuntimeOutput:
    """Output from ACES execution."""
    pauli_expectations: Dict[str, float] = field(default_factory=dict)
    marginal_probabilities: Dict[str, float] = field(default_factory=dict)
    samples: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def expectation(self, pauli_string: str) -> float:
        if pauli_string not in self.pauli_expectations:
            raise KeyError(f"Observable '{pauli_string}' not computed")
        return self.pauli_expectations[pauli_string]
    
    def probability(self, bitstring: str) -> float:
        return self.marginal_probabilities.get(bitstring, 0.0)
    
    @property
    def num_samples(self) -> int:
        return len(self.samples) if self.samples else 0
    
    def __repr__(self) -> str:
        return f"RuntimeOutput(observables={len(self.pauli_expectations)}, samples={self.num_samples})"
