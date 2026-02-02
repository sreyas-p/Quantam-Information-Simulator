"""
Main ACES compiler.

Compiles a quantum circuit + measurement specification into a fixed
causal object that can be efficiently executed at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional, Any
import numpy as np

from aces.core.ceg import CausalEntanglementGraph, CEGBuilder
from aces.core.gates import GateLibrary, Gate
from aces.compiler.parser import ParsedCircuit, ParsedGate, parse_qiskit_circuit
from aces.compiler.lightcone import (
    CircuitDAG, analyze_lightcone, LightconeAnalysis
)
from aces.compiler.budget import EntanglementBudget, estimate_budget_for_circuit


@dataclass
class FrozenGate:
    """
    A gate in the frozen execution sequence.
    
    Contains everything needed to apply the gate at runtime.
    """
    index: int
    name: str
    qubits: Tuple[int, ...]
    params: Dict[str, float]
    is_parametric: bool
    param_names: Tuple[str, ...]
    gate_object: Gate
    
    def get_params(self, runtime_params: Dict[str, float]) -> Dict[str, float]:
        """Get parameters, substituting runtime values if parametric."""
        if not self.is_parametric:
            return self.params
        
        result = dict(self.params)
        for name in self.param_names:
            if name in runtime_params:
                # Map to the gate's expected param name
                # For single-parameter gates, use "theta"
                if len(self.param_names) == 1:
                    result["theta"] = runtime_params[name]
                else:
                    result[name] = runtime_params[name]
        return result


@dataclass
class CompiledACES:
    """
    A compiled ACES object ready for execution.
    
    This is the output of the compilation process and contains:
    - The frozen gate sequence (only gates in the lightcone)
    - The Causal Entanglement Graph
    - Metadata about the circuit and compilation
    
    At runtime, this object is passed to ACESRuntime along with
    parameter values to produce outputs.
    
    Attributes:
        num_qubits: Total number of qubits
        measured_qubits: Set of qubit indices being measured
        gate_sequence: Frozen sequence of gates to execute
        ceg: The Causal Entanglement Graph
        lightcone: Lightcone analysis results
        budget: Entanglement budget configuration
        parameter_names: Names of runtime parameters
        metadata: Additional compilation metadata
    """
    num_qubits: int
    measured_qubits: Set[int]
    gate_sequence: List[FrozenGate]
    ceg: CausalEntanglementGraph
    lightcone: LightconeAnalysis
    budget: EntanglementBudget
    parameter_names: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_gates(self) -> int:
        """Number of gates in the frozen sequence."""
        return len(self.gate_sequence)
    
    @property
    def depth(self) -> int:
        """Circuit depth (from metadata)."""
        return self.metadata.get("depth", self.num_gates)
    
    @property
    def original_num_gates(self) -> int:
        """Original number of gates before lightcone pruning."""
        return self.metadata.get("original_num_gates", self.num_gates)
    
    @property
    def pruning_ratio(self) -> float:
        """Fraction of gates pruned by lightcone analysis."""
        return self.lightcone.pruning_ratio
    
    def summary(self) -> str:
        """Return a summary of the compiled circuit."""
        return (
            f"CompiledACES:\n"
            f"  Qubits: {self.num_qubits} ({len(self.measured_qubits)} measured)\n"
            f"  Gates: {self.num_gates}/{self.original_num_gates} "
            f"({self.pruning_ratio:.1%} pruned)\n"
            f"  CEG edges: {self.ceg.num_edges}\n"
            f"  Parameters: {self.parameter_names or 'none'}"
        )
    
    def __repr__(self) -> str:
        return (f"CompiledACES(qubits={self.num_qubits}, "
                f"gates={self.num_gates}, measured={len(self.measured_qubits)})")


def compile_circuit(
    circuit,
    measured_qubits: Optional[Set[int]] = None,
    entanglement_budget: Optional[float] = None,
    observables: Optional[List[str]] = None,
    noise_model = None
) -> CompiledACES:
    """
    Compile a quantum circuit into an ACES executable.
    
    This is the main entry point for ACES compilation.
    
    Args:
        circuit: A Qiskit QuantumCircuit or similar frontend circuit
        measured_qubits: Set of qubit indices to measure. If None,
                        inferred from circuit measurements or all qubits.
        entanglement_budget: Maximum entanglement to track (higher = more accurate
                           but slower). If None, estimated automatically.
        observables: Optional list of observable specifications (e.g., ["Z0", "Z0Z1"])
        noise_model: Optional noise model to incorporate
        
    Returns:
        CompiledACES object ready for runtime execution
    """
    # Parse the circuit
    parsed = parse_qiskit_circuit(circuit)
    
    # Determine measured qubits
    if measured_qubits is None:
        if parsed.measured_qubits:
            measured_qubits = parsed.measured_qubits
        else:
            # Default: measure all qubits
            measured_qubits = set(range(parsed.num_qubits))
    else:
        measured_qubits = set(measured_qubits)
    
    # Perform lightcone analysis
    lightcone = analyze_lightcone(parsed.dag, measured_qubits)
    
    # Build frozen gate sequence (only gates in lightcone)
    gate_sequence = []
    for idx, gate in enumerate(parsed.gates):
        if idx in lightcone.relevant_gates:
            try:
                gate_object = GateLibrary.get_gate(gate.name)
            except ValueError:
                # Unknown gate - create dummy
                gate_object = None
            
            frozen = FrozenGate(
                index=idx,
                name=gate.name,
                qubits=gate.qubits,
                params=gate.params,
                is_parametric=gate.is_parametric,
                param_names=gate.param_names,
                gate_object=gate_object
            )
            gate_sequence.append(frozen)
    
    # Build CEG
    ceg_builder = CEGBuilder(
        num_qubits=parsed.num_qubits,
        measured_qubits=measured_qubits,
        entanglement_budget=entanglement_budget or 100.0
    )
    ceg = ceg_builder.build_from_gates(
        lightcone.two_qubit_gate_pairs,
        lightcone.relevant_gates
    )
    
    # Configure budget
    if entanglement_budget is not None:
        budget = EntanglementBudget(max_weight=entanglement_budget)
    else:
        budget = estimate_budget_for_circuit(
            num_qubits=parsed.num_qubits,
            depth=parsed.depth,
            measured_fraction=len(measured_qubits) / parsed.num_qubits
        )
    
    # Collect parameter names
    parameter_names = set()
    for gate in gate_sequence:
        parameter_names.update(gate.param_names)
    
    return CompiledACES(
        num_qubits=parsed.num_qubits,
        measured_qubits=measured_qubits,
        gate_sequence=gate_sequence,
        ceg=ceg,
        lightcone=lightcone,
        budget=budget,
        parameter_names=parameter_names,
        metadata={
            "depth": parsed.depth,
            "original_num_gates": len(parsed.gates),
            "num_clbits": parsed.num_clbits,
        }
    )


def compile_from_gate_list(
    gates: List[Dict[str, Any]],
    num_qubits: int,
    measured_qubits: Set[int],
    entanglement_budget: float = 100.0
) -> CompiledACES:
    """
    Compile from a simple gate list format (no Qiskit required).
    
    Args:
        gates: List of gate dictionaries with "name", "qubits", "params"
        num_qubits: Total number of qubits
        measured_qubits: Set of measured qubit indices
        entanglement_budget: Entanglement budget
        
    Returns:
        CompiledACES object
    """
    from aces.compiler.parser import parse_gate_sequence
    
    parsed = parse_gate_sequence(gates, num_qubits)
    parsed.measurements = {q: q for q in measured_qubits}
    
    # Lightcone analysis
    lightcone = analyze_lightcone(parsed.dag, measured_qubits)
    
    # Build gate sequence
    gate_sequence = []
    for idx, gate in enumerate(parsed.gates):
        if idx in lightcone.relevant_gates:
            try:
                gate_object = GateLibrary.get_gate(gate.name)
            except ValueError:
                gate_object = None
            
            frozen = FrozenGate(
                index=idx,
                name=gate.name,
                qubits=gate.qubits,
                params=gate.params,
                is_parametric=gate.is_parametric,
                param_names=gate.param_names,
                gate_object=gate_object
            )
            gate_sequence.append(frozen)
    
    # Build CEG
    ceg_builder = CEGBuilder(
        num_qubits=num_qubits,
        measured_qubits=measured_qubits,
        entanglement_budget=entanglement_budget
    )
    ceg = ceg_builder.build_from_gates(
        lightcone.two_qubit_gate_pairs,
        lightcone.relevant_gates
    )
    
    budget = EntanglementBudget(max_weight=entanglement_budget)
    
    return CompiledACES(
        num_qubits=num_qubits,
        measured_qubits=measured_qubits,
        gate_sequence=gate_sequence,
        ceg=ceg,
        lightcone=lightcone,
        budget=budget,
        metadata={
            "depth": parsed.depth,
            "original_num_gates": len(parsed.gates),
        }
    )
