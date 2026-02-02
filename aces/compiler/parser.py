"""
Circuit parser for ACES.

Parses Qiskit circuits into the internal representation used by ACES.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np

from aces.compiler.lightcone import CircuitDAG, GateNode


# Gate name mapping from Qiskit to ACES
QISKIT_GATE_MAP = {
    "x": "X",
    "y": "Y", 
    "z": "Z",
    "h": "H",
    "s": "S",
    "sdg": "Sdg",
    "t": "T",
    "tdg": "Tdg",
    "sx": "SX",
    "rx": "Rx",
    "ry": "Ry",
    "rz": "Rz",
    "u": "U3",
    "u3": "U3",
    "u2": "U2",
    "u1": "Rz",  # U1 is equivalent to Rz up to global phase
    "cx": "CX",
    "cnot": "CX",
    "cz": "CZ",
    "swap": "SWAP",
    "iswap": "iSWAP",
    "rxx": "Rxx",
    "ryy": "Ryy",
    "rzz": "Rzz",
    "cp": "CP",
    "crz": "CRz",
    "id": "I",
    "barrier": "BARRIER",
    "measure": "MEASURE",
}


@dataclass
class ParsedGate:
    """
    Represents a parsed gate from a quantum circuit.
    
    Attributes:
        name: Standardized gate name
        qubits: Tuple of qubit indices
        params: Dictionary of parameters (e.g., {"theta": 0.5})
        original_name: Original gate name from the frontend
        is_parametric: Whether this gate has variable parameters
        param_names: Names of the parameters for lookup
    """
    name: str
    qubits: Tuple[int, ...]
    params: Dict[str, float] = field(default_factory=dict)
    original_name: str = ""
    is_parametric: bool = False
    param_names: Tuple[str, ...] = field(default_factory=tuple)
    
    @property
    def num_qubits(self) -> int:
        return len(self.qubits)
    
    def __repr__(self) -> str:
        if self.params:
            param_str = ", ".join(f"{k}={v:.4f}" for k, v in self.params.items())
            return f"{self.name}({param_str}) @ {self.qubits}"
        return f"{self.name} @ {self.qubits}"


@dataclass
class ParsedCircuit:
    """
    Parsed circuit representation.
    
    Attributes:
        num_qubits: Number of qubits
        num_clbits: Number of classical bits
        gates: List of parsed gates in order
        measurements: Mapping of qubit -> classical bit for measurements
        dag: CircuitDAG representation
        parameter_map: Mapping of parameter name -> gate indices that use it
    """
    num_qubits: int
    num_clbits: int = 0
    gates: List[ParsedGate] = field(default_factory=list)
    measurements: Dict[int, int] = field(default_factory=dict)
    dag: Optional[CircuitDAG] = None
    parameter_map: Dict[str, List[int]] = field(default_factory=lambda: {})
    
    @property
    def measured_qubits(self) -> Set[int]:
        """Get set of measured qubit indices."""
        return set(self.measurements.keys())
    
    @property
    def depth(self) -> int:
        """Circuit depth."""
        if self.dag:
            return self.dag.depth
        return len(self.gates)  # Conservative estimate
    
    def __repr__(self) -> str:
        return (f"ParsedCircuit(qubits={self.num_qubits}, "
                f"gates={len(self.gates)}, depth={self.depth})")


def parse_qiskit_circuit(circuit) -> ParsedCircuit:
    """
    Parse a Qiskit QuantumCircuit into ACES internal representation.
    
    Args:
        circuit: A qiskit.QuantumCircuit object
        
    Returns:
        ParsedCircuit with all gates and measurements extracted
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter, ParameterExpression
    except ImportError:
        raise ImportError("Qiskit is required for circuit parsing. "
                         "Install with: pip install qiskit")
    
    num_qubits = circuit.num_qubits
    num_clbits = circuit.num_clbits
    
    gates: List[ParsedGate] = []
    measurements: Dict[int, int] = {}
    dag = CircuitDAG()
    dag.num_qubits = num_qubits
    parameter_map: Dict[str, List[int]] = {}
    
    for instruction in circuit.data:
        op = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits
        
        # Get qubit indices
        qubit_indices = tuple(circuit.find_bit(q).index for q in qargs)
        
        # Get gate name
        original_name = op.name.lower()
        gate_name = QISKIT_GATE_MAP.get(original_name, op.name.upper())
        
        # Handle measurements
        if gate_name == "MEASURE":
            if cargs:
                clbit_idx = circuit.find_bit(cargs[0]).index
                measurements[qubit_indices[0]] = clbit_idx
            continue
        
        # Skip barriers
        if gate_name == "BARRIER":
            continue
        
        # Extract parameters
        params: Dict[str, float] = {}
        is_parametric = False
        param_names: List[str] = []
        
        if hasattr(op, 'params') and op.params:
            # Map parameters to standard names
            if gate_name in ("Rx", "Ry", "Rz", "Rxx", "Ryy", "Rzz"):
                param_labels = ["theta"]
            elif gate_name == "U3":
                param_labels = ["theta", "phi", "lambda"]
            elif gate_name == "U2":
                param_labels = ["phi", "lambda"]
            elif gate_name in ("CP", "CRz"):
                param_labels = ["theta"]
            else:
                param_labels = [f"p{i}" for i in range(len(op.params))]
            
            for i, param in enumerate(op.params):
                if i >= len(param_labels):
                    break
                    
                label = param_labels[i]
                
                # Check if parameter is symbolic
                if isinstance(param, (Parameter, ParameterExpression)):
                    is_parametric = True
                    param_name = str(param)
                    param_names.append(param_name)
                    params[label] = 0.0  # Placeholder
                    
                    # Track which gates use this parameter
                    if param_name not in parameter_map:
                        parameter_map[param_name] = []
                    parameter_map[param_name].append(len(gates))
                else:
                    # Concrete value
                    params[label] = float(param)
        
        parsed_gate = ParsedGate(
            name=gate_name,
            qubits=qubit_indices,
            params=params,
            original_name=original_name,
            is_parametric=is_parametric,
            param_names=tuple(param_names)
        )
        gates.append(parsed_gate)
        
        # Add to DAG
        dag.add_gate(gate_name, qubit_indices, params)
    
    return ParsedCircuit(
        num_qubits=num_qubits,
        num_clbits=num_clbits,
        gates=gates,
        measurements=measurements,
        dag=dag,
        parameter_map=parameter_map
    )


def parse_gate_sequence(
    gate_list: List[Dict[str, Any]],
    num_qubits: int
) -> ParsedCircuit:
    """
    Parse a simple gate sequence format.
    
    Args:
        gate_list: List of gate dictionaries with keys:
            - "name": Gate name
            - "qubits": List of qubit indices
            - "params": Optional dict of parameters
        num_qubits: Number of qubits
        
    Returns:
        ParsedCircuit
    """
    gates: List[ParsedGate] = []
    dag = CircuitDAG()
    dag.num_qubits = num_qubits
    
    for gate_dict in gate_list:
        name = gate_dict["name"]
        qubits = tuple(gate_dict["qubits"])
        params = gate_dict.get("params", {})
        
        parsed_gate = ParsedGate(
            name=name,
            qubits=qubits,
            params=params,
            original_name=name.lower()
        )
        gates.append(parsed_gate)
        dag.add_gate(name, qubits, params)
    
    return ParsedCircuit(
        num_qubits=num_qubits,
        gates=gates,
        dag=dag
    )


def circuit_to_gate_list(parsed: ParsedCircuit) -> List[Dict[str, Any]]:
    """
    Convert ParsedCircuit back to a simple gate list format.
    
    Useful for serialization.
    """
    return [
        {
            "name": gate.name,
            "qubits": list(gate.qubits),
            "params": gate.params
        }
        for gate in parsed.gates
    ]
