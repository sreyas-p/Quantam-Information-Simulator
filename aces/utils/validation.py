"""
Validation utilities for ACES.

Compares ACES outputs against exact simulators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class ValidationResult:
    """Result of validating ACES against exact simulation."""
    observables: List[str]
    aces_values: Dict[str, float]
    exact_values: Dict[str, float]
    errors: Dict[str, float]
    max_error: float
    mean_error: float
    passed: bool
    threshold: float
    details: Dict[str, Any]


def validate_against_exact(
    circuit,
    measured_qubits: Optional[List[int]] = None,
    observables: Optional[List[str]] = None,
    params: Optional[Dict[str, float]] = None,
    threshold: float = 0.01,
    verbose: bool = False
) -> ValidationResult:
    """
    Validate ACES against exact Qiskit simulation.
    
    Args:
        circuit: Qiskit QuantumCircuit
        measured_qubits: Qubits to measure (default: all)
        observables: Observables to check (default: Z on measured qubits)
        params: Parameter values for parametric circuits
        threshold: Maximum allowed error
        verbose: Print detailed output
        
    Returns:
        ValidationResult with comparison data
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        from qiskit.quantum_info import Statevector
    except ImportError:
        raise ImportError("Qiskit and qiskit-aer required for validation")
    
    from aces import compile_circuit, ACESRuntime
    
    # Bind parameters if provided
    if params:
        bound_circuit = circuit.assign_parameters(params)
    else:
        bound_circuit = circuit
        params = {}
    
    # Determine measured qubits
    if measured_qubits is None:
        measured_qubits = list(range(circuit.num_qubits))
    measured_set = set(measured_qubits)
    
    # Determine observables
    if observables is None:
        observables = [f"Z{q}" for q in measured_qubits]
    
    # Run ACES
    compiled = compile_circuit(circuit, measured_qubits=measured_set)
    runtime = ACESRuntime(compiled)
    aces_result = runtime.execute(params=params, observables=observables)
    aces_values = aces_result.expectation_values
    
    # Run exact simulation
    exact_values = _exact_expectations(bound_circuit, observables)
    
    # Compare
    errors = {}
    for obs in observables:
        aces_val = aces_values.get(obs, 0.0)
        exact_val = exact_values.get(obs, 0.0)
        errors[obs] = abs(aces_val - exact_val)
    
    max_error = max(errors.values()) if errors else 0.0
    mean_error = np.mean(list(errors.values())) if errors else 0.0
    passed = max_error <= threshold
    
    if verbose:
        print(f"\n{'Observable':<15} {'ACES':<12} {'Exact':<12} {'Error':<12}")
        print("-" * 51)
        for obs in observables:
            print(f"{obs:<15} {aces_values.get(obs, 0):<12.6f} "
                  f"{exact_values.get(obs, 0):<12.6f} {errors.get(obs, 0):<12.6f}")
        print("-" * 51)
        print(f"Max error: {max_error:.6f}, Mean error: {mean_error:.6f}")
        print(f"Result: {'PASSED' if passed else 'FAILED'}")
    
    return ValidationResult(
        observables=observables,
        aces_values=aces_values,
        exact_values=exact_values,
        errors=errors,
        max_error=max_error,
        mean_error=mean_error,
        passed=passed,
        threshold=threshold,
        details={
            "num_qubits": circuit.num_qubits,
            "circuit_depth": circuit.depth(),
            "aces_gates": len(compiled.gate_sequence),
            "pruning_ratio": compiled.pruning_ratio,
        }
    )


def _exact_expectations(circuit, observables: List[str]) -> Dict[str, float]:
    """Compute exact expectation values using statevector simulation."""
    from qiskit.quantum_info import Statevector, Operator, Pauli
    import re
    
    sv = Statevector.from_instruction(circuit)
    rho = np.outer(sv.data, sv.data.conj())
    
    expectations = {}
    
    for obs in observables:
        # Parse Pauli string
        pattern = r'([IXYZ])(\d+)'
        matches = re.findall(pattern, obs.upper())
        
        if not matches:
            expectations[obs] = 1.0
            continue
        
        # Build Pauli string for Qiskit
        n_qubits = circuit.num_qubits
        pauli_str = ['I'] * n_qubits
        
        for pauli, qubit_str in matches:
            qubit = int(qubit_str)
            if qubit < n_qubits:
                pauli_str[n_qubits - 1 - qubit] = pauli  # Qiskit uses reversed order
        
        pauli = Pauli(''.join(pauli_str))
        op = Operator(pauli)
        
        # ⟨O⟩ = Tr(ρO)
        exp_val = np.real(np.trace(rho @ op.data))
        expectations[obs] = float(exp_val)
    
    return expectations


def quick_validation(
    num_qubits: int = 5,
    depth: int = 10,
    seed: int = 42
) -> ValidationResult:
    """
    Quick validation with a random circuit.
    
    Useful for sanity checking ACES installation.
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import EfficientSU2
    except ImportError:
        raise ImportError("Qiskit required for validation")
    
    np.random.seed(seed)
    
    # Create random-ish circuit
    qc = QuantumCircuit(num_qubits)
    
    for _ in range(depth):
        # Random single-qubit gates
        for q in range(num_qubits):
            gate = np.random.choice(['h', 'rx', 'ry', 'rz'])
            if gate == 'h':
                qc.h(q)
            else:
                angle = np.random.uniform(0, 2 * np.pi)
                getattr(qc, gate)(angle, q)
        
        # Random CNOTs
        for q in range(num_qubits - 1):
            if np.random.random() < 0.5:
                qc.cx(q, q + 1)
    
    return validate_against_exact(
        qc,
        measured_qubits=[0, 1, 2] if num_qubits >= 3 else [0],
        verbose=True
    )
