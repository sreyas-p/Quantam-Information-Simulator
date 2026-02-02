"""
Expectation value computation for ACES.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import re
import numpy as np

from aces.core.density_matrix import SingleQubitRDM, TwoQubitRDM


def compute_expectation(
    observable: str,
    state: Dict[int, SingleQubitRDM],
    correlators: Dict[Tuple[int, int], TwoQubitRDM]
) -> float:
    """
    Compute expectation value of a Pauli observable.
    
    Args:
        observable: Observable string like "Z0", "Z0Z1", "X0Y2Z5"
        state: Single-qubit RDM dictionary
        correlators: Two-qubit RDM dictionary
        
    Returns:
        Expectation value
    """
    # Parse observable
    terms = parse_pauli_string(observable)
    
    if not terms:
        return 1.0  # Identity
    
    if len(terms) == 1:
        pauli, qubit = terms[0]
        if qubit in state:
            return state[qubit].expectation(pauli)
        return 0.0
    
    if len(terms) == 2:
        return _two_qubit_expectation(terms, state, correlators)
    
    # Multi-qubit: factorize into 2-qubit and 1-qubit terms
    return _multi_qubit_expectation(terms, state, correlators)


def parse_pauli_string(observable: str) -> List[Tuple[str, int]]:
    """
    Parse a Pauli string into (pauli, qubit) tuples.
    
    Examples:
        "Z0" -> [("Z", 0)]
        "Z0Z1" -> [("Z", 0), ("Z", 1)]
        "X0Y2Z5" -> [("X", 0), ("Y", 2), ("Z", 5)]
    """
    pattern = r'([IXYZ])(\d+)'
    matches = re.findall(pattern, observable.upper())
    return [(pauli, int(qubit)) for pauli, qubit in matches if pauli != 'I']


def _two_qubit_expectation(
    terms: List[Tuple[str, int]],
    state: Dict[int, SingleQubitRDM],
    correlators: Dict[Tuple[int, int], TwoQubitRDM]
) -> float:
    """Compute two-qubit expectation using correlator if available."""
    (p1, q1), (p2, q2) = terms
    
    key = (min(q1, q2), max(q1, q2))
    
    if key in correlators:
        T = correlators[key].correlation_tensor
        pauli_idx = {'X': 1, 'Y': 2, 'Z': 3}
        
        if q1 < q2:
            return T[pauli_idx[p1], pauli_idx[p2]]
        else:
            return T[pauli_idx[p2], pauli_idx[p1]]
    else:
        # No correlator - use product approximation
        exp1 = state[q1].expectation(p1) if q1 in state else 0.0
        exp2 = state[q2].expectation(p2) if q2 in state else 0.0
        return exp1 * exp2


def _multi_qubit_expectation(
    terms: List[Tuple[str, int]],
    state: Dict[int, SingleQubitRDM],
    correlators: Dict[Tuple[int, int], TwoQubitRDM]
) -> float:
    """
    Compute multi-qubit expectation using best available approximation.
    
    Strategy: pair up qubits that have correlators, treat rest as products.
    """
    # Sort by qubit index
    terms = sorted(terms, key=lambda x: x[1])
    
    result = 1.0
    used = set()
    
    # Try to use correlators for adjacent pairs
    for i in range(len(terms) - 1):
        if i in used:
            continue
            
        p1, q1 = terms[i]
        p2, q2 = terms[i + 1]
        
        key = (min(q1, q2), max(q1, q2))
        
        if key in correlators and (i + 1) not in used:
            # Use correlator
            T = correlators[key].correlation_tensor
            pauli_idx = {'X': 1, 'Y': 2, 'Z': 3}
            
            if q1 < q2:
                result *= T[pauli_idx[p1], pauli_idx[p2]]
            else:
                result *= T[pauli_idx[p2], pauli_idx[p1]]
            
            used.add(i)
            used.add(i + 1)
    
    # Handle remaining single-qubit terms
    for i, (pauli, qubit) in enumerate(terms):
        if i not in used:
            if qubit in state:
                result *= state[qubit].expectation(pauli)
            else:
                return 0.0  # Unknown qubit
    
    return result


def compute_hamiltonian_expectation(
    hamiltonian: List[Tuple[float, str]],
    state: Dict[int, SingleQubitRDM],
    correlators: Dict[Tuple[int, int], TwoQubitRDM]
) -> float:
    """
    Compute expectation value of a Hamiltonian.
    
    Args:
        hamiltonian: List of (coefficient, pauli_string) tuples
        state: Single-qubit RDMs
        correlators: Two-qubit RDMs
        
    Returns:
        Total Hamiltonian expectation
    """
    total = 0.0
    
    for coeff, pauli_string in hamiltonian:
        total += coeff * compute_expectation(pauli_string, state, correlators)
    
    return total


def compute_all_z_expectations(
    qubits: List[int],
    state: Dict[int, SingleQubitRDM]
) -> Dict[int, float]:
    """Compute Z expectation for all specified qubits."""
    return {q: state[q].expectation('Z') if q in state else 0.0 for q in qubits}
