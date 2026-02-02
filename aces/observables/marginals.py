"""
Marginal distribution extraction for ACES.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Set
import numpy as np

from aces.core.density_matrix import SingleQubitRDM, TwoQubitRDM


def extract_marginals(
    qubits: List[int],
    state: Dict[int, SingleQubitRDM],
    correlators: Dict[Tuple[int, int], TwoQubitRDM]
) -> Dict[int, np.ndarray]:
    """
    Extract marginal probability distributions for specified qubits.
    
    Args:
        qubits: List of qubit indices
        state: Single-qubit RDMs
        correlators: Two-qubit RDMs
        
    Returns:
        Dictionary mapping qubit -> [p(0), p(1)] probability array
    """
    marginals = {}
    
    for q in qubits:
        if q in state:
            marginals[q] = np.array([
                state[q].probability_zero(),
                state[q].probability_one()
            ])
        else:
            # Unknown qubit - assume maximally mixed
            marginals[q] = np.array([0.5, 0.5])
    
    return marginals


def extract_joint_marginal(
    qubit_a: int,
    qubit_b: int,
    state: Dict[int, SingleQubitRDM],
    correlators: Dict[Tuple[int, int], TwoQubitRDM]
) -> np.ndarray:
    """
    Extract joint probability distribution P(q_a, q_b).
    
    Args:
        qubit_a: First qubit index
        qubit_b: Second qubit index
        state: Single-qubit RDMs
        correlators: Two-qubit RDMs
        
    Returns:
        2x2 array where [i,j] = P(q_a=i, q_b=j)
    """
    key = (min(qubit_a, qubit_b), max(qubit_a, qubit_b))
    
    if key in correlators:
        rho = correlators[key].rho
        
        # Extract diagonal (computational basis probabilities)
        joint = np.zeros((2, 2))
        
        if qubit_a < qubit_b:
            # Standard ordering: |00⟩, |01⟩, |10⟩, |11⟩
            joint[0, 0] = np.abs(rho[0, 0])  # |00⟩
            joint[0, 1] = np.abs(rho[1, 1])  # |01⟩
            joint[1, 0] = np.abs(rho[2, 2])  # |10⟩
            joint[1, 1] = np.abs(rho[3, 3])  # |11⟩
        else:
            # Swap ordering
            joint[0, 0] = np.abs(rho[0, 0])  # |00⟩
            joint[1, 0] = np.abs(rho[1, 1])  # |01⟩ -> |10⟩ in swapped
            joint[0, 1] = np.abs(rho[2, 2])  # |10⟩ -> |01⟩ in swapped
            joint[1, 1] = np.abs(rho[3, 3])  # |11⟩
        
        return joint
    else:
        # No correlator - use product of marginals
        p_a = extract_marginals([qubit_a], state, correlators)[qubit_a]
        p_b = extract_marginals([qubit_b], state, correlators)[qubit_b]
        
        return np.outer(p_a, p_b)


def extract_bitstring_distribution(
    qubits: List[int],
    state: Dict[int, SingleQubitRDM],
    correlators: Dict[Tuple[int, int], TwoQubitRDM]
) -> Dict[str, float]:
    """
    Extract full probability distribution over bitstrings.
    
    Note: This is only tractable for small numbers of qubits.
    For larger systems, use sampling instead.
    
    Args:
        qubits: List of qubit indices (ordered)
        state: Single-qubit RDMs
        correlators: Two-qubit RDMs
        
    Returns:
        Dictionary mapping bitstring -> probability
    """
    n = len(qubits)
    
    if n > 20:
        raise ValueError(f"Cannot compute full distribution for {n} qubits. "
                        "Use sampling instead.")
    
    # For small n, enumerate all bitstrings
    distribution = {}
    
    for i in range(2 ** n):
        bitstring = format(i, f'0{n}b')
        
        # Compute probability using marginals and correlations
        # This is an approximation - true distribution would need n-qubit RDM
        prob = _approximate_bitstring_probability(
            bitstring, qubits, state, correlators
        )
        distribution[bitstring] = prob
    
    # Normalize
    total = sum(distribution.values())
    if total > 0:
        distribution = {k: v / total for k, v in distribution.items()}
    
    return distribution


def _approximate_bitstring_probability(
    bitstring: str,
    qubits: List[int],
    state: Dict[int, SingleQubitRDM],
    correlators: Dict[Tuple[int, int], TwoQubitRDM]
) -> float:
    """
    Approximate probability of a bitstring using available RDMs.
    
    Uses product of single-qubit probabilities, adjusted by correlations.
    """
    # Start with product of marginals
    prob = 1.0
    used_in_correlator = set()
    
    # Use correlators for adjacent pairs when available
    for i in range(len(qubits) - 1):
        qa, qb = qubits[i], qubits[i + 1]
        key = (min(qa, qb), max(qa, qb))
        
        if key in correlators and i not in used_in_correlator and (i + 1) not in used_in_correlator:
            joint = extract_joint_marginal(qa, qb, state, correlators)
            
            bit_a = int(bitstring[i])
            bit_b = int(bitstring[i + 1])
            
            prob *= joint[bit_a, bit_b]
            used_in_correlator.add(i)
            used_in_correlator.add(i + 1)
    
    # Use single-qubit marginals for uncorrelated qubits
    for i, q in enumerate(qubits):
        if i not in used_in_correlator:
            if q in state:
                bit = int(bitstring[i])
                if bit == 0:
                    prob *= state[q].probability_zero()
                else:
                    prob *= state[q].probability_one()
            else:
                prob *= 0.5
    
    return prob
