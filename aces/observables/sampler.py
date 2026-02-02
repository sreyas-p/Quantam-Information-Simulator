"""
Bitstring sampling for ACES.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import Counter

from aces.core.density_matrix import SingleQubitRDM, TwoQubitRDM
from aces.observables.marginals import extract_joint_marginal


def sample_bitstrings(
    num_samples: int,
    qubits: List[int],
    state: Dict[int, SingleQubitRDM],
    correlators: Dict[Tuple[int, int], TwoQubitRDM],
    seed: Optional[int] = None
) -> List[str]:
    """
    Sample bitstrings from the quantum state.
    
    Uses marginal distributions for sampling, with correlations
    incorporated where available.
    
    Args:
        num_samples: Number of bitstrings to sample
        qubits: List of qubit indices in order
        state: Single-qubit RDMs
        correlators: Two-qubit RDMs
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled bitstrings
    """
    if seed is not None:
        np.random.seed(seed)
    
    samples = []
    
    for _ in range(num_samples):
        bitstring = _sample_single(qubits, state, correlators)
        samples.append(bitstring)
    
    return samples


def _sample_single(
    qubits: List[int],
    state: Dict[int, SingleQubitRDM],
    correlators: Dict[Tuple[int, int], TwoQubitRDM]
) -> str:
    """Sample a single bitstring."""
    bits = []
    
    for i, q in enumerate(qubits):
        if q in state:
            # Check if we can use a correlator with previous qubit
            if i > 0:
                prev_q = qubits[i - 1]
                key = (min(prev_q, q), max(prev_q, q))
                
                if key in correlators and len(bits) > 0:
                    # Conditional probability given previous bit
                    joint = extract_joint_marginal(prev_q, q, state, correlators)
                    prev_bit = bits[-1]
                    
                    # P(q=1 | prev=prev_bit)
                    if prev_q < q:
                        marginal = joint[prev_bit, :]
                    else:
                        marginal = joint[:, prev_bit]
                    
                    # Normalize
                    if marginal.sum() > 0:
                        marginal = marginal / marginal.sum()
                    else:
                        marginal = np.array([0.5, 0.5])
                    
                    bit = 0 if np.random.random() < marginal[0] else 1
                    bits.append(bit)
                    continue
            
            # Independent sampling from marginal
            p0 = state[q].probability_zero()
            bit = 0 if np.random.random() < p0 else 1
        else:
            # Unknown qubit - random
            bit = np.random.randint(2)
        
        bits.append(bit)
    
    return ''.join(map(str, bits))


def sample_with_postselection(
    num_samples: int,
    qubits: List[int],
    state: Dict[int, SingleQubitRDM],
    correlators: Dict[Tuple[int, int], TwoQubitRDM],
    condition: Dict[int, int],
    max_attempts: int = 100000,
    seed: Optional[int] = None
) -> List[str]:
    """
    Sample bitstrings with postselection on specific qubits.
    
    Args:
        num_samples: Number of bitstrings to sample
        qubits: List of qubit indices
        state: Single-qubit RDMs
        correlators: Two-qubit RDMs
        condition: Dict of qubit_index -> required_value for postselection
        max_attempts: Maximum sampling attempts
        seed: Random seed
        
    Returns:
        List of sampled bitstrings satisfying condition
    """
    if seed is not None:
        np.random.seed(seed)
    
    samples = []
    attempts = 0
    
    condition_indices = {qubits.index(q): v for q, v in condition.items() if q in qubits}
    
    while len(samples) < num_samples and attempts < max_attempts:
        bitstring = _sample_single(qubits, state, correlators)
        attempts += 1
        
        # Check condition
        passes = True
        for idx, required_val in condition_indices.items():
            if int(bitstring[idx]) != required_val:
                passes = False
                break
        
        if passes:
            samples.append(bitstring)
    
    return samples


def estimate_counts(
    num_samples: int,
    qubits: List[int],
    state: Dict[int, SingleQubitRDM],
    correlators: Dict[Tuple[int, int], TwoQubitRDM],
    seed: Optional[int] = None
) -> Dict[str, int]:
    """
    Sample and return counts (like Qiskit's counts format).
    
    Returns:
        Dictionary mapping bitstring -> count
    """
    samples = sample_bitstrings(num_samples, qubits, state, correlators, seed)
    return dict(Counter(samples))


def shots_to_expectations(
    counts: Dict[str, int],
    observables: List[str]
) -> Dict[str, float]:
    """
    Estimate expectation values from measurement counts.
    
    Args:
        counts: Bitstring -> count dictionary
        observables: List of Z-basis observables like "Z0", "Z0Z1"
        
    Returns:
        Dictionary of observable -> expectation value
    """
    import re
    
    total = sum(counts.values())
    expectations = {}
    
    for obs in observables:
        # Parse Z indices
        pattern = r'Z(\d+)'
        matches = re.findall(pattern, obs.upper())
        z_indices = [int(m) for m in matches]
        
        if not z_indices:
            expectations[obs] = 1.0
            continue
        
        # Compute expectation
        exp_sum = 0
        for bitstring, count in counts.items():
            # Parity of bits at z_indices
            parity = 1
            for idx in z_indices:
                if idx < len(bitstring):
                    parity *= (1 - 2 * int(bitstring[idx]))
            exp_sum += count * parity
        
        expectations[obs] = exp_sum / total
    
    return expectations
