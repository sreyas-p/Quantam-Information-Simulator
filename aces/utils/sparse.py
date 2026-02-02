"""
Sparse data structures for ACES.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterator, Optional
import numpy as np
from scipy import sparse


@dataclass
class SparseCorrelationMatrix:
    """
    Sparse matrix storing pairwise correlations.
    
    More memory-efficient than storing full TwoQubitRDM objects
    when only correlations (not full density matrices) are needed.
    
    Stores correlation tensor values T[i,j] for each qubit pair.
    """
    num_qubits: int
    _data: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    
    def set_correlation(
        self,
        qubit_a: int,
        qubit_b: int,
        correlation_tensor: np.ndarray
    ) -> None:
        """Set correlation tensor for a qubit pair."""
        key = (min(qubit_a, qubit_b), max(qubit_a, qubit_b))
        self._data[key] = correlation_tensor.copy()
    
    def get_correlation(
        self,
        qubit_a: int,
        qubit_b: int
    ) -> Optional[np.ndarray]:
        """Get correlation tensor for a qubit pair."""
        key = (min(qubit_a, qubit_b), max(qubit_a, qubit_b))
        return self._data.get(key)
    
    def has_correlation(self, qubit_a: int, qubit_b: int) -> bool:
        """Check if correlation exists for pair."""
        key = (min(qubit_a, qubit_b), max(qubit_a, qubit_b))
        return key in self._data
    
    def remove_correlation(self, qubit_a: int, qubit_b: int) -> bool:
        """Remove correlation for pair. Returns True if existed."""
        key = (min(qubit_a, qubit_b), max(qubit_a, qubit_b))
        if key in self._data:
            del self._data[key]
            return True
        return False
    
    @property
    def num_correlations(self) -> int:
        """Number of stored correlations."""
        return len(self._data)
    
    def iter_correlations(self) -> Iterator[Tuple[int, int, np.ndarray]]:
        """Iterate over (qubit_a, qubit_b, correlation_tensor)."""
        for (qa, qb), tensor in self._data.items():
            yield qa, qb, tensor
    
    def to_scipy_sparse(self, pauli_a: int = 3, pauli_b: int = 3) -> sparse.csr_matrix:
        """
        Convert to scipy sparse matrix for specific Pauli indices.
        
        Args:
            pauli_a: Pauli index for first qubit (1=X, 2=Y, 3=Z)
            pauli_b: Pauli index for second qubit
            
        Returns:
            Sparse matrix where [i,j] = T^{ij}_{pauli_a, pauli_b}
        """
        rows = []
        cols = []
        data = []
        
        for (qa, qb), tensor in self._data.items():
            rows.append(qa)
            cols.append(qb)
            data.append(tensor[pauli_a, pauli_b])
            
            # Symmetric
            rows.append(qb)
            cols.append(qa)
            data.append(tensor[pauli_a, pauli_b])
        
        return sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(self.num_qubits, self.num_qubits)
        )
    
    def zz_correlation_matrix(self) -> sparse.csr_matrix:
        """Get sparse matrix of ZZ correlations."""
        return self.to_scipy_sparse(pauli_a=3, pauli_b=3)
    
    def memory_usage_bytes(self) -> int:
        """Estimate memory usage in bytes."""
        # Each correlation tensor is 4x4 float64 = 128 bytes
        # Plus key overhead ~24 bytes
        return self.num_correlations * (128 + 24)


@dataclass
class SparseMarginals:
    """
    Sparse storage for single-qubit marginal information.
    
    Stores Bloch vectors for qubits that have non-trivial states.
    """
    num_qubits: int
    _bloch_vectors: Dict[int, np.ndarray] = field(default_factory=dict)
    
    def set_bloch(self, qubit: int, bloch: np.ndarray) -> None:
        """Set Bloch vector for a qubit."""
        if np.allclose(bloch, [0, 0, 1]):
            # |0⟩ state - don't store (implicit default)
            self._bloch_vectors.pop(qubit, None)
        else:
            self._bloch_vectors[qubit] = bloch.copy()
    
    def get_bloch(self, qubit: int) -> np.ndarray:
        """Get Bloch vector (default: |0⟩ state)."""
        return self._bloch_vectors.get(qubit, np.array([0.0, 0.0, 1.0]))
    
    @property
    def num_nontrivial(self) -> int:
        """Number of qubits with non-trivial states."""
        return len(self._bloch_vectors)
    
    def z_expectations(self) -> np.ndarray:
        """Get Z expectations for all qubits."""
        result = np.ones(self.num_qubits)  # Default |0⟩ has ⟨Z⟩ = 1
        for q, bloch in self._bloch_vectors.items():
            result[q] = bloch[2]
        return result
