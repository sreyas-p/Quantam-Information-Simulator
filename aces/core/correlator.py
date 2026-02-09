"""Correlator storage in Pauli basis representation."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


PAULI_LABELS = ('I', 'X', 'Y', 'Z')
PAULI_INDEX = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}


@dataclass
class CorrelatorStorage:
    """
    2-qubit correlators in Pauli basis: ρ = (1/4) Σᵢⱼ Tᵢⱼ (σᵢ ⊗ σⱼ).
    
    Stores 15 independent coefficients (T₀₀=1 by normalization).
    """
    qubit_a: int
    qubit_b: int
    local_a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    local_b: np.ndarray = field(default_factory=lambda: np.zeros(3))
    correlation: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    
    def __post_init__(self):
        self.local_a = np.asarray(self.local_a, dtype=np.float64)
        self.local_b = np.asarray(self.local_b, dtype=np.float64)
        self.correlation = np.asarray(self.correlation, dtype=np.float64)
        
        if self.local_a.shape != (3,):
            raise ValueError(f"local_a must have shape (3,), got {self.local_a.shape}")
        if self.local_b.shape != (3,):
            raise ValueError(f"local_b must have shape (3,), got {self.local_b.shape}")
        if self.correlation.shape != (3, 3):
            raise ValueError(f"correlation must have shape (3,3), got {self.correlation.shape}")
    
    @classmethod
    def from_full_tensor(cls, qubit_a: int, qubit_b: int, T: np.ndarray) -> CorrelatorStorage:
        if T.shape != (4, 4):
            raise ValueError(f"Tensor must be 4x4, got {T.shape}")
        return cls(
            qubit_a=qubit_a, qubit_b=qubit_b,
            local_a=T[1:4, 0].copy(),
            local_b=T[0, 1:4].copy(),
            correlation=T[1:4, 1:4].copy()
        )
    
    @classmethod
    def from_product_state(cls, qubit_a: int, qubit_b: int, 
                           bloch_a: np.ndarray, bloch_b: np.ndarray) -> CorrelatorStorage:
        bloch_a = np.asarray(bloch_a)
        bloch_b = np.asarray(bloch_b)
        return cls(
            qubit_a=qubit_a, qubit_b=qubit_b,
            local_a=bloch_a.copy(), local_b=bloch_b.copy(),
            correlation=np.outer(bloch_a, bloch_b)
        )
    
    @classmethod
    def zero_state(cls, qubit_a: int, qubit_b: int) -> CorrelatorStorage:
        return cls.from_product_state(qubit_a, qubit_b,
                                       np.array([0., 0., 1.]), np.array([0., 0., 1.]))
    
    @classmethod
    def bell_phi_plus(cls, qubit_a: int, qubit_b: int) -> CorrelatorStorage:
        return cls(
            qubit_a=qubit_a, qubit_b=qubit_b,
            local_a=np.zeros(3), local_b=np.zeros(3),
            correlation=np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
        )
    
    def to_full_tensor(self) -> np.ndarray:
        T = np.zeros((4, 4))
        T[0, 0] = 1.0
        T[1:4, 0] = self.local_a
        T[0, 1:4] = self.local_b
        T[1:4, 1:4] = self.correlation
        return T
    
    def expectation(self, pauli_a: str, pauli_b: str) -> float:
        i, j = PAULI_INDEX[pauli_a.upper()], PAULI_INDEX[pauli_b.upper()]
        if i == 0 and j == 0:
            return 1.0
        elif i == 0:
            return self.local_b[j - 1]
        elif j == 0:
            return self.local_a[i - 1]
        return self.correlation[i - 1, j - 1]
    
    def connected_correlation(self, pauli_a: str, pauli_b: str) -> float:
        i, j = PAULI_INDEX[pauli_a.upper()], PAULI_INDEX[pauli_b.upper()]
        if i == 0 or j == 0:
            return 0.0
        full = self.correlation[i - 1, j - 1]
        product = self.local_a[i - 1] * self.local_b[j - 1]
        return full - product
    
    def is_product_state(self, tol: float = 1e-8) -> bool:
        expected = np.outer(self.local_a, self.local_b)
        return np.allclose(self.correlation, expected, atol=tol)
    
    def validate(self, strict: bool = True) -> Tuple[bool, Optional[str]]:
        if np.any(np.abs(self.local_a) > 1.0 + 1e-10):
            return False, f"local_a exceeds bounds: {self.local_a}"
        if np.any(np.abs(self.local_b) > 1.0 + 1e-10):
            return False, f"local_b exceeds bounds: {self.local_b}"
        if np.any(np.abs(self.correlation) > 1.0 + 1e-10):
            return False, "correlation exceeds bounds"
        
        if strict:
            rho = self.to_density_matrix()
            eigenvalues = np.linalg.eigvalsh(rho)
            if np.any(eigenvalues < -1e-10):
                return False, f"Not positive semi-definite: min eigenvalue = {eigenvalues.min()}"
        return True, None
    
    def to_density_matrix(self) -> np.ndarray:
        from aces.core.density_matrix import PAULI_I, PAULI_X, PAULI_Y, PAULI_Z
        paulis = [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]
        T = self.to_full_tensor()
        rho = np.zeros((4, 4), dtype=np.complex128)
        for i, sigma_i in enumerate(paulis):
            for j, sigma_j in enumerate(paulis):
                rho += T[i, j] * np.kron(sigma_i, sigma_j)
        return rho / 4
    
    def copy(self) -> CorrelatorStorage:
        return CorrelatorStorage(
            qubit_a=self.qubit_a, qubit_b=self.qubit_b,
            local_a=self.local_a.copy(), local_b=self.local_b.copy(),
            correlation=self.correlation.copy()
        )
    
    def __repr__(self) -> str:
        return f"CorrelatorStorage(q{self.qubit_a}, q{self.qubit_b}, product={self.is_product_state()})"
