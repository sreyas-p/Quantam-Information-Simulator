"""Reduced Density Matrix representations for ACES."""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, field


PAULI_I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


@dataclass
class SingleQubitRDM:
    """Single-qubit RDM using Bloch representation: ρ = (I + r·σ)/2."""
    qubit_id: int
    bloch: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    
    def __post_init__(self):
        self.bloch = np.asarray(self.bloch, dtype=np.float64)
        if self.bloch.shape != (3,):
            raise ValueError(f"Bloch vector must have shape (3,), got {self.bloch.shape}")
    
    @classmethod
    def from_density_matrix(cls, qubit_id: int, rho: np.ndarray) -> SingleQubitRDM:
        rx = np.real(np.trace(rho @ PAULI_X))
        ry = np.real(np.trace(rho @ PAULI_Y))
        rz = np.real(np.trace(rho @ PAULI_Z))
        return cls(qubit_id=qubit_id, bloch=np.array([rx, ry, rz]))
    
    @classmethod
    def zero_state(cls, qubit_id: int) -> SingleQubitRDM:
        return cls(qubit_id=qubit_id, bloch=np.array([0.0, 0.0, 1.0]))
    
    @classmethod
    def one_state(cls, qubit_id: int) -> SingleQubitRDM:
        return cls(qubit_id=qubit_id, bloch=np.array([0.0, 0.0, -1.0]))
    
    @classmethod
    def plus_state(cls, qubit_id: int) -> SingleQubitRDM:
        return cls(qubit_id=qubit_id, bloch=np.array([1.0, 0.0, 0.0]))
    
    @classmethod
    def maximally_mixed(cls, qubit_id: int) -> SingleQubitRDM:
        return cls(qubit_id=qubit_id, bloch=np.array([0.0, 0.0, 0.0]))
    
    def to_density_matrix(self) -> np.ndarray:
        rx, ry, rz = self.bloch
        return (PAULI_I + rx * PAULI_X + ry * PAULI_Y + rz * PAULI_Z) / 2
    
    @property
    def purity(self) -> float:
        return (1 + np.dot(self.bloch, self.bloch)) / 2
    
    @property
    def is_pure(self) -> bool:
        return np.isclose(self.purity, 1.0, atol=1e-10)
    
    def von_neumann_entropy(self) -> float:
        r_norm = np.linalg.norm(self.bloch)
        if r_norm >= 1.0 - 1e-10:
            return 0.0
        lambda_plus = (1 + r_norm) / 2
        lambda_minus = (1 - r_norm) / 2
        entropy = 0.0
        if lambda_plus > 1e-15:
            entropy -= lambda_plus * np.log2(lambda_plus)
        if lambda_minus > 1e-15:
            entropy -= lambda_minus * np.log2(lambda_minus)
        return entropy
    
    def expectation(self, pauli: str) -> float:
        pauli = pauli.upper()
        if pauli == 'X':
            return self.bloch[0]
        elif pauli == 'Y':
            return self.bloch[1]
        elif pauli == 'Z':
            return self.bloch[2]
        elif pauli == 'I':
            return 1.0
        raise ValueError(f"Unknown Pauli operator: {pauli}")
    
    def probability_zero(self) -> float:
        return (1 + self.bloch[2]) / 2
    
    def probability_one(self) -> float:
        return (1 - self.bloch[2]) / 2
    
    def copy(self) -> SingleQubitRDM:
        return SingleQubitRDM(qubit_id=self.qubit_id, bloch=self.bloch.copy())
    
    @property
    def is_valid_density_matrix(self) -> bool:
        return np.linalg.norm(self.bloch) <= 1.0 + 1e-10
    
    def validate(self, strict: bool = True) -> Tuple[bool, Optional[str]]:
        r_norm = np.linalg.norm(self.bloch)
        if r_norm > 1.0 + 1e-10:
            return False, f"Bloch vector norm {r_norm:.6f} > 1"
        if strict:
            rho = self.to_density_matrix()
            if not np.allclose(rho, rho.conj().T, atol=1e-12):
                return False, "Not Hermitian"
            if not np.isclose(np.trace(rho), 1.0, atol=1e-12):
                return False, "Trace != 1"
            eigenvalues = np.linalg.eigvalsh(rho)
            if np.any(eigenvalues < -1e-12):
                return False, f"Not positive semi-definite"
        return True, None
    
    def __repr__(self) -> str:
        return f"SingleQubitRDM(q{self.qubit_id}, bloch={self.bloch})"


@dataclass
class TwoQubitRDM:
    """Two-qubit RDM storing full 4x4 density matrix."""
    qubit_a: int
    qubit_b: int
    rho: np.ndarray = field(default_factory=lambda: np.diag([1.0, 0, 0, 0]).astype(np.complex128))
    _correlation_tensor: Optional[np.ndarray] = field(default=None, repr=False)
    
    def __post_init__(self):
        self.rho = np.asarray(self.rho, dtype=np.complex128)
        if self.rho.shape != (4, 4):
            raise ValueError(f"Density matrix must be 4x4, got {self.rho.shape}")
        self._correlation_tensor = None
    
    @classmethod
    def product_state(cls, rdm_a: SingleQubitRDM, rdm_b: SingleQubitRDM) -> TwoQubitRDM:
        rho = np.kron(rdm_a.to_density_matrix(), rdm_b.to_density_matrix())
        return cls(qubit_a=rdm_a.qubit_id, qubit_b=rdm_b.qubit_id, rho=rho)
    
    @classmethod
    def zero_zero_state(cls, qubit_a: int, qubit_b: int) -> TwoQubitRDM:
        rho = np.diag([1.0, 0, 0, 0]).astype(np.complex128)
        return cls(qubit_a=qubit_a, qubit_b=qubit_b, rho=rho)
    
    @classmethod
    def bell_phi_plus(cls, qubit_a: int, qubit_b: int) -> TwoQubitRDM:
        rho = np.array([
            [0.5, 0, 0, 0.5],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0.5, 0, 0, 0.5]
        ], dtype=np.complex128)
        return cls(qubit_a=qubit_a, qubit_b=qubit_b, rho=rho)
    
    @property
    def correlation_tensor(self) -> np.ndarray:
        if self._correlation_tensor is not None:
            return self._correlation_tensor
        paulis = [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]
        T = np.zeros((4, 4), dtype=np.float64)
        for i, sigma_i in enumerate(paulis):
            for j, sigma_j in enumerate(paulis):
                op = np.kron(sigma_i, sigma_j)
                T[i, j] = np.real(np.trace(self.rho @ op))
        self._correlation_tensor = T
        return T
    
    def marginal_a(self) -> SingleQubitRDM:
        rho_a = np.zeros((2, 2), dtype=np.complex128)
        rho_a[0, 0] = self.rho[0, 0] + self.rho[1, 1]
        rho_a[0, 1] = self.rho[0, 2] + self.rho[1, 3]
        rho_a[1, 0] = self.rho[2, 0] + self.rho[3, 1]
        rho_a[1, 1] = self.rho[2, 2] + self.rho[3, 3]
        return SingleQubitRDM.from_density_matrix(self.qubit_a, rho_a)
    
    def marginal_b(self) -> SingleQubitRDM:
        rho_b = np.zeros((2, 2), dtype=np.complex128)
        rho_b[0, 0] = self.rho[0, 0] + self.rho[2, 2]
        rho_b[0, 1] = self.rho[0, 1] + self.rho[2, 3]
        rho_b[1, 0] = self.rho[1, 0] + self.rho[3, 2]
        rho_b[1, 1] = self.rho[1, 1] + self.rho[3, 3]
        return SingleQubitRDM.from_density_matrix(self.qubit_b, rho_b)
    
    def connected_correlation(self, pauli_a: str, pauli_b: str) -> float:
        T = self.correlation_tensor
        idx_map = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
        i, j = idx_map[pauli_a.upper()], idx_map[pauli_b.upper()]
        return T[i, j] - T[i, 0] * T[0, j]
    
    def mutual_information(self) -> float:
        s_a = self.marginal_a().von_neumann_entropy()
        s_b = self.marginal_b().von_neumann_entropy()
        eigenvalues = np.linalg.eigvalsh(self.rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        s_ab = -np.sum(eigenvalues * np.log2(eigenvalues))
        return s_a + s_b - s_ab
    
    def concurrence(self) -> float:
        sigma_y_tensor = np.kron(PAULI_Y, PAULI_Y)
        rho_tilde = sigma_y_tensor @ np.conj(self.rho) @ sigma_y_tensor
        R = self.rho @ rho_tilde
        eigenvalues = np.sqrt(np.maximum(np.real(np.linalg.eigvals(R)), 0))
        eigenvalues = np.sort(eigenvalues)[::-1]
        return max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])
    
    def is_product_state(self, tol: float = 1e-8) -> bool:
        return self.concurrence() < tol
    
    def to_product_approximation(self) -> Tuple[SingleQubitRDM, SingleQubitRDM]:
        return self.marginal_a(), self.marginal_b()
    
    @property
    def purity(self) -> float:
        return np.real(np.trace(self.rho @ self.rho))
    
    def copy(self) -> TwoQubitRDM:
        return TwoQubitRDM(qubit_a=self.qubit_a, qubit_b=self.qubit_b, rho=self.rho.copy())
    
    def invalidate_cache(self):
        self._correlation_tensor = None
    
    @property
    def is_valid_density_matrix(self) -> bool:
        if not np.allclose(self.rho, self.rho.conj().T, atol=1e-10):
            return False
        if not np.isclose(np.trace(self.rho), 1.0, atol=1e-10):
            return False
        eigenvalues = np.linalg.eigvalsh(self.rho)
        return not np.any(eigenvalues < -1e-10)
    
    def validate(self, strict: bool = True) -> Tuple[bool, Optional[str]]:
        if not np.allclose(self.rho, self.rho.conj().T, atol=1e-10):
            return False, "Not Hermitian"
        trace = np.real(np.trace(self.rho))
        if not np.isclose(trace, 1.0, atol=1e-10):
            return False, f"Trace = {trace:.6f}"
        if strict:
            eigenvalues = np.linalg.eigvalsh(self.rho)
            if eigenvalues.min() < -1e-10:
                return False, f"Not positive semi-definite"
        return True, None
    
    def to_correlator_storage(self):
        from aces.core.correlator import CorrelatorStorage
        return CorrelatorStorage.from_full_tensor(self.qubit_a, self.qubit_b, self.correlation_tensor)
    
    def __repr__(self) -> str:
        return f"TwoQubitRDM(q{self.qubit_a}, q{self.qubit_b}, purity={self.purity:.4f})"
