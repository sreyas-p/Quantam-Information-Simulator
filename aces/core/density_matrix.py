"""
Reduced Density Matrix representations for ACES.

This module provides efficient representations for 1-qubit and 2-qubit
reduced density matrices, which are the core state representations used
throughout the ACES runtime.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, field


# Pauli matrices for convenience
PAULI_I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


@dataclass
class SingleQubitRDM:
    """
    Single-qubit reduced density matrix.
    
    Internally stores the Bloch vector (rx, ry, rz) for efficiency:
        ρ = (I + rx*X + ry*Y + rz*Z) / 2
    
    This parameterization is more efficient for updates and naturally
    represents the physical constraint |r| <= 1 (equality for pure states).
    
    Attributes:
        qubit_id: Identifier for this qubit
        bloch: 3-element array [rx, ry, rz] representing the Bloch vector
    """
    qubit_id: int
    bloch: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    
    def __post_init__(self):
        """Ensure bloch vector is the right shape and type."""
        self.bloch = np.asarray(self.bloch, dtype=np.float64)
        if self.bloch.shape != (3,):
            raise ValueError(f"Bloch vector must have shape (3,), got {self.bloch.shape}")
    
    @classmethod
    def from_density_matrix(cls, qubit_id: int, rho: np.ndarray) -> SingleQubitRDM:
        """
        Create from a 2x2 density matrix.
        
        Args:
            qubit_id: Identifier for this qubit
            rho: 2x2 complex density matrix
            
        Returns:
            SingleQubitRDM instance
        """
        # Extract Bloch components: r_i = Tr(ρ * σ_i)
        rx = np.real(np.trace(rho @ PAULI_X))
        ry = np.real(np.trace(rho @ PAULI_Y))
        rz = np.real(np.trace(rho @ PAULI_Z))
        return cls(qubit_id=qubit_id, bloch=np.array([rx, ry, rz]))
    
    @classmethod
    def zero_state(cls, qubit_id: int) -> SingleQubitRDM:
        """Create RDM for |0⟩ state (Bloch vector pointing +Z)."""
        return cls(qubit_id=qubit_id, bloch=np.array([0.0, 0.0, 1.0]))
    
    @classmethod
    def one_state(cls, qubit_id: int) -> SingleQubitRDM:
        """Create RDM for |1⟩ state (Bloch vector pointing -Z)."""
        return cls(qubit_id=qubit_id, bloch=np.array([0.0, 0.0, -1.0]))
    
    @classmethod
    def plus_state(cls, qubit_id: int) -> SingleQubitRDM:
        """Create RDM for |+⟩ state (Bloch vector pointing +X)."""
        return cls(qubit_id=qubit_id, bloch=np.array([1.0, 0.0, 0.0]))
    
    @classmethod
    def maximally_mixed(cls, qubit_id: int) -> SingleQubitRDM:
        """Create maximally mixed state (Bloch vector at origin)."""
        return cls(qubit_id=qubit_id, bloch=np.array([0.0, 0.0, 0.0]))
    
    def to_density_matrix(self) -> np.ndarray:
        """
        Convert to 2x2 density matrix representation.
        
        Returns:
            2x2 complex numpy array
        """
        rx, ry, rz = self.bloch
        return (PAULI_I + rx * PAULI_X + ry * PAULI_Y + rz * PAULI_Z) / 2
    
    @property
    def purity(self) -> float:
        """
        Compute purity Tr(ρ²).
        
        For Bloch representation: purity = (1 + |r|²) / 2
        Range: [0.5, 1.0] where 0.5 is maximally mixed, 1.0 is pure.
        """
        return (1 + np.dot(self.bloch, self.bloch)) / 2
    
    @property
    def is_pure(self) -> bool:
        """Check if state is approximately pure."""
        return np.isclose(self.purity, 1.0, atol=1e-10)
    
    def von_neumann_entropy(self) -> float:
        """
        Compute von Neumann entropy S(ρ) = -Tr(ρ log ρ).
        
        For a qubit: S = -λ₊ log λ₊ - λ₋ log λ₋
        where λ± = (1 ± |r|) / 2 are the eigenvalues.
        """
        r_norm = np.linalg.norm(self.bloch)
        if r_norm >= 1.0 - 1e-10:
            return 0.0  # Pure state
        
        lambda_plus = (1 + r_norm) / 2
        lambda_minus = (1 - r_norm) / 2
        
        entropy = 0.0
        if lambda_plus > 1e-15:
            entropy -= lambda_plus * np.log2(lambda_plus)
        if lambda_minus > 1e-15:
            entropy -= lambda_minus * np.log2(lambda_minus)
        
        return entropy
    
    def expectation(self, pauli: str) -> float:
        """
        Compute expectation value of a Pauli operator.
        
        Args:
            pauli: One of 'X', 'Y', 'Z', 'I'
            
        Returns:
            Expectation value ⟨σ⟩
        """
        pauli = pauli.upper()
        if pauli == 'X':
            return self.bloch[0]
        elif pauli == 'Y':
            return self.bloch[1]
        elif pauli == 'Z':
            return self.bloch[2]
        elif pauli == 'I':
            return 1.0
        else:
            raise ValueError(f"Unknown Pauli operator: {pauli}")
    
    def probability_zero(self) -> float:
        """Probability of measuring |0⟩ in computational basis."""
        return (1 + self.bloch[2]) / 2
    
    def probability_one(self) -> float:
        """Probability of measuring |1⟩ in computational basis."""
        return (1 - self.bloch[2]) / 2
    
    def copy(self) -> SingleQubitRDM:
        """Create a deep copy of this RDM."""
        return SingleQubitRDM(qubit_id=self.qubit_id, bloch=self.bloch.copy())
    
    def __repr__(self) -> str:
        return f"SingleQubitRDM(q{self.qubit_id}, bloch={self.bloch})"


@dataclass
class TwoQubitRDM:
    """
    Two-qubit reduced density matrix.
    
    Stores the full 4x4 density matrix for a pair of qubits.
    Also caches the correlation tensor for efficient access.
    
    The density matrix is parameterized as:
        ρ = (1/4) Σᵢⱼ Tᵢⱼ (σᵢ ⊗ σⱼ)
    
    where T is the correlation tensor with:
        - T[0,0] = 1 (normalization)
        - T[i,0] = ⟨σᵢ ⊗ I⟩ (local expectation on qubit A)
        - T[0,j] = ⟨I ⊗ σⱼ⟩ (local expectation on qubit B)
        - T[i,j] = ⟨σᵢ ⊗ σⱼ⟩ (correlations)
    
    Attributes:
        qubit_a: Identifier for first qubit
        qubit_b: Identifier for second qubit  
        rho: 4x4 complex density matrix
    """
    qubit_a: int
    qubit_b: int
    rho: np.ndarray = field(default_factory=lambda: np.diag([1.0, 0, 0, 0]).astype(np.complex128))
    _correlation_tensor: Optional[np.ndarray] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate density matrix."""
        self.rho = np.asarray(self.rho, dtype=np.complex128)
        if self.rho.shape != (4, 4):
            raise ValueError(f"Density matrix must be 4x4, got {self.rho.shape}")
        self._correlation_tensor = None
    
    @classmethod
    def product_state(cls, rdm_a: SingleQubitRDM, rdm_b: SingleQubitRDM) -> TwoQubitRDM:
        """
        Create a product state from two single-qubit RDMs.
        
        Args:
            rdm_a: RDM for first qubit
            rdm_b: RDM for second qubit
            
        Returns:
            TwoQubitRDM representing ρ_A ⊗ ρ_B
        """
        rho = np.kron(rdm_a.to_density_matrix(), rdm_b.to_density_matrix())
        return cls(qubit_a=rdm_a.qubit_id, qubit_b=rdm_b.qubit_id, rho=rho)
    
    @classmethod
    def zero_zero_state(cls, qubit_a: int, qubit_b: int) -> TwoQubitRDM:
        """Create |00⟩ state."""
        rho = np.diag([1.0, 0, 0, 0]).astype(np.complex128)
        return cls(qubit_a=qubit_a, qubit_b=qubit_b, rho=rho)
    
    @classmethod
    def bell_phi_plus(cls, qubit_a: int, qubit_b: int) -> TwoQubitRDM:
        """Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
        rho = np.array([
            [0.5, 0, 0, 0.5],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0.5, 0, 0, 0.5]
        ], dtype=np.complex128)
        return cls(qubit_a=qubit_a, qubit_b=qubit_b, rho=rho)
    
    @property
    def correlation_tensor(self) -> np.ndarray:
        """
        Compute the 4x4 correlation tensor T where T[i,j] = ⟨σᵢ ⊗ σⱼ⟩.
        
        Index mapping: 0=I, 1=X, 2=Y, 3=Z
        """
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
        """Compute reduced density matrix by tracing out qubit B."""
        # Partial trace over qubit B
        rho_a = np.zeros((2, 2), dtype=np.complex128)
        rho_a[0, 0] = self.rho[0, 0] + self.rho[1, 1]
        rho_a[0, 1] = self.rho[0, 2] + self.rho[1, 3]
        rho_a[1, 0] = self.rho[2, 0] + self.rho[3, 1]
        rho_a[1, 1] = self.rho[2, 2] + self.rho[3, 3]
        return SingleQubitRDM.from_density_matrix(self.qubit_a, rho_a)
    
    def marginal_b(self) -> SingleQubitRDM:
        """Compute reduced density matrix by tracing out qubit A."""
        # Partial trace over qubit A
        rho_b = np.zeros((2, 2), dtype=np.complex128)
        rho_b[0, 0] = self.rho[0, 0] + self.rho[2, 2]
        rho_b[0, 1] = self.rho[0, 1] + self.rho[2, 3]
        rho_b[1, 0] = self.rho[1, 0] + self.rho[3, 2]
        rho_b[1, 1] = self.rho[1, 1] + self.rho[3, 3]
        return SingleQubitRDM.from_density_matrix(self.qubit_b, rho_b)
    
    def connected_correlation(self, pauli_a: str, pauli_b: str) -> float:
        """
        Compute connected correlation ⟨σₐσᵦ⟩ - ⟨σₐ⟩⟨σᵦ⟩.
        
        This measures genuine quantum/classical correlations beyond
        what's expected from the marginals alone.
        """
        T = self.correlation_tensor
        idx_map = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
        i = idx_map[pauli_a.upper()]
        j = idx_map[pauli_b.upper()]
        
        # Full correlation
        full = T[i, j]
        
        # Product of marginal expectations
        marginal_product = T[i, 0] * T[0, j]
        
        return full - marginal_product
    
    def mutual_information(self) -> float:
        """
        Compute quantum mutual information I(A:B) = S(A) + S(B) - S(AB).
        """
        s_a = self.marginal_a().von_neumann_entropy()
        s_b = self.marginal_b().von_neumann_entropy()
        
        # S(AB) from eigenvalues of full density matrix
        eigenvalues = np.linalg.eigvalsh(self.rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        s_ab = -np.sum(eigenvalues * np.log2(eigenvalues))
        
        return s_a + s_b - s_ab
    
    def concurrence(self) -> float:
        """
        Compute concurrence, a measure of entanglement.
        
        C(ρ) = max(0, λ₁ - λ₂ - λ₃ - λ₄)
        where λᵢ are square roots of eigenvalues of ρ(σʸ⊗σʸ)ρ*(σʸ⊗σʸ)
        in decreasing order.
        """
        sigma_y_tensor = np.kron(PAULI_Y, PAULI_Y)
        rho_tilde = sigma_y_tensor @ np.conj(self.rho) @ sigma_y_tensor
        
        R = self.rho @ rho_tilde
        eigenvalues = np.sqrt(np.maximum(np.real(np.linalg.eigvals(R)), 0))
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        return max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])
    
    def is_product_state(self, tol: float = 1e-8) -> bool:
        """Check if state is approximately a product state."""
        return self.concurrence() < tol
    
    def to_product_approximation(self) -> Tuple[SingleQubitRDM, SingleQubitRDM]:
        """
        Return the product state that best approximates this state
        (simply the tensor product of marginals).
        """
        return self.marginal_a(), self.marginal_b()
    
    @property
    def purity(self) -> float:
        """Compute purity Tr(ρ²)."""
        return np.real(np.trace(self.rho @ self.rho))
    
    def copy(self) -> TwoQubitRDM:
        """Create a deep copy of this RDM."""
        return TwoQubitRDM(
            qubit_a=self.qubit_a,
            qubit_b=self.qubit_b,
            rho=self.rho.copy()
        )
    
    def invalidate_cache(self):
        """Invalidate cached values after mutation."""
        self._correlation_tensor = None
    
    def __repr__(self) -> str:
        return f"TwoQubitRDM(q{self.qubit_a}, q{self.qubit_b}, purity={self.purity:.4f})"
