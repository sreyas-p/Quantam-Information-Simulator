"""
Gate library with CPTP update rules for ACES.

This module defines quantum gates and provides methods to apply them
to single-qubit and two-qubit reduced density matrices.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Callable, Optional, Tuple, List
from enum import Enum
from abc import ABC, abstractmethod

from aces.core.density_matrix import SingleQubitRDM, TwoQubitRDM


# Common gate matrices
PAULI_I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
S_GATE = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
SQRT_X = np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=np.complex128) / 2


def rx_matrix(theta: float) -> np.ndarray:
    """Rotation around X-axis: Rx(θ) = exp(-iθX/2)"""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def ry_matrix(theta: float) -> np.ndarray:
    """Rotation around Y-axis: Ry(θ) = exp(-iθY/2)"""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz_matrix(theta: float) -> np.ndarray:
    """Rotation around Z-axis: Rz(θ) = exp(-iθZ/2)"""
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=np.complex128)


def u3_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    """General single-qubit unitary U3(θ, φ, λ)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
    ], dtype=np.complex128)


# Two-qubit gates
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=np.complex128)

CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
], dtype=np.complex128)

SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=np.complex128)

ISWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1]
], dtype=np.complex128)


def rxx_matrix(theta: float) -> np.ndarray:
    """XX rotation: Rxx(θ) = exp(-iθXX/2)"""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [c, 0, 0, -1j * s],
        [0, c, -1j * s, 0],
        [0, -1j * s, c, 0],
        [-1j * s, 0, 0, c]
    ], dtype=np.complex128)


def ryy_matrix(theta: float) -> np.ndarray:
    """YY rotation: Ryy(θ) = exp(-iθYY/2)"""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [c, 0, 0, 1j * s],
        [0, c, -1j * s, 0],
        [0, -1j * s, c, 0],
        [1j * s, 0, 0, c]
    ], dtype=np.complex128)


def rzz_matrix(theta: float) -> np.ndarray:
    """ZZ rotation: Rzz(θ) = exp(-iθZZ/2)"""
    return np.array([
        [np.exp(-1j * theta / 2), 0, 0, 0],
        [0, np.exp(1j * theta / 2), 0, 0],
        [0, 0, np.exp(1j * theta / 2), 0],
        [0, 0, 0, np.exp(-1j * theta / 2)]
    ], dtype=np.complex128)


class GateType(Enum):
    """Classification of gate types."""
    SINGLE_QUBIT = "single"
    TWO_QUBIT = "two"


@dataclass
class Gate(ABC):
    """Abstract base class for quantum gates."""
    name: str
    
    @property
    @abstractmethod
    def gate_type(self) -> GateType:
        """Return the type of this gate."""
        pass
    
    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits this gate acts on."""
        pass
    
    @abstractmethod
    def matrix(self, params: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Return the unitary matrix for this gate.
        
        Args:
            params: Optional parameter dictionary for parametric gates
        """
        pass
    
    @abstractmethod
    def apply_to_rdm(
        self,
        state: Dict[int, SingleQubitRDM],
        correlators: Dict[Tuple[int, int], TwoQubitRDM],
        qubits: Tuple[int, ...],
        params: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Apply this gate to the RDM state representation.
        
        This modifies state and correlators in-place.
        
        Args:
            state: Dictionary mapping qubit_id -> SingleQubitRDM
            correlators: Dictionary mapping (q1, q2) -> TwoQubitRDM
            qubits: Tuple of qubit indices this gate acts on
            params: Optional parameters for parametric gates
        """
        pass


@dataclass
class SingleQubitGate(Gate):
    """Single-qubit unitary gate."""
    _matrix: np.ndarray = field(repr=False)
    param_names: Tuple[str, ...] = field(default_factory=tuple)
    matrix_fn: Optional[Callable[..., np.ndarray]] = field(default=None, repr=False)
    
    @property
    def gate_type(self) -> GateType:
        return GateType.SINGLE_QUBIT
    
    @property
    def num_qubits(self) -> int:
        return 1
    
    def matrix(self, params: Optional[Dict[str, float]] = None) -> np.ndarray:
        if self.matrix_fn is not None and params is not None:
            param_values = [params[p] for p in self.param_names]
            return self.matrix_fn(*param_values)
        return self._matrix
    
    def apply_to_rdm(
        self,
        state: Dict[int, SingleQubitRDM],
        correlators: Dict[Tuple[int, int], TwoQubitRDM],
        qubits: Tuple[int, ...],
        params: Optional[Dict[str, float]] = None
    ) -> None:
        """Apply single-qubit gate to RDM state."""
        q = qubits[0]
        U = self.matrix(params)
        U_dag = U.conj().T
        
        # Update single-qubit RDM: ρ' = U ρ U†
        if q in state:
            rho = state[q].to_density_matrix()
            rho_new = U @ rho @ U_dag
            state[q] = SingleQubitRDM.from_density_matrix(q, rho_new)
        
        # Update all correlators involving this qubit
        for (qa, qb), corr in list(correlators.items()):
            if qa == q or qb == q:
                # Build operator: U ⊗ I or I ⊗ U depending on position
                if qa == q:
                    full_U = np.kron(U, PAULI_I)
                else:
                    full_U = np.kron(PAULI_I, U)
                
                rho_new = full_U @ corr.rho @ full_U.conj().T
                corr.rho = rho_new
                corr.invalidate_cache()


@dataclass  
class TwoQubitGate(Gate):
    """Two-qubit unitary gate."""
    _matrix: np.ndarray = field(repr=False)
    param_names: Tuple[str, ...] = field(default_factory=tuple)
    matrix_fn: Optional[Callable[..., np.ndarray]] = field(default=None, repr=False)
    
    @property
    def gate_type(self) -> GateType:
        return GateType.TWO_QUBIT
    
    @property
    def num_qubits(self) -> int:
        return 2
    
    def matrix(self, params: Optional[Dict[str, float]] = None) -> np.ndarray:
        if self.matrix_fn is not None and params is not None:
            param_values = [params[p] for p in self.param_names]
            return self.matrix_fn(*param_values)
        return self._matrix
    
    def apply_to_rdm(
        self,
        state: Dict[int, SingleQubitRDM],
        correlators: Dict[Tuple[int, int], TwoQubitRDM],
        qubits: Tuple[int, ...],
        params: Optional[Dict[str, float]] = None
    ) -> None:
        """Apply two-qubit gate to RDM state."""
        qa, qb = qubits
        U = self.matrix(params)
        U_dag = U.conj().T
        
        # Ensure correlator exists for this pair
        key = (qa, qb) if qa < qb else (qb, qa)
        swap_needed = key != (qa, qb)
        
        if key not in correlators:
            # Create product state correlator from marginals
            rdm_a = state.get(key[0], SingleQubitRDM.zero_state(key[0]))
            rdm_b = state.get(key[1], SingleQubitRDM.zero_state(key[1]))
            correlators[key] = TwoQubitRDM.product_state(rdm_a, rdm_b)
        
        corr = correlators[key]
        
        # If qubits are in reversed order, we need to swap
        if swap_needed:
            U_applied = SWAP @ U @ SWAP
        else:
            U_applied = U
        
        # Apply gate: ρ' = U ρ U†
        rho_new = U_applied @ corr.rho @ U_applied.conj().T
        corr.rho = rho_new
        corr.invalidate_cache()
        
        # Update single-qubit marginals
        state[qa] = corr.marginal_a() if not swap_needed else corr.marginal_b()
        state[qb] = corr.marginal_b() if not swap_needed else corr.marginal_a()
        state[qa].qubit_id = qa
        state[qb].qubit_id = qb
        
        # Update other correlators that share qubits with this pair
        self._propagate_to_other_correlators(state, correlators, qubits, U, params)
    
    def _propagate_to_other_correlators(
        self,
        state: Dict[int, SingleQubitRDM],
        correlators: Dict[Tuple[int, int], TwoQubitRDM],
        acted_qubits: Tuple[int, int],
        U: np.ndarray,
        params: Optional[Dict[str, float]]
    ) -> None:
        """
        Propagate gate effects to correlators that share one qubit with the acted pair.
        
        This is where causality-aware truncation becomes important: we may choose
        not to create or update correlators that fall below the entanglement threshold.
        """
        qa, qb = acted_qubits
        
        for (q1, q2), corr in list(correlators.items()):
            if (q1, q2) == (min(qa, qb), max(qa, qb)):
                continue  # Already handled
            
            # If correlator shares exactly one qubit with the gate
            if qa in (q1, q2) or qb in (q1, q2):
                # For now, update marginals only (approximation)
                # Full update would require tracking 3-qubit correlators
                new_marginal_1 = state.get(q1, SingleQubitRDM.zero_state(q1))
                new_marginal_2 = state.get(q2, SingleQubitRDM.zero_state(q2))
                
                # Update correlator to maintain consistency with new marginals
                # This is an approximation that preserves local marginals
                # but may not perfectly preserve all correlations
                corr_new = TwoQubitRDM.product_state(new_marginal_1, new_marginal_2)
                
                # Blend with existing correlations (heuristic)
                # Keep connected correlations at reduced strength
                old_T = corr.correlation_tensor
                new_T = corr_new.correlation_tensor
                
                # Preserve some of the off-diagonal correlations
                # This is a simplification; proper treatment needs 3-body terms
                blend_factor = 0.5
                for i in range(1, 4):
                    for j in range(1, 4):
                        new_T[i, j] = blend_factor * old_T[i, j] + (1 - blend_factor) * new_T[i, j]


class GateLibrary:
    """Library of common gates with efficient update rules."""
    
    # Single-qubit gates
    X = SingleQubitGate(name="X", _matrix=PAULI_X)
    Y = SingleQubitGate(name="Y", _matrix=PAULI_Y)
    Z = SingleQubitGate(name="Z", _matrix=PAULI_Z)
    H = SingleQubitGate(name="H", _matrix=HADAMARD)
    S = SingleQubitGate(name="S", _matrix=S_GATE)
    Sdg = SingleQubitGate(name="Sdg", _matrix=S_GATE.conj().T)
    T = SingleQubitGate(name="T", _matrix=T_GATE)
    Tdg = SingleQubitGate(name="Tdg", _matrix=T_GATE.conj().T)
    SX = SingleQubitGate(name="SX", _matrix=SQRT_X)
    
    # Parametric single-qubit gates
    Rx = SingleQubitGate(
        name="Rx",
        _matrix=PAULI_I,  # Placeholder
        param_names=("theta",),
        matrix_fn=rx_matrix
    )
    Ry = SingleQubitGate(
        name="Ry", 
        _matrix=PAULI_I,
        param_names=("theta",),
        matrix_fn=ry_matrix
    )
    Rz = SingleQubitGate(
        name="Rz",
        _matrix=PAULI_I,
        param_names=("theta",),
        matrix_fn=rz_matrix
    )
    U3 = SingleQubitGate(
        name="U3",
        _matrix=PAULI_I,
        param_names=("theta", "phi", "lambda"),
        matrix_fn=u3_matrix
    )
    
    # Two-qubit gates
    CX = TwoQubitGate(name="CX", _matrix=CNOT)
    CNOT = CX  # Alias
    CZ = TwoQubitGate(name="CZ", _matrix=CZ)
    SWAP = TwoQubitGate(name="SWAP", _matrix=SWAP)
    iSWAP = TwoQubitGate(name="iSWAP", _matrix=ISWAP)
    
    # Parametric two-qubit gates  
    Rxx = TwoQubitGate(
        name="Rxx",
        _matrix=np.eye(4, dtype=np.complex128),
        param_names=("theta",),
        matrix_fn=rxx_matrix
    )
    Ryy = TwoQubitGate(
        name="Ryy",
        _matrix=np.eye(4, dtype=np.complex128),
        param_names=("theta",),
        matrix_fn=ryy_matrix
    )
    Rzz = TwoQubitGate(
        name="Rzz",
        _matrix=np.eye(4, dtype=np.complex128),
        param_names=("theta",),
        matrix_fn=rzz_matrix
    )
    
    @classmethod
    def get_gate(cls, name: str) -> Gate:
        """Get a gate by name."""
        name_upper = name.upper()
        name_map = {
            "X": cls.X, "Y": cls.Y, "Z": cls.Z,
            "H": cls.H, "S": cls.S, "SDG": cls.Sdg,
            "T": cls.T, "TDG": cls.Tdg, "SX": cls.SX,
            "RX": cls.Rx, "RY": cls.Ry, "RZ": cls.Rz,
            "U3": cls.U3, "U": cls.U3,
            "CX": cls.CX, "CNOT": cls.CNOT,
            "CZ": cls.CZ, "SWAP": cls.SWAP, "ISWAP": cls.iSWAP,
            "RXX": cls.Rxx, "RYY": cls.Ryy, "RZZ": cls.Rzz,
        }
        
        if name_upper not in name_map:
            raise ValueError(f"Unknown gate: {name}")
        
        return name_map[name_upper]


# Noise channels as CPTP maps
@dataclass
class NoiseChannel:
    """Base class for noise channels (CPTP maps)."""
    name: str
    
    @abstractmethod
    def apply(self, rdm: SingleQubitRDM, params: Dict[str, float]) -> None:
        """Apply noise channel to single-qubit RDM in-place."""
        pass


@dataclass
class DepolarizingChannel(NoiseChannel):
    """
    Depolarizing channel: ρ → (1-p)ρ + p·I/2
    
    In Bloch representation: r → (1-p)r
    """
    name: str = "depolarizing"
    
    def apply(self, rdm: SingleQubitRDM, params: Dict[str, float]) -> None:
        p = params.get("p", 0.0)
        rdm.bloch *= (1 - p)


@dataclass
class DephazingChannel(NoiseChannel):
    """
    Dephasing (phase damping) channel.
    
    In Bloch representation:
        rx → (1-p)rx
        ry → (1-p)ry  
        rz → rz (unchanged)
    """
    name: str = "dephasing"
    
    def apply(self, rdm: SingleQubitRDM, params: Dict[str, float]) -> None:
        p = params.get("p", 0.0)
        rdm.bloch[0] *= (1 - p)
        rdm.bloch[1] *= (1 - p)


@dataclass
class AmplitudeDampingChannel(NoiseChannel):
    """
    Amplitude damping channel (T1 decay).
    
    Models energy relaxation to |0⟩ state.
    """
    name: str = "amplitude_damping"
    
    def apply(self, rdm: SingleQubitRDM, params: Dict[str, float]) -> None:
        gamma = params.get("gamma", 0.0)
        
        # Transform Bloch vector for amplitude damping
        rx, ry, rz = rdm.bloch
        rdm.bloch[0] = np.sqrt(1 - gamma) * rx
        rdm.bloch[1] = np.sqrt(1 - gamma) * ry
        rdm.bloch[2] = gamma + (1 - gamma) * rz


class NoiseLibrary:
    """Library of noise channels."""
    
    DEPOLARIZING = DepolarizingChannel()
    DEPHASING = DephazingChannel()
    AMPLITUDE_DAMPING = AmplitudeDampingChannel()
    
    @classmethod
    def get_channel(cls, name: str) -> NoiseChannel:
        """Get a noise channel by name."""
        name_lower = name.lower()
        channel_map = {
            "depolarizing": cls.DEPOLARIZING,
            "dephasing": cls.DEPHASING,
            "amplitude_damping": cls.AMPLITUDE_DAMPING,
            "t1": cls.AMPLITUDE_DAMPING,
            "t2": cls.DEPHASING,
        }
        
        if name_lower not in channel_map:
            raise ValueError(f"Unknown noise channel: {name}")
        
        return channel_map[name_lower]
