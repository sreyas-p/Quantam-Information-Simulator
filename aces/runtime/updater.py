"""
CPTP map updater for ACES runtime.

Applies quantum gates and noise channels to the RDM state representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import numpy as np

from aces.core.density_matrix import SingleQubitRDM, TwoQubitRDM
from aces.core.gates import Gate, GateLibrary, NoiseChannel, NoiseLibrary


@dataclass
class CPTPUpdater:
    """
    Applies CPTP (Completely Positive Trace-Preserving) maps to RDM state.
    
    This is the core update mechanism in ACES. Each quantum gate or
    noise channel is a CPTP map that transforms density matrices.
    
    Attributes:
        noise_after_gate: Optional noise to apply after each gate type
        global_depolarizing_rate: Global depolarizing noise rate
    """
    noise_after_gate: Dict[str, Tuple[NoiseChannel, Dict[str, float]]] = field(
        default_factory=dict
    )
    global_depolarizing_rate: float = 0.0
    
    def apply_gate(
        self,
        gate: Gate,
        state: Dict[int, SingleQubitRDM],
        correlators: Dict[Tuple[int, int], TwoQubitRDM],
        qubits: Tuple[int, ...],
        params: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Apply a gate to the RDM state.
        
        Args:
            gate: The gate to apply
            state: Dictionary of single-qubit RDMs
            correlators: Dictionary of two-qubit RDMs
            qubits: Qubits the gate acts on
            params: Optional parameters for parametric gates
        """
        if gate is None:
            return
        
        # Apply the gate
        gate.apply_to_rdm(state, correlators, qubits, params)
        
        # Apply noise if configured for this gate type
        if gate.name in self.noise_after_gate:
            channel, noise_params = self.noise_after_gate[gate.name]
            for q in qubits:
                if q in state:
                    channel.apply(state[q], noise_params)
        
        # Apply global depolarizing noise
        if self.global_depolarizing_rate > 0:
            for q in qubits:
                if q in state:
                    state[q].bloch *= (1 - self.global_depolarizing_rate)
    
    def apply_noise(
        self,
        channel: NoiseChannel,
        state: Dict[int, SingleQubitRDM],
        qubits: List[int],
        params: Dict[str, float]
    ) -> None:
        """
        Apply a noise channel to specified qubits.
        
        Args:
            channel: The noise channel to apply
            state: Dictionary of single-qubit RDMs  
            qubits: Qubits to apply noise to
            params: Noise parameters
        """
        for q in qubits:
            if q in state:
                channel.apply(state[q], params)
    
    def apply_two_qubit_depolarizing(
        self,
        correlator: TwoQubitRDM,
        p: float
    ) -> None:
        """
        Apply two-qubit depolarizing noise.
        
        ρ → (1-p)ρ + p·I/4
        """
        if p <= 0:
            return
        
        correlator.rho = (1 - p) * correlator.rho + p * np.eye(4) / 4
        correlator.invalidate_cache()


@dataclass  
class NoiseModel:
    """
    Noise model configuration for ACES simulation.
    
    Specifies noise channels to apply during simulation.
    """
    single_qubit_depolarizing: float = 0.0
    two_qubit_depolarizing: float = 0.0
    t1: float = 0.0  # Relaxation time (inverse rate)
    t2: float = 0.0  # Dephasing time (inverse rate)
    gate_times: Dict[str, float] = field(default_factory=dict)
    
    def create_updater(self) -> CPTPUpdater:
        """Create a CPTPUpdater configured with this noise model."""
        noise_after_gate = {}
        
        # Add depolarizing after single-qubit gates
        if self.single_qubit_depolarizing > 0:
            for gate_name in ["X", "Y", "Z", "H", "S", "T", "Rx", "Ry", "Rz"]:
                noise_after_gate[gate_name] = (
                    NoiseLibrary.DEPOLARIZING,
                    {"p": self.single_qubit_depolarizing}
                )
        
        # Add depolarizing after two-qubit gates
        if self.two_qubit_depolarizing > 0:
            for gate_name in ["CX", "CZ", "SWAP", "Rxx", "Ryy", "Rzz"]:
                noise_after_gate[gate_name] = (
                    NoiseLibrary.DEPOLARIZING,
                    {"p": self.two_qubit_depolarizing}
                )
        
        return CPTPUpdater(
            noise_after_gate=noise_after_gate,
            global_depolarizing_rate=0.0
        )
    
    @classmethod
    def ibm_like(cls, error_rate: float = 0.001) -> NoiseModel:
        """Create a noise model similar to IBM quantum devices."""
        return cls(
            single_qubit_depolarizing=error_rate,
            two_qubit_depolarizing=error_rate * 10,  # Two-qubit gates are ~10x noisier
        )
    
    @classmethod
    def google_like(cls, error_rate: float = 0.001) -> NoiseModel:
        """Create a noise model similar to Google Sycamore."""
        return cls(
            single_qubit_depolarizing=error_rate * 0.5,
            two_qubit_depolarizing=error_rate * 5,
        )
