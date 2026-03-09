"""
ACES - Adaptive Causal-Entropy Simulation

A Pauli-based compiled classical emulator for NISQ-era quantum hardware.
"""

# Compiler
from aces.compiler import CircuitCompiler, CompiledCircuit, CompiledStep, GateOp, parse_circuit

# Runtime
from aces.runtime import (
    PauliState,
    EntanglementTracker,
    PruningEngine,
    DepolarizingNoise,
    DephasingNoise,
)

# Analysis
from aces.analysis import (
    compute_bloch_vectors,
    compute_purity,
    entanglement_matrix,
    compression_stats,
    ACESBenchmark,
)

# Visualization
from aces.visualization import print_bloch_vectors, print_entanglement_map

__version__ = "0.3.0"
