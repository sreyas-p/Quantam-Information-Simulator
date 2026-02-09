"""
ACES - Adaptive Causal-Entropy Simulation

A compiled classical emulator for NISQ-era quantum hardware.

Step 1: Core foundational data structures and mathematical definitions.
"""

from aces.core import (
    SingleQubitRDM,
    TwoQubitRDM,
    CorrelatorStorage,
    CausalEntanglementGraph,
    CESEdge,
    CEGEdge,
    Gate,
    GateLibrary,
    CompileTimeInput,
    RuntimeInput,
    RuntimeOutput,
    MeasurementSpec,
    NoiseModelSpec,
)

__version__ = "0.1.0"
__all__ = [
    "SingleQubitRDM",
    "TwoQubitRDM",
    "CorrelatorStorage",
    "CausalEntanglementGraph",
    "CESEdge",
    "CEGEdge",
    "Gate",
    "GateLibrary",
    "CompileTimeInput",
    "RuntimeInput",
    "RuntimeOutput",
    "MeasurementSpec",
    "NoiseModelSpec",
]
