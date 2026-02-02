"""
ACES - Adaptive Causal-Entropy Simulation

A compiled classical emulator for NISQ-era quantum hardware.
"""

from aces.compiler.compile import compile_circuit
from aces.runtime.engine import ACESRuntime

__version__ = "0.1.0"
__all__ = ["compile_circuit", "ACESRuntime"]
