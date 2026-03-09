"""Bloch Vector Viewer — text-based display of per-qubit Bloch vectors."""

from __future__ import annotations

import numpy as np
from typing import Optional

from aces.runtime.pauli_state import PauliState


def print_bloch_vectors(state: PauliState, label: Optional[str] = None) -> None:
    """Print formatted Bloch vectors for all qubits.

    Shows ⟨X⟩, ⟨Y⟩, ⟨Z⟩, radius, P(0), and a visual bar for radius.
    """
    if label:
        print(f"\n  {label}")
    print(f"  {'Qubit':<7} {'⟨X⟩':>7} {'⟨Y⟩':>7} {'⟨Z⟩':>7} {'|r|':>7} {'P(0)':>7}  Bloch")
    print(f"  {'-'*55}")

    for q in range(state.num_qubits):
        rx, ry, rz = state.bloch_vector(q)
        r = np.sqrt(rx**2 + ry**2 + rz**2)
        p0 = (1 + rz) / 2

        # Visual bar: ████░░░░░░ (filled proportional to radius)
        bar_len = 10
        filled = int(round(r * bar_len))
        bar = "█" * filled + "░" * (bar_len - filled)

        print(f"  q{q:<5} {rx:>+7.3f} {ry:>+7.3f} {rz:>+7.3f} {r:>7.3f} {p0:>7.3f}  {bar}")
