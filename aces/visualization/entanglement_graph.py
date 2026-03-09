"""Entanglement Graph Viewer — text-based heatmap of pairwise correlations."""

from __future__ import annotations

import numpy as np
from typing import Optional

from aces.runtime.pauli_state import PauliState
from aces.analysis.metrics import entanglement_matrix


def print_entanglement_map(state: PauliState, label: Optional[str] = None) -> None:
    """Print a text-based entanglement heatmap.

    Uses block characters to show correlation strength between qubit pairs.
    """
    n = state.num_qubits
    mat = entanglement_matrix(state)

    if label:
        print(f"\n  {label}")

    # Header
    header = "      " + "".join(f"  q{j:<3}" for j in range(n))
    print(header)

    # Rows
    max_val = mat.max() if mat.max() > 0 else 1.0
    blocks = " ░▒▓█"

    for i in range(n):
        row = f"  q{i:<3}"
        for j in range(n):
            if i == j:
                row += "  ·   "
            else:
                val = mat[i, j] / max_val
                idx = min(int(val * (len(blocks) - 1)), len(blocks) - 1)
                row += f" {blocks[idx]}{val:4.2f}"
        print(row)

    print(f"\n  Peak correlation: {max_val:.3f}")
