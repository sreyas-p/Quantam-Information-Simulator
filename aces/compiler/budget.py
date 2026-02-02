"""
Entanglement budget management for ACES.

Controls how much entanglement (tracked correlations) can be maintained
during simulation. When the budget is exceeded, correlations are pruned
to maintain tractability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np

from aces.core.ceg import CausalEntanglementGraph


@dataclass
class EntanglementBudget:
    """
    Manages the entanglement budget for an ACES simulation.
    
    The budget limits the total "entanglement cost" tracked during simulation.
    This cost can be measured in various ways:
    - Number of 2-qubit correlators tracked
    - Sum of concurrence/mutual information
    - Weighted sum based on distance from measured qubits
    
    When the budget is exceeded, the pruner removes the lowest-contribution
    correlations.
    
    Attributes:
        max_correlators: Maximum number of 2-qubit RDMs to track
        max_weight: Maximum total edge weight in CEG
        distance_decay: Factor for reducing weight with distance from measurement
        min_concurrence: Minimum concurrence to keep a correlator
    """
    max_correlators: int = 1000
    max_weight: float = 100.0
    distance_decay: float = 0.9
    min_concurrence: float = 1e-6
    
    # Tracking current usage
    _current_correlators: int = field(default=0, repr=False)
    _current_weight: float = field(default=0.0, repr=False)
    
    @property
    def correlator_usage(self) -> float:
        """Fraction of correlator budget used."""
        return self._current_correlators / self.max_correlators
    
    @property
    def weight_usage(self) -> float:
        """Fraction of weight budget used."""
        return self._current_weight / self.max_weight
    
    @property
    def is_exceeded(self) -> bool:
        """Check if budget is exceeded."""
        return (self._current_correlators > self.max_correlators or
                self._current_weight > self.max_weight)
    
    def update_from_ceg(self, ceg: CausalEntanglementGraph) -> None:
        """Update budget tracking from CEG state."""
        self._current_correlators = ceg.num_edges
        self._current_weight = ceg.total_weight
    
    def should_track(
        self,
        qubit_a: int,
        qubit_b: int,
        estimated_entanglement: float,
        distance_from_measurement: int = 0
    ) -> bool:
        """
        Decide whether to create/maintain a correlator for this qubit pair.
        
        Args:
            qubit_a: First qubit
            qubit_b: Second qubit
            estimated_entanglement: Estimated entanglement strength
            distance_from_measurement: Shortest path to a measured qubit
            
        Returns:
            True if the correlator should be tracked
        """
        # Always track if we're under budget
        if not self.is_exceeded:
            return True
        
        # Weight by distance from measurement
        effective_weight = estimated_entanglement * (
            self.distance_decay ** distance_from_measurement
        )
        
        return effective_weight >= self.min_concurrence
    
    def allocate(self, n_correlators: int, weight: float) -> bool:
        """
        Try to allocate budget for new correlators.
        
        Returns True if allocation succeeds, False otherwise.
        """
        if (self._current_correlators + n_correlators > self.max_correlators or
            self._current_weight + weight > self.max_weight):
            return False
        
        self._current_correlators += n_correlators
        self._current_weight += weight
        return True
    
    def release(self, n_correlators: int, weight: float) -> None:
        """Release budget from pruned correlators."""
        self._current_correlators = max(0, self._current_correlators - n_correlators)
        self._current_weight = max(0, self._current_weight - weight)
    
    def reset(self) -> None:
        """Reset budget tracking."""
        self._current_correlators = 0
        self._current_weight = 0.0


@dataclass
class AdaptiveBudget(EntanglementBudget):
    """
    Adaptive budget that adjusts based on simulation progress.
    
    Starts with a higher budget and reduces it as noise accumulates,
    since noise naturally reduces correlations.
    """
    initial_multiplier: float = 2.0
    decay_per_layer: float = 0.95
    _layer_count: int = field(default=0, repr=False)
    
    @property
    def effective_max_correlators(self) -> int:
        """Current effective correlator limit."""
        multiplier = self.initial_multiplier * (self.decay_per_layer ** self._layer_count)
        return int(self.max_correlators * max(multiplier, 1.0))
    
    @property
    def effective_max_weight(self) -> float:
        """Current effective weight limit."""
        multiplier = self.initial_multiplier * (self.decay_per_layer ** self._layer_count)
        return self.max_weight * max(multiplier, 1.0)
    
    @property
    def is_exceeded(self) -> bool:
        """Check if effective budget is exceeded."""
        return (self._current_correlators > self.effective_max_correlators or
                self._current_weight > self.effective_max_weight)
    
    def advance_layer(self) -> None:
        """Call when simulation advances to next circuit layer."""
        self._layer_count += 1


def estimate_budget_for_circuit(
    num_qubits: int,
    depth: int,
    connectivity: str = "linear",
    measured_fraction: float = 0.1
) -> EntanglementBudget:
    """
    Estimate appropriate budget for a circuit.
    
    Args:
        num_qubits: Number of qubits
        depth: Circuit depth
        connectivity: "linear", "grid", or "all-to-all"
        measured_fraction: Fraction of qubits being measured
        
    Returns:
        Configured EntanglementBudget
    """
    # Heuristic: correlators scale with locality
    if connectivity == "linear":
        # Linear: correlations decay with distance
        max_correlators = min(num_qubits * 5, 500)
        max_weight = num_qubits * 2
    elif connectivity == "grid":
        # 2D grid: more correlations
        max_correlators = min(num_qubits * 10, 1000)
        max_weight = num_qubits * 4
    else:
        # All-to-all: potentially many correlations
        max_correlators = min(num_qubits * (num_qubits - 1) // 4, 2000)
        max_weight = num_qubits * 10
    
    # Scale down if measuring few qubits
    scale = max(measured_fraction, 0.1)
    max_correlators = int(max_correlators * scale)
    max_weight = max_weight * scale
    
    return EntanglementBudget(
        max_correlators=max_correlators,
        max_weight=max_weight
    )
