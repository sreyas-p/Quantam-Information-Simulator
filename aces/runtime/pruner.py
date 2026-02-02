"""
Correlation pruning for ACES runtime.

Removes weak correlations to maintain tractability while preserving
accuracy on measured observables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, Optional
import numpy as np

from aces.core.density_matrix import SingleQubitRDM, TwoQubitRDM
from aces.core.ceg import CausalEntanglementGraph
from aces.compiler.budget import EntanglementBudget


@dataclass
class PruneResult:
    """Result of a pruning operation."""
    num_pruned: int
    pruned_pairs: List[Tuple[int, int]]
    total_weight_removed: float
    remaining_correlators: int


@dataclass
class CorrelationPruner:
    """
    Prunes weak correlations from the RDM state.
    
    This is essential for maintaining tractability in ACES. When
    correlations fall below threshold (either due to noise or
    distance from measurements), they are replaced with product
    states of the marginals.
    
    Attributes:
        budget: Entanglement budget configuration
        concurrence_threshold: Minimum concurrence to keep a correlator
        mutual_info_threshold: Minimum mutual information threshold
        prune_frequency: How often to run full pruning (every N gates)
    """
    budget: EntanglementBudget = field(default_factory=EntanglementBudget)
    concurrence_threshold: float = 1e-4
    mutual_info_threshold: float = 1e-4
    prune_frequency: int = 10
    
    _gate_counter: int = field(default=0, repr=False)
    
    def should_prune(self) -> bool:
        """Check if it's time for a pruning pass."""
        return self._gate_counter >= self.prune_frequency
    
    def tick(self) -> None:
        """Increment gate counter."""
        self._gate_counter += 1
    
    def reset_counter(self) -> None:
        """Reset gate counter after pruning."""
        self._gate_counter = 0
    
    def prune(
        self,
        state: Dict[int, SingleQubitRDM],
        correlators: Dict[Tuple[int, int], TwoQubitRDM],
        ceg: CausalEntanglementGraph,
        measured_qubits: Set[int]
    ) -> PruneResult:
        """
        Prune weak correlations from the state.
        
        Correlations are pruned if:
        1. Their concurrence is below threshold
        2. They don't involve measured qubits (with lower threshold)
        3. The budget is exceeded (weakest first)
        
        Args:
            state: Single-qubit RDM dictionary
            correlators: Two-qubit RDM dictionary
            ceg: Causal Entanglement Graph
            measured_qubits: Set of measured qubit indices
            
        Returns:
            PruneResult with statistics
        """
        self.reset_counter()
        
        pruned_pairs = []
        total_weight = 0.0
        
        # Evaluate all correlators
        to_remove = []
        correlator_scores = []
        
        for (qa, qb), corr in correlators.items():
            try:
                concurrence = corr.concurrence()
            except:
                concurrence = 0.0
            
            try:
                mi = corr.mutual_information()
            except:
                mi = 0.0
            
            # Determine threshold based on whether qubits are measured
            involves_measured = qa in measured_qubits or qb in measured_qubits
            
            if involves_measured:
                threshold = self.concurrence_threshold / 10  # Lower threshold for measured
            else:
                threshold = self.concurrence_threshold
            
            # Score for prioritizing pruning (lower = prune first)
            score = concurrence + mi + (10.0 if involves_measured else 0.0)
            correlator_scores.append(((qa, qb), score, concurrence))
            
            # Mark for removal if below threshold
            if concurrence < threshold and mi < self.mutual_info_threshold:
                to_remove.append((qa, qb))
        
        # If budget exceeded, also remove lowest-scoring correlators
        if self.budget.is_exceeded:
            # Sort by score (ascending)
            correlator_scores.sort(key=lambda x: x[1])
            
            while self.budget.is_exceeded and correlator_scores:
                pair, score, _ = correlator_scores.pop(0)
                if pair not in to_remove:
                    to_remove.append(pair)
        
        # Perform removals
        for (qa, qb) in to_remove:
            if (qa, qb) in correlators:
                corr = correlators[(qa, qb)]
                
                # Update marginals before removing
                if qa in state:
                    new_marginal_a = corr.marginal_a()
                    new_marginal_a.qubit_id = qa
                    state[qa] = new_marginal_a
                if qb in state:
                    new_marginal_b = corr.marginal_b()
                    new_marginal_b.qubit_id = qb
                    state[qb] = new_marginal_b
                
                # Get weight for budget tracking
                edge = ceg.get_edge(qa, qb)
                weight = edge.weight if edge else 1.0
                total_weight += weight
                
                # Remove from correlators and CEG
                del correlators[(qa, qb)]
                ceg.remove_edge(qa, qb)
                pruned_pairs.append((qa, qb))
        
        # Update budget
        self.budget.release(len(pruned_pairs), total_weight)
        
        return PruneResult(
            num_pruned=len(pruned_pairs),
            pruned_pairs=pruned_pairs,
            total_weight_removed=total_weight,
            remaining_correlators=len(correlators)
        )
    
    def quick_prune(
        self,
        correlators: Dict[Tuple[int, int], TwoQubitRDM],
        ceg: CausalEntanglementGraph
    ) -> int:
        """
        Quick pruning pass that only checks purity.
        
        Much faster than full prune - use between gates.
        Returns number of correlators pruned.
        """
        to_remove = []
        
        for (qa, qb), corr in correlators.items():
            # If nearly product state, remove
            if corr.is_product_state(tol=self.concurrence_threshold):
                to_remove.append((qa, qb))
        
        for (qa, qb) in to_remove:
            del correlators[(qa, qb)]
            ceg.remove_edge(qa, qb)
        
        return len(to_remove)


@dataclass
class AdaptivePruner(CorrelationPruner):
    """
    Adaptive pruner that adjusts thresholds based on budget usage.
    """
    
    def adaptive_threshold(self) -> float:
        """Get current threshold based on budget usage."""
        usage = max(self.budget.correlator_usage, self.budget.weight_usage)
        
        if usage < 0.5:
            # Under half budget, use loose threshold
            return self.concurrence_threshold / 10
        elif usage < 0.8:
            # Getting close, use normal threshold
            return self.concurrence_threshold
        else:
            # Near or over budget, aggressive pruning
            return self.concurrence_threshold * 10
    
    def prune(
        self,
        state: Dict[int, SingleQubitRDM],
        correlators: Dict[Tuple[int, int], TwoQubitRDM],
        ceg: CausalEntanglementGraph,
        measured_qubits: Set[int]
    ) -> PruneResult:
        """Prune with adaptive thresholds."""
        # Temporarily adjust threshold
        original_threshold = self.concurrence_threshold
        self.concurrence_threshold = self.adaptive_threshold()
        
        result = super().prune(state, correlators, ceg, measured_qubits)
        
        # Restore threshold
        self.concurrence_threshold = original_threshold
        
        return result
