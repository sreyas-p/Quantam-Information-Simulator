"""
Causal Entanglement Graph (CEG) for ACES.

The CEG is the central data structure that captures which qubit pairs
have correlations that can causally influence measured observables.
It's constructed during compilation and guides which 2-qubit RDMs
need to be tracked during runtime.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, Optional, Iterator
from collections import defaultdict

import networkx as nx


@dataclass
class CEGEdge:
    """
    Edge in the Causal Entanglement Graph.
    
    Represents a tracked correlation between two qubits.
    
    Attributes:
        qubit_a: First qubit index
        qubit_b: Second qubit index  
        weight: Entanglement strength estimate (e.g., max possible concurrence)
        gate_indices: List of gate indices that contribute to this correlation
        is_measured: Whether either qubit is in the measurement set
    """
    qubit_a: int
    qubit_b: int
    weight: float = 1.0
    gate_indices: List[int] = field(default_factory=list)
    is_measured: bool = False
    
    @property
    def key(self) -> Tuple[int, int]:
        """Canonical (sorted) key for this edge."""
        return (min(self.qubit_a, self.qubit_b), max(self.qubit_a, self.qubit_b))
    
    def __hash__(self):
        return hash(self.key)
    
    def __eq__(self, other):
        if not isinstance(other, CEGEdge):
            return False
        return self.key == other.key


@dataclass
class CausalEntanglementGraph:
    """
    Causal Entanglement Graph (CEG).
    
    This graph tracks:
    - Which qubit pairs have correlations that matter for the measured observables
    - The strength of those correlations (for pruning decisions)
    - Which gates created which correlations
    
    The CEG is constructed during compilation based on:
    1. The circuit's two-qubit gates
    2. The forward lightcone from measured qubits
    3. Entanglement budget constraints
    
    Attributes:
        num_qubits: Total number of qubits in the circuit
        measured_qubits: Set of qubit indices that are measured
        edges: Dictionary mapping (q1, q2) -> CEGEdge
        entanglement_budget: Maximum total edge weight before pruning
    """
    num_qubits: int
    measured_qubits: Set[int] = field(default_factory=set)
    edges: Dict[Tuple[int, int], CEGEdge] = field(default_factory=dict)
    entanglement_budget: float = 100.0
    _graph: Optional[nx.Graph] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Build internal graph representation."""
        self._rebuild_graph()
    
    def _rebuild_graph(self):
        """Rebuild the networkx graph from edges."""
        self._graph = nx.Graph()
        self._graph.add_nodes_from(range(self.num_qubits))
        for (qa, qb), edge in self.edges.items():
            self._graph.add_edge(qa, qb, weight=edge.weight, edge=edge)
    
    def add_edge(
        self, 
        qubit_a: int, 
        qubit_b: int,
        weight: float = 1.0,
        gate_index: Optional[int] = None
    ) -> CEGEdge:
        """
        Add or update an edge in the CEG.
        
        Args:
            qubit_a: First qubit
            qubit_b: Second qubit
            weight: Entanglement strength
            gate_index: Index of gate that created this correlation
            
        Returns:
            The created or updated edge
        """
        key = (min(qubit_a, qubit_b), max(qubit_a, qubit_b))
        
        if key in self.edges:
            edge = self.edges[key]
            edge.weight = max(edge.weight, weight)  # Keep max weight
            if gate_index is not None:
                edge.gate_indices.append(gate_index)
        else:
            is_measured = (qubit_a in self.measured_qubits or 
                          qubit_b in self.measured_qubits)
            edge = CEGEdge(
                qubit_a=key[0],
                qubit_b=key[1],
                weight=weight,
                gate_indices=[gate_index] if gate_index is not None else [],
                is_measured=is_measured
            )
            self.edges[key] = edge
            
            # Update internal graph
            if self._graph is not None:
                self._graph.add_edge(key[0], key[1], weight=weight, edge=edge)
        
        return edge
    
    def remove_edge(self, qubit_a: int, qubit_b: int) -> Optional[CEGEdge]:
        """Remove an edge from the CEG."""
        key = (min(qubit_a, qubit_b), max(qubit_a, qubit_b))
        edge = self.edges.pop(key, None)
        if edge and self._graph is not None:
            self._graph.remove_edge(key[0], key[1])
        return edge
    
    def has_edge(self, qubit_a: int, qubit_b: int) -> bool:
        """Check if an edge exists."""
        key = (min(qubit_a, qubit_b), max(qubit_a, qubit_b))
        return key in self.edges
    
    def get_edge(self, qubit_a: int, qubit_b: int) -> Optional[CEGEdge]:
        """Get edge between two qubits."""
        key = (min(qubit_a, qubit_b), max(qubit_a, qubit_b))
        return self.edges.get(key)
    
    def neighbors(self, qubit: int) -> Set[int]:
        """Get all qubits connected to the given qubit."""
        if self._graph is None:
            self._rebuild_graph()
        return set(self._graph.neighbors(qubit))
    
    def degree(self, qubit: int) -> int:
        """Get the degree (number of correlations) for a qubit."""
        if self._graph is None:
            self._rebuild_graph()
        return self._graph.degree(qubit)
    
    @property
    def total_weight(self) -> float:
        """Total entanglement weight across all edges."""
        return sum(e.weight for e in self.edges.values())
    
    @property
    def num_edges(self) -> int:
        """Number of tracked correlations."""
        return len(self.edges)
    
    def edges_by_weight(self, ascending: bool = True) -> List[CEGEdge]:
        """Get edges sorted by weight."""
        return sorted(self.edges.values(), key=lambda e: e.weight, reverse=not ascending)
    
    def prune_to_budget(self, budget: Optional[float] = None) -> List[CEGEdge]:
        """
        Prune edges to stay within entanglement budget.
        
        Removes lowest-weight edges (that don't touch measured qubits)
        until total weight is under budget.
        
        Args:
            budget: Budget to use (defaults to self.entanglement_budget)
            
        Returns:
            List of removed edges
        """
        budget = budget or self.entanglement_budget
        removed = []
        
        while self.total_weight > budget:
            # Find lowest-weight non-measured edge
            candidates = [e for e in self.edges.values() if not e.is_measured]
            if not candidates:
                break  # Can't prune measured edges
            
            weakest = min(candidates, key=lambda e: e.weight)
            self.remove_edge(weakest.qubit_a, weakest.qubit_b)
            removed.append(weakest)
        
        return removed
    
    def connected_component(self, qubit: int) -> Set[int]:
        """Get all qubits in the same connected component."""
        if self._graph is None:
            self._rebuild_graph()
        return set(nx.node_connected_component(self._graph, qubit))
    
    def causal_cone_to_measured(self) -> Set[int]:
        """
        Find all qubits that can causally influence measured qubits
        through the correlation structure.
        """
        if not self.measured_qubits:
            return set()
        
        # BFS from measured qubits through the CEG
        visited = set(self.measured_qubits)
        frontier = list(self.measured_qubits)
        
        while frontier:
            current = frontier.pop(0)
            for neighbor in self.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
        
        return visited
    
    def subgraph(self, qubits: Set[int]) -> CausalEntanglementGraph:
        """Create a subgraph containing only specified qubits."""
        new_edges = {}
        for key, edge in self.edges.items():
            if key[0] in qubits and key[1] in qubits:
                new_edges[key] = CEGEdge(
                    qubit_a=edge.qubit_a,
                    qubit_b=edge.qubit_b,
                    weight=edge.weight,
                    gate_indices=edge.gate_indices.copy(),
                    is_measured=edge.is_measured
                )
        
        return CausalEntanglementGraph(
            num_qubits=self.num_qubits,
            measured_qubits=self.measured_qubits & qubits,
            edges=new_edges,
            entanglement_budget=self.entanglement_budget
        )
    
    def iter_edges(self) -> Iterator[CEGEdge]:
        """Iterate over all edges."""
        return iter(self.edges.values())
    
    def to_adjacency_matrix(self) -> np.ndarray:
        """Convert to weighted adjacency matrix."""
        adj = np.zeros((self.num_qubits, self.num_qubits))
        for (qa, qb), edge in self.edges.items():
            adj[qa, qb] = edge.weight
            adj[qb, qa] = edge.weight
        return adj
    
    def copy(self) -> CausalEntanglementGraph:
        """Create a deep copy."""
        new_edges = {
            k: CEGEdge(
                qubit_a=e.qubit_a,
                qubit_b=e.qubit_b,
                weight=e.weight,
                gate_indices=e.gate_indices.copy(),
                is_measured=e.is_measured
            )
            for k, e in self.edges.items()
        }
        
        return CausalEntanglementGraph(
            num_qubits=self.num_qubits,
            measured_qubits=self.measured_qubits.copy(),
            edges=new_edges,
            entanglement_budget=self.entanglement_budget
        )
    
    def __repr__(self) -> str:
        return (f"CausalEntanglementGraph(qubits={self.num_qubits}, "
                f"edges={self.num_edges}, measured={len(self.measured_qubits)}, "
                f"total_weight={self.total_weight:.2f})")


@dataclass
class CEGBuilder:
    """
    Builder for constructing a CEG from circuit analysis.
    
    This is used during compilation to determine which correlations
    need to be tracked based on the circuit structure and measurement specification.
    """
    num_qubits: int
    measured_qubits: Set[int]
    entanglement_budget: float = 100.0
    
    def build_from_gates(
        self,
        two_qubit_gates: List[Tuple[int, int, int]],  # (gate_idx, qubit_a, qubit_b)
        relevant_gates: Optional[Set[int]] = None
    ) -> CausalEntanglementGraph:
        """
        Build CEG from list of two-qubit gate interactions.
        
        Args:
            two_qubit_gates: List of (gate_index, qubit_a, qubit_b)
            relevant_gates: Optional set of gate indices in the lightcone
                          (if None, all gates are included)
                          
        Returns:
            Constructed CausalEntanglementGraph
        """
        ceg = CausalEntanglementGraph(
            num_qubits=self.num_qubits,
            measured_qubits=self.measured_qubits,
            entanglement_budget=self.entanglement_budget
        )
        
        for gate_idx, qa, qb in two_qubit_gates:
            if relevant_gates is None or gate_idx in relevant_gates:
                # Weight could be refined based on gate type
                ceg.add_edge(qa, qb, weight=1.0, gate_index=gate_idx)
        
        return ceg
    
    def build_from_circuit_dag(
        self,
        gate_sequence: List[dict],  # From circuit parser
        lightcone_gates: Set[int]
    ) -> CausalEntanglementGraph:
        """
        Build CEG from parsed circuit with lightcone analysis.
        
        Args:
            gate_sequence: List of gate dictionaries from parser
            lightcone_gates: Set of gate indices in the causal lightcone
            
        Returns:
            Constructed CausalEntanglementGraph
        """
        two_qubit_gates = []
        
        for idx, gate in enumerate(gate_sequence):
            if len(gate.get("qubits", [])) == 2:
                qa, qb = gate["qubits"]
                two_qubit_gates.append((idx, qa, qb))
        
        return self.build_from_gates(two_qubit_gates, lightcone_gates)
