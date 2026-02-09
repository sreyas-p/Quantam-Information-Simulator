"""Causal Entanglement Skeleton (CES) for tracking qubit correlations."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, Optional, Iterator

import networkx as nx


@dataclass
class CESEdge:
    """Edge representing a retained correlator between two qubits."""
    qubit_a: int
    qubit_b: int
    weight: float = 1.0
    noise_decay_estimate: float = 1.0
    gate_indices: List[int] = field(default_factory=list)
    is_measured: bool = False
    depth_at_creation: int = 0
    
    @property
    def effective_weight(self) -> float:
        return self.weight * self.noise_decay_estimate
    
    @property
    def key(self) -> Tuple[int, int]:
        return (min(self.qubit_a, self.qubit_b), max(self.qubit_a, self.qubit_b))
    
    def decay_by(self, factor: float) -> None:
        self.noise_decay_estimate *= factor
    
    def __hash__(self):
        return hash(self.key)
    
    def __eq__(self, other):
        if not isinstance(other, CESEdge):
            return False
        return self.key == other.key


CEGEdge = CESEdge


@dataclass
class CausalEntanglementGraph:
    """Graph tracking correlations that influence measured observables."""
    num_qubits: int
    measured_qubits: Set[int] = field(default_factory=set)
    edges: Dict[Tuple[int, int], CESEdge] = field(default_factory=dict)
    entanglement_budget: float = 100.0
    _graph: Optional[nx.Graph] = field(default=None, repr=False)
    
    def __post_init__(self):
        self._rebuild_graph()
    
    def _rebuild_graph(self):
        self._graph = nx.Graph()
        self._graph.add_nodes_from(range(self.num_qubits))
        for (qa, qb), edge in self.edges.items():
            self._graph.add_edge(qa, qb, weight=edge.weight, edge=edge)
    
    def add_edge(self, qubit_a: int, qubit_b: int, weight: float = 1.0,
                 gate_index: Optional[int] = None) -> CESEdge:
        key = (min(qubit_a, qubit_b), max(qubit_a, qubit_b))
        if key in self.edges:
            edge = self.edges[key]
            edge.weight = max(edge.weight, weight)
            if gate_index is not None:
                edge.gate_indices.append(gate_index)
        else:
            is_measured = qubit_a in self.measured_qubits or qubit_b in self.measured_qubits
            edge = CESEdge(
                qubit_a=key[0], qubit_b=key[1], weight=weight,
                gate_indices=[gate_index] if gate_index is not None else [],
                is_measured=is_measured
            )
            self.edges[key] = edge
            if self._graph is not None:
                self._graph.add_edge(key[0], key[1], weight=weight, edge=edge)
        return edge
    
    def remove_edge(self, qubit_a: int, qubit_b: int) -> Optional[CESEdge]:
        key = (min(qubit_a, qubit_b), max(qubit_a, qubit_b))
        edge = self.edges.pop(key, None)
        if edge and self._graph is not None:
            self._graph.remove_edge(key[0], key[1])
        return edge
    
    def has_edge(self, qubit_a: int, qubit_b: int) -> bool:
        key = (min(qubit_a, qubit_b), max(qubit_a, qubit_b))
        return key in self.edges
    
    def get_edge(self, qubit_a: int, qubit_b: int) -> Optional[CESEdge]:
        key = (min(qubit_a, qubit_b), max(qubit_a, qubit_b))
        return self.edges.get(key)
    
    def neighbors(self, qubit: int) -> Set[int]:
        if self._graph is None:
            self._rebuild_graph()
        return set(self._graph.neighbors(qubit))
    
    def degree(self, qubit: int) -> int:
        if self._graph is None:
            self._rebuild_graph()
        return self._graph.degree(qubit)
    
    @property
    def total_weight(self) -> float:
        return sum(e.weight for e in self.edges.values())
    
    @property
    def num_edges(self) -> int:
        return len(self.edges)
    
    def edges_by_weight(self, ascending: bool = True) -> List[CESEdge]:
        return sorted(self.edges.values(), key=lambda e: e.weight, reverse=not ascending)
    
    def prune_to_budget(self, budget: Optional[float] = None) -> List[CESEdge]:
        budget = budget or self.entanglement_budget
        removed = []
        while self.total_weight > budget:
            candidates = [e for e in self.edges.values() if not e.is_measured]
            if not candidates:
                break
            weakest = min(candidates, key=lambda e: e.weight)
            self.remove_edge(weakest.qubit_a, weakest.qubit_b)
            removed.append(weakest)
        return removed
    
    def connected_component(self, qubit: int) -> Set[int]:
        if self._graph is None:
            self._rebuild_graph()
        return set(nx.node_connected_component(self._graph, qubit))
    
    def causal_cone_to_measured(self) -> Set[int]:
        if not self.measured_qubits:
            return set()
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
        new_edges = {}
        for key, edge in self.edges.items():
            if key[0] in qubits and key[1] in qubits:
                new_edges[key] = CESEdge(
                    qubit_a=edge.qubit_a, qubit_b=edge.qubit_b,
                    weight=edge.weight, gate_indices=edge.gate_indices.copy(),
                    is_measured=edge.is_measured
                )
        return CausalEntanglementGraph(
            num_qubits=self.num_qubits,
            measured_qubits=self.measured_qubits & qubits,
            edges=new_edges,
            entanglement_budget=self.entanglement_budget
        )
    
    def iter_edges(self) -> Iterator[CESEdge]:
        return iter(self.edges.values())
    
    def to_adjacency_matrix(self) -> np.ndarray:
        adj = np.zeros((self.num_qubits, self.num_qubits))
        for (qa, qb), edge in self.edges.items():
            adj[qa, qb] = edge.weight
            adj[qb, qa] = edge.weight
        return adj
    
    def copy(self) -> CausalEntanglementGraph:
        new_edges = {
            k: CESEdge(
                qubit_a=e.qubit_a, qubit_b=e.qubit_b, weight=e.weight,
                gate_indices=e.gate_indices.copy(), is_measured=e.is_measured
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
        return f"CausalEntanglementGraph(qubits={self.num_qubits}, edges={self.num_edges})"


@dataclass
class CEGBuilder:
    """Builder for constructing CES from circuit analysis."""
    num_qubits: int
    measured_qubits: Set[int]
    entanglement_budget: float = 100.0
    
    def build_from_gates(self, two_qubit_gates: List[Tuple[int, int, int]],
                         relevant_gates: Optional[Set[int]] = None) -> CausalEntanglementGraph:
        ceg = CausalEntanglementGraph(
            num_qubits=self.num_qubits,
            measured_qubits=self.measured_qubits,
            entanglement_budget=self.entanglement_budget
        )
        for gate_idx, qa, qb in two_qubit_gates:
            if relevant_gates is None or gate_idx in relevant_gates:
                ceg.add_edge(qa, qb, weight=1.0, gate_index=gate_idx)
        return ceg
    
    def build_from_circuit_dag(self, gate_sequence: List[dict],
                                lightcone_gates: Set[int]) -> CausalEntanglementGraph:
        two_qubit_gates = []
        for idx, gate in enumerate(gate_sequence):
            if len(gate.get("qubits", [])) == 2:
                qa, qb = gate["qubits"]
                two_qubit_gates.append((idx, qa, qb))
        return self.build_from_gates(two_qubit_gates, lightcone_gates)
