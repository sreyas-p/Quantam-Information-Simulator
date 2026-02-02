"""
Lightcone computation for ACES.

Computes the forward/backward causal cone from measured qubits to identify
which gates can influence the measurement outcomes. Gates outside this
cone can be safely ignored during simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict


@dataclass
class GateNode:
    """
    Represents a gate in the circuit DAG.
    
    Attributes:
        index: Position in the gate sequence
        name: Gate name (e.g., "CNOT", "H")
        qubits: Tuple of qubit indices this gate acts on
        params: Optional parameters for parametric gates
        depth: Circuit depth at this gate
    """
    index: int
    name: str
    qubits: Tuple[int, ...]
    params: Dict[str, float] = field(default_factory=dict)
    depth: int = 0


@dataclass
class CircuitDAG:
    """
    Directed Acyclic Graph representation of a quantum circuit.
    
    The DAG captures the temporal/causal order of gates:
    - Nodes are gates
    - Directed edges represent "must happen before" relationships
    - Gates on the same qubit must be ordered
    - Independent gates on different qubits can be parallelized
    """
    gates: List[GateNode] = field(default_factory=list)
    num_qubits: int = 0
    
    # Adjacency lists for the DAG
    _predecessors: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    _successors: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    
    # Track last gate on each qubit for building dependencies
    _last_gate_on_qubit: Dict[int, int] = field(default_factory=dict)
    
    def add_gate(
        self,
        name: str,
        qubits: Tuple[int, ...],
        params: Optional[Dict[str, float]] = None
    ) -> GateNode:
        """
        Add a gate to the circuit DAG.
        
        Automatically creates edges to previous gates on the same qubits.
        """
        index = len(self.gates)
        
        # Find predecessors (last gate on each qubit this gate touches)
        predecessors = set()
        max_pred_depth = -1
        
        for q in qubits:
            if q in self._last_gate_on_qubit:
                pred_idx = self._last_gate_on_qubit[q]
                predecessors.add(pred_idx)
                max_pred_depth = max(max_pred_depth, self.gates[pred_idx].depth)
            self.num_qubits = max(self.num_qubits, q + 1)
        
        node = GateNode(
            index=index,
            name=name,
            qubits=qubits,
            params=params or {},
            depth=max_pred_depth + 1
        )
        self.gates.append(node)
        
        # Update adjacency
        self._predecessors[index] = predecessors
        for pred in predecessors:
            self._successors[pred].add(index)
        
        # Update last gate tracking
        for q in qubits:
            self._last_gate_on_qubit[q] = index
        
        return node
    
    def predecessors(self, gate_idx: int) -> Set[int]:
        """Get indices of gates that must happen before this gate."""
        return self._predecessors.get(gate_idx, set())
    
    def successors(self, gate_idx: int) -> Set[int]:
        """Get indices of gates that must happen after this gate."""
        return self._successors.get(gate_idx, set())
    
    def get_output_gates(self) -> List[int]:
        """Get indices of gates with no successors (circuit outputs)."""
        return [i for i in range(len(self.gates)) if not self._successors.get(i)]
    
    def get_input_gates(self) -> List[int]:
        """Get indices of gates with no predecessors (circuit inputs)."""
        return [i for i in range(len(self.gates)) if not self._predecessors.get(i)]
    
    @property
    def depth(self) -> int:
        """Maximum circuit depth."""
        if not self.gates:
            return 0
        return max(g.depth for g in self.gates) + 1
    
    def topological_order(self) -> List[int]:
        """Return gate indices in topological (executable) order."""
        # Already in topological order since we add gates sequentially
        return list(range(len(self.gates)))


def compute_lightcone(
    dag: CircuitDAG,
    measured_qubits: Set[int],
    direction: str = "backward"
) -> Set[int]:
    """
    Compute the causal lightcone for measured qubits.
    
    Args:
        dag: Circuit DAG
        measured_qubits: Set of qubit indices that are measured
        direction: "backward" (default) to find all gates that can influence
                  measurements, "forward" to find all gates influenced by
                  initial state preparation
                  
    Returns:
        Set of gate indices in the lightcone
    """
    if direction == "backward":
        return _backward_lightcone(dag, measured_qubits)
    elif direction == "forward":
        return _forward_lightcone(dag, measured_qubits)
    else:
        raise ValueError(f"Unknown direction: {direction}")


def _backward_lightcone(dag: CircuitDAG, measured_qubits: Set[int]) -> Set[int]:
    """
    Compute backward lightcone: all gates that can influence measured qubits.
    
    Algorithm:
    1. Start with final gates on measured qubits
    2. Traverse backwards through the DAG
    3. Include all predecessors of included gates
    """
    if not measured_qubits:
        return set()
    
    # Find final gates on measured qubits
    final_gates_on_qubits = {}
    for gate in dag.gates:
        for q in gate.qubits:
            if q in measured_qubits:
                final_gates_on_qubits[q] = gate.index
    
    # Initialize with final gates on measured qubits
    lightcone = set(final_gates_on_qubits.values())
    frontier = list(lightcone)
    
    # BFS backwards through the circuit
    while frontier:
        current = frontier.pop(0)
        
        # Also include any gate that shares a qubit with current gate
        # (more conservative, but ensures we capture all causal dependencies)
        current_gate = dag.gates[current]
        
        for pred_idx in dag.predecessors(current):
            if pred_idx not in lightcone:
                lightcone.add(pred_idx)
                frontier.append(pred_idx)
        
        # Include predecessors of predecessors that touch same qubits
        # This handles cases where entanglement propagates
        for q in current_gate.qubits:
            for gate in dag.gates[:current]:
                if q in gate.qubits and gate.index not in lightcone:
                    lightcone.add(gate.index)
                    frontier.append(gate.index)
    
    return lightcone


def _forward_lightcone(dag: CircuitDAG, initial_qubits: Set[int]) -> Set[int]:
    """
    Compute forward lightcone: all gates influenced by initial qubits.
    
    This is useful for determining how information spreads from
    specific initial states.
    """
    if not initial_qubits:
        return set()
    
    # Find first gates on initial qubits
    first_gates_on_qubits = {}
    for qubit in initial_qubits:
        for gate in dag.gates:
            if qubit in gate.qubits:
                first_gates_on_qubits[qubit] = gate.index
                break
    
    # Initialize with first gates on initial qubits
    lightcone = set(first_gates_on_qubits.values())
    frontier = list(lightcone)
    
    # BFS forward through the circuit
    while frontier:
        current = frontier.pop(0)
        
        for succ_idx in dag.successors(current):
            if succ_idx not in lightcone:
                lightcone.add(succ_idx)
                frontier.append(succ_idx)
    
    return lightcone


def compute_qubit_lightcone(
    dag: CircuitDAG,
    measured_qubits: Set[int]
) -> Set[int]:
    """
    Find all qubits that can influence measured qubits.
    
    This is the transitive closure of qubit correlations.
    """
    gate_lightcone = compute_lightcone(dag, measured_qubits, direction="backward")
    
    relevant_qubits = set(measured_qubits)
    for gate_idx in gate_lightcone:
        relevant_qubits.update(dag.gates[gate_idx].qubits)
    
    return relevant_qubits


@dataclass
class LightconeAnalysis:
    """
    Complete lightcone analysis results.
    
    Attributes:
        measured_qubits: The qubits being measured
        relevant_gates: Gate indices in the backward lightcone
        relevant_qubits: All qubits that can influence measurements
        pruned_gates: Gate indices that can be ignored
        two_qubit_gate_pairs: List of (gate_idx, qubit_a, qubit_b) for 2-qubit gates
    """
    measured_qubits: Set[int]
    relevant_gates: Set[int]
    relevant_qubits: Set[int]
    pruned_gates: Set[int]
    two_qubit_gate_pairs: List[Tuple[int, int, int]]
    
    @property
    def num_gates_total(self) -> int:
        return len(self.relevant_gates) + len(self.pruned_gates)
    
    @property
    def pruning_ratio(self) -> float:
        """Fraction of gates pruned."""
        total = self.num_gates_total
        if total == 0:
            return 0.0
        return len(self.pruned_gates) / total
    
    def __repr__(self) -> str:
        return (f"LightconeAnalysis(measured={len(self.measured_qubits)}, "
                f"relevant_gates={len(self.relevant_gates)}/{self.num_gates_total}, "
                f"pruned={self.pruning_ratio:.1%})")


def analyze_lightcone(
    dag: CircuitDAG,
    measured_qubits: Set[int]
) -> LightconeAnalysis:
    """
    Perform complete lightcone analysis on a circuit.
    
    Args:
        dag: Circuit DAG
        measured_qubits: Set of measured qubit indices
        
    Returns:
        LightconeAnalysis with all relevant information
    """
    relevant_gates = compute_lightcone(dag, measured_qubits, direction="backward")
    relevant_qubits = compute_qubit_lightcone(dag, measured_qubits)
    
    all_gates = set(range(len(dag.gates)))
    pruned_gates = all_gates - relevant_gates
    
    # Extract two-qubit gate pairs in the lightcone
    two_qubit_pairs = []
    for gate_idx in relevant_gates:
        gate = dag.gates[gate_idx]
        if len(gate.qubits) == 2:
            two_qubit_pairs.append((gate_idx, gate.qubits[0], gate.qubits[1]))
    
    # Sort by gate index for consistent ordering
    two_qubit_pairs.sort(key=lambda x: x[0])
    
    return LightconeAnalysis(
        measured_qubits=measured_qubits,
        relevant_gates=relevant_gates,
        relevant_qubits=relevant_qubits,
        pruned_gates=pruned_gates,
        two_qubit_gate_pairs=two_qubit_pairs
    )
