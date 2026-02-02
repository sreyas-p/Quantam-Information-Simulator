"""
ACES Runtime Engine.

The main execution engine that runs compiled ACES objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, Optional, Any
import numpy as np

from aces.core.density_matrix import SingleQubitRDM, TwoQubitRDM
from aces.core.ceg import CausalEntanglementGraph
from aces.core.gates import GateLibrary
from aces.compiler.compile import CompiledACES, FrozenGate
from aces.runtime.updater import CPTPUpdater, NoiseModel
from aces.runtime.pruner import CorrelationPruner, AdaptivePruner


@dataclass
class ExecutionResult:
    """
    Result of an ACES execution.
    
    Attributes:
        state: Final single-qubit RDMs
        correlators: Final two-qubit correlators
        expectation_values: Computed expectation values
        marginals: Marginal probability distributions
        samples: Sampled bitstrings (if requested)
        metadata: Execution metadata
    """
    state: Dict[int, SingleQubitRDM]
    correlators: Dict[Tuple[int, int], TwoQubitRDM]
    expectation_values: Dict[str, float] = field(default_factory=dict)
    marginals: Dict[int, np.ndarray] = field(default_factory=dict)
    samples: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def expectation(self, observable: str) -> float:
        """Get expectation value for an observable."""
        return self.expectation_values.get(observable, 0.0)
    
    def probability(self, qubit: int, outcome: int) -> float:
        """Get probability of measuring a specific outcome on a qubit."""
        if qubit not in self.marginals:
            if qubit in self.state:
                if outcome == 0:
                    return self.state[qubit].probability_zero()
                else:
                    return self.state[qubit].probability_one()
            return 0.5
        return self.marginals[qubit][outcome]


@dataclass
class ACESRuntime:
    """
    Runtime engine for executing compiled ACES objects.
    
    Usage:
        compiled = compile_circuit(circuit, measured_qubits={0, 1})
        runtime = ACESRuntime(compiled)
        result = runtime.execute(params={"theta": 0.5})
        print(result.expectation_values)
    
    Attributes:
        compiled: The compiled ACES object
        noise_model: Optional noise model
        pruner: Correlation pruner
        updater: CPTP map updater
    """
    compiled: CompiledACES
    noise_model: Optional[NoiseModel] = None
    pruner: Optional[CorrelationPruner] = None
    updater: Optional[CPTPUpdater] = None
    
    def __post_init__(self):
        """Initialize runtime components."""
        if self.noise_model is None:
            self.noise_model = NoiseModel()
        
        if self.updater is None:
            self.updater = self.noise_model.create_updater()
        
        if self.pruner is None:
            self.pruner = AdaptivePruner(budget=self.compiled.budget)
    
    def initialize_state(self) -> Tuple[Dict[int, SingleQubitRDM], 
                                        Dict[Tuple[int, int], TwoQubitRDM]]:
        """
        Initialize the RDM state to |0...0⟩.
        
        Returns:
            Tuple of (single-qubit RDMs, two-qubit correlators)
        """
        state = {}
        for q in range(self.compiled.num_qubits):
            state[q] = SingleQubitRDM.zero_state(q)
        
        # Initialize correlators as product states
        # (will be populated as 2-qubit gates are applied)
        correlators = {}
        
        return state, correlators
    
    def execute(
        self,
        params: Optional[Dict[str, float]] = None,
        observables: Optional[List[str]] = None,
        num_samples: int = 0,
        return_state: bool = True
    ) -> ExecutionResult:
        """
        Execute the compiled circuit.
        
        Args:
            params: Runtime parameters for parametric gates
            observables: List of observables to compute (e.g., ["Z0", "Z0Z1"])
                        If None, computes Z on all measured qubits.
            num_samples: Number of bitstrings to sample (0 = no sampling)
            return_state: Whether to include full state in result
            
        Returns:
            ExecutionResult with computed values
        """
        params = params or {}
        
        # Initialize state
        state, correlators = self.initialize_state()
        
        # Make working copy of CEG
        ceg = self.compiled.ceg.copy()
        
        # Execute gate sequence
        for frozen_gate in self.compiled.gate_sequence:
            self._apply_gate(frozen_gate, state, correlators, ceg, params)
            
            # Periodic pruning
            self.pruner.tick()
            if self.pruner.should_prune():
                self.pruner.prune(state, correlators, ceg, self.compiled.measured_qubits)
        
        # Compute observables
        if observables is None:
            # Default: Z on all measured qubits
            observables = [f"Z{q}" for q in sorted(self.compiled.measured_qubits)]
        
        expectation_values = self._compute_expectations(state, correlators, observables)
        
        # Compute marginals for measured qubits
        marginals = {}
        for q in self.compiled.measured_qubits:
            if q in state:
                marginals[q] = np.array([
                    state[q].probability_zero(),
                    state[q].probability_one()
                ])
        
        # Sample if requested
        samples = None
        if num_samples > 0:
            samples = self._sample_bitstrings(state, num_samples)
        
        # Build result
        result_state = state if return_state else {}
        result_correlators = correlators if return_state else {}
        
        return ExecutionResult(
            state=result_state,
            correlators=result_correlators,
            expectation_values=expectation_values,
            marginals=marginals,
            samples=samples,
            metadata={
                "num_gates_executed": len(self.compiled.gate_sequence),
                "final_correlators": len(correlators),
                "ceg_edges": ceg.num_edges,
            }
        )
    
    def _apply_gate(
        self,
        frozen_gate: FrozenGate,
        state: Dict[int, SingleQubitRDM],
        correlators: Dict[Tuple[int, int], TwoQubitRDM],
        ceg: CausalEntanglementGraph,
        params: Dict[str, float]
    ) -> None:
        """Apply a single frozen gate."""
        if frozen_gate.gate_object is None:
            return
        
        # Get runtime parameters
        gate_params = frozen_gate.get_params(params)
        
        # For two-qubit gates, ensure correlator exists
        if len(frozen_gate.qubits) == 2:
            qa, qb = frozen_gate.qubits
            key = (min(qa, qb), max(qa, qb))
            
            if key not in correlators:
                # Create product state correlator
                rdm_a = state.get(key[0], SingleQubitRDM.zero_state(key[0]))
                rdm_b = state.get(key[1], SingleQubitRDM.zero_state(key[1]))
                correlators[key] = TwoQubitRDM.product_state(rdm_a, rdm_b)
                ceg.add_edge(key[0], key[1], weight=1.0)
        
        # Apply gate
        self.updater.apply_gate(
            frozen_gate.gate_object,
            state,
            correlators,
            frozen_gate.qubits,
            gate_params
        )
    
    def _compute_expectations(
        self,
        state: Dict[int, SingleQubitRDM],
        correlators: Dict[Tuple[int, int], TwoQubitRDM],
        observables: List[str]
    ) -> Dict[str, float]:
        """Compute expectation values for observables."""
        expectations = {}
        
        for obs in observables:
            expectations[obs] = self._compute_single_expectation(
                state, correlators, obs
            )
        
        return expectations
    
    def _compute_single_expectation(
        self,
        state: Dict[int, SingleQubitRDM],
        correlators: Dict[Tuple[int, int], TwoQubitRDM],
        observable: str
    ) -> float:
        """
        Compute expectation value of a single observable.
        
        Supported formats:
        - "Z0": Single Pauli on qubit 0
        - "Z0Z1": Product of Paulis on qubits 0 and 1
        - "X0Y2Z5": Multi-qubit Pauli string
        """
        import re
        
        # Parse observable string
        pattern = r'([IXYZ])(\d+)'
        matches = re.findall(pattern, observable.upper())
        
        if not matches:
            return 0.0
        
        if len(matches) == 1:
            # Single-qubit observable
            pauli, qubit_str = matches[0]
            qubit = int(qubit_str)
            
            if qubit in state:
                return state[qubit].expectation(pauli)
            return 0.0
        
        elif len(matches) == 2:
            # Two-qubit observable
            p1, q1_str = matches[0]
            p2, q2_str = matches[1]
            q1, q2 = int(q1_str), int(q2_str)
            
            key = (min(q1, q2), max(q1, q2))
            
            if key in correlators:
                # Use correlator
                T = correlators[key].correlation_tensor
                pauli_idx = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
                
                # Handle qubit ordering
                if q1 < q2:
                    return T[pauli_idx[p1], pauli_idx[p2]]
                else:
                    return T[pauli_idx[p2], pauli_idx[p1]]
            else:
                # No correlator - use product of marginals
                exp1 = state[q1].expectation(p1) if q1 in state else 0.0
                exp2 = state[q2].expectation(p2) if q2 in state else 0.0
                return exp1 * exp2
        
        else:
            # Multi-qubit observable - approximate with product of marginals
            result = 1.0
            for pauli, qubit_str in matches:
                qubit = int(qubit_str)
                if qubit in state:
                    result *= state[qubit].expectation(pauli)
                else:
                    result *= 0.0
            return result
    
    def _sample_bitstrings(
        self,
        state: Dict[int, SingleQubitRDM],
        num_samples: int
    ) -> List[str]:
        """Sample bitstrings from the marginal distributions."""
        samples = []
        measured = sorted(self.compiled.measured_qubits)
        
        for _ in range(num_samples):
            bitstring = ""
            for q in measured:
                if q in state:
                    p0 = state[q].probability_zero()
                    outcome = 0 if np.random.random() < p0 else 1
                else:
                    outcome = np.random.randint(2)
                bitstring += str(outcome)
            samples.append(bitstring)
        
        return samples
    
    def sweep(
        self,
        param_name: str,
        param_values: List[float],
        observables: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Sweep a parameter and collect results.
        
        Args:
            param_name: Name of parameter to sweep
            param_values: Values to sweep over
            observables: Observables to compute
            
        Returns:
            Dictionary mapping observable names to arrays of values
        """
        if observables is None:
            observables = [f"Z{q}" for q in sorted(self.compiled.measured_qubits)]
        
        results = {obs: [] for obs in observables}
        
        for val in param_values:
            result = self.execute(params={param_name: val}, observables=observables)
            for obs in observables:
                results[obs].append(result.expectation_values.get(obs, 0.0))
        
        return {obs: np.array(vals) for obs, vals in results.items()}
