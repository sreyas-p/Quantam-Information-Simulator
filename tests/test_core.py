"""
Tests for ACES core components.
"""

import pytest
import numpy as np

from aces.core.density_matrix import SingleQubitRDM, TwoQubitRDM
from aces.core.gates import GateLibrary
from aces.core.ceg import CausalEntanglementGraph, CEGBuilder


class TestSingleQubitRDM:
    """Tests for SingleQubitRDM."""
    
    def test_zero_state(self):
        rdm = SingleQubitRDM.zero_state(0)
        assert np.allclose(rdm.bloch, [0, 0, 1])
        assert np.isclose(rdm.probability_zero(), 1.0)
        assert np.isclose(rdm.probability_one(), 0.0)
    
    def test_one_state(self):
        rdm = SingleQubitRDM.one_state(0)
        assert np.allclose(rdm.bloch, [0, 0, -1])
        assert np.isclose(rdm.probability_zero(), 0.0)
        assert np.isclose(rdm.probability_one(), 1.0)
    
    def test_plus_state(self):
        rdm = SingleQubitRDM.plus_state(0)
        assert np.allclose(rdm.bloch, [1, 0, 0])
        assert np.isclose(rdm.probability_zero(), 0.5)
        assert np.isclose(rdm.probability_one(), 0.5)
    
    def test_purity(self):
        # Pure state
        rdm = SingleQubitRDM.zero_state(0)
        assert np.isclose(rdm.purity, 1.0)
        
        # Mixed state
        rdm_mixed = SingleQubitRDM.maximally_mixed(0)
        assert np.isclose(rdm_mixed.purity, 0.5)
    
    def test_expectation_values(self):
        rdm = SingleQubitRDM.zero_state(0)
        assert np.isclose(rdm.expectation('Z'), 1.0)
        assert np.isclose(rdm.expectation('X'), 0.0)
        
        rdm_plus = SingleQubitRDM.plus_state(0)
        assert np.isclose(rdm_plus.expectation('X'), 1.0)
        assert np.isclose(rdm_plus.expectation('Z'), 0.0)
    
    def test_roundtrip_density_matrix(self):
        bloch = np.array([0.3, 0.4, 0.5])
        bloch = bloch / np.linalg.norm(bloch)  # Normalize to be pure
        
        rdm = SingleQubitRDM(qubit_id=0, bloch=bloch)
        rho = rdm.to_density_matrix()
        rdm2 = SingleQubitRDM.from_density_matrix(0, rho)
        
        assert np.allclose(rdm.bloch, rdm2.bloch)


class TestTwoQubitRDM:
    """Tests for TwoQubitRDM."""
    
    def test_product_state(self):
        rdm_a = SingleQubitRDM.zero_state(0)
        rdm_b = SingleQubitRDM.zero_state(1)
        
        two_qubit = TwoQubitRDM.product_state(rdm_a, rdm_b)
        
        # Product state has zero concurrence
        assert two_qubit.concurrence() < 1e-10
        assert two_qubit.is_product_state()
    
    def test_bell_state(self):
        bell = TwoQubitRDM.bell_phi_plus(0, 1)
        
        # Bell state is maximally entangled
        assert np.isclose(bell.concurrence(), 1.0)
        assert not bell.is_product_state()
    
    def test_marginals(self):
        bell = TwoQubitRDM.bell_phi_plus(0, 1)
        
        marginal_a = bell.marginal_a()
        marginal_b = bell.marginal_b()
        
        # Marginals of Bell state are maximally mixed
        assert np.isclose(marginal_a.purity, 0.5)
        assert np.isclose(marginal_b.purity, 0.5)
    
    def test_mutual_information(self):
        # Product state has zero mutual information
        rdm_a = SingleQubitRDM.zero_state(0)
        rdm_b = SingleQubitRDM.zero_state(1)
        product = TwoQubitRDM.product_state(rdm_a, rdm_b)
        assert product.mutual_information() < 1e-10
        
        # Bell state has maximal mutual information (2 bits)
        bell = TwoQubitRDM.bell_phi_plus(0, 1)
        assert np.isclose(bell.mutual_information(), 2.0)


class TestGates:
    """Tests for gate operations."""
    
    def test_hadamard(self):
        state = {0: SingleQubitRDM.zero_state(0)}
        correlators = {}
        
        GateLibrary.H.apply_to_rdm(state, correlators, (0,))
        
        # H|0⟩ = |+⟩
        assert np.isclose(state[0].expectation('X'), 1.0, atol=1e-10)
        assert np.isclose(state[0].expectation('Z'), 0.0, atol=1e-10)
    
    def test_pauli_x(self):
        state = {0: SingleQubitRDM.zero_state(0)}
        correlators = {}
        
        GateLibrary.X.apply_to_rdm(state, correlators, (0,))
        
        # X|0⟩ = |1⟩
        assert np.isclose(state[0].probability_one(), 1.0)
    
    def test_cnot_creates_entanglement(self):
        # Start with |+0⟩
        state = {
            0: SingleQubitRDM.plus_state(0),
            1: SingleQubitRDM.zero_state(1)
        }
        correlators = {}
        
        GateLibrary.CX.apply_to_rdm(state, correlators, (0, 1))
        
        # Should have created a correlator
        assert (0, 1) in correlators
        
        # Result is Bell state, check entanglement
        corr = correlators[(0, 1)]
        assert corr.concurrence() > 0.9  # Should be ~1.0
    
    def test_parametric_rotation(self):
        state = {0: SingleQubitRDM.zero_state(0)}
        correlators = {}
        
        # Ry(π) should flip |0⟩ to |1⟩
        GateLibrary.Ry.apply_to_rdm(state, correlators, (0,), {'theta': np.pi})
        
        assert np.isclose(state[0].probability_one(), 1.0, atol=1e-10)


class TestCEG:
    """Tests for Causal Entanglement Graph."""
    
    def test_add_edge(self):
        ceg = CausalEntanglementGraph(num_qubits=4, measured_qubits={0})
        
        ceg.add_edge(0, 1)
        ceg.add_edge(1, 2)
        
        assert ceg.num_edges == 2
        assert ceg.has_edge(0, 1)
        assert ceg.has_edge(1, 2)
        assert not ceg.has_edge(0, 2)
    
    def test_neighbors(self):
        ceg = CausalEntanglementGraph(num_qubits=4, measured_qubits={0})
        
        ceg.add_edge(0, 1)
        ceg.add_edge(0, 2)
        
        neighbors = ceg.neighbors(0)
        assert neighbors == {1, 2}
    
    def test_prune_to_budget(self):
        ceg = CausalEntanglementGraph(
            num_qubits=4,
            measured_qubits={0},
            entanglement_budget=2.0
        )
        
        ceg.add_edge(0, 1, weight=1.0)  # Connected to measured
        ceg.add_edge(1, 2, weight=0.5)
        ceg.add_edge(2, 3, weight=0.3)
        
        assert ceg.total_weight == 1.8
        
        # Budget exceeded (1.8 > we want lower), prune weakest
        removed = ceg.prune_to_budget(1.5)
        
        # Should remove 2-3 edge (weakest, not touching measured)
        assert len(removed) == 1
        assert not ceg.has_edge(2, 3)
    
    def test_builder(self):
        builder = CEGBuilder(
            num_qubits=4,
            measured_qubits={0, 1}
        )
        
        gates = [
            (0, 0, 1),  # Gate 0: qubits 0, 1
            (1, 1, 2),  # Gate 1: qubits 1, 2
            (2, 2, 3),  # Gate 2: qubits 2, 3
        ]
        
        ceg = builder.build_from_gates(gates)
        
        assert ceg.num_edges == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
