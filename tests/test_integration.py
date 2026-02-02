"""
Tests for ACES compiler and runtime.
"""

import pytest
import numpy as np


class TestCompiler:
    """Tests for ACES compiler."""
    
    def test_compile_simple_circuit(self):
        """Test compiling a basic circuit."""
        from aces.compiler.compile import compile_from_gate_list
        
        gates = [
            {"name": "H", "qubits": [0], "params": {}},
            {"name": "CX", "qubits": [0, 1], "params": {}},
        ]
        
        compiled = compile_from_gate_list(
            gates,
            num_qubits=2,
            measured_qubits={0, 1}
        )
        
        assert compiled.num_qubits == 2
        assert compiled.num_gates == 2
        assert compiled.measured_qubits == {0, 1}
    
    def test_lightcone_pruning(self):
        """Test that gates outside lightcone are pruned."""
        from aces.compiler.compile import compile_from_gate_list
        
        # Gate on qubit 2 should be pruned if not measuring qubit 2
        gates = [
            {"name": "H", "qubits": [0], "params": {}},
            {"name": "H", "qubits": [1], "params": {}},
            {"name": "H", "qubits": [2], "params": {}},  # Not measured
            {"name": "CX", "qubits": [0, 1], "params": {}},
        ]
        
        compiled = compile_from_gate_list(
            gates,
            num_qubits=3,
            measured_qubits={0, 1}  # Not measuring qubit 2
        )
        
        # Should prune the H gate on qubit 2
        assert compiled.num_gates < 4
        assert compiled.pruning_ratio > 0


class TestRuntime:
    """Tests for ACES runtime."""
    
    def test_execute_simple(self):
        """Test executing a simple circuit."""
        from aces.compiler.compile import compile_from_gate_list
        from aces.runtime.engine import ACESRuntime
        
        # Create Bell state
        gates = [
            {"name": "H", "qubits": [0], "params": {}},
            {"name": "CX", "qubits": [0, 1], "params": {}},
        ]
        
        compiled = compile_from_gate_list(gates, num_qubits=2, measured_qubits={0, 1})
        runtime = ACESRuntime(compiled)
        result = runtime.execute(observables=["Z0", "Z1", "Z0Z1"])
        
        # Bell state properties
        assert abs(result.expectation("Z0")) < 0.1  # ~0 for Bell state
        assert abs(result.expectation("Z1")) < 0.1
        assert abs(result.expectation("Z0Z1") - 1.0) < 0.1  # Strong ZZ correlation
    
    def test_parametric_execution(self):
        """Test executing with runtime parameters using Qiskit parametric circuit."""
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
        from aces import compile_circuit, ACESRuntime
        
        theta = Parameter('theta')
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        
        compiled = compile_circuit(qc, measured_qubits={0})
        runtime = ACESRuntime(compiled)
        
        # Ry(0) should leave |0⟩ unchanged
        result = runtime.execute(params={"theta": 0}, observables=["Z0"])
        assert result.expectation("Z0") > 0.9
        
        # Ry(π) should give -|Z⟩
        result = runtime.execute(params={"theta": np.pi}, observables=["Z0"])
        assert result.expectation("Z0") < -0.9
    
    def test_sampling(self):
        """Test bitstring sampling."""
        from aces.compiler.compile import compile_from_gate_list
        from aces.runtime.engine import ACESRuntime
        
        # All zeros circuit
        gates = []  # No gates = |00⟩
        
        compiled = compile_from_gate_list(gates, num_qubits=2, measured_qubits={0, 1})
        runtime = ACESRuntime(compiled)
        result = runtime.execute(num_samples=100)
        
        # Should mostly sample "00"
        assert result.samples is not None
        assert len(result.samples) == 100
        # Most samples should be "00"
        assert result.samples.count("00") > 90


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_ghz_state(self):
        """Test GHZ state preparation and measurement."""
        from aces.compiler.compile import compile_from_gate_list
        from aces.runtime.engine import ACESRuntime
        
        # GHZ state on 3 qubits
        gates = [
            {"name": "H", "qubits": [0], "params": {}},
            {"name": "CX", "qubits": [0, 1], "params": {}},
            {"name": "CX", "qubits": [1, 2], "params": {}},
        ]
        
        compiled = compile_from_gate_list(gates, num_qubits=3, measured_qubits={0, 1, 2})
        runtime = ACESRuntime(compiled)
        result = runtime.execute(observables=["Z0", "Z1", "Z2", "Z0Z1", "Z1Z2"])
        
        # GHZ state: |000⟩ + |111⟩
        # Individual Z expectations are 0
        assert abs(result.expectation("Z0")) < 0.2
        
        # ZZ correlations are +1
        assert result.expectation("Z0Z1") > 0.8
        assert result.expectation("Z1Z2") > 0.8
    
    def test_with_noise(self):
        """Test execution with noise model."""
        from aces.compiler.compile import compile_from_gate_list
        from aces.runtime.engine import ACESRuntime
        from aces.runtime.updater import NoiseModel
        
        gates = [
            {"name": "H", "qubits": [0], "params": {}},
        ]
        
        compiled = compile_from_gate_list(gates, num_qubits=1, measured_qubits={0})
        
        # Strong noise
        noise = NoiseModel(single_qubit_depolarizing=0.1)
        runtime = ACESRuntime(compiled, noise_model=noise)
        
        result = runtime.execute(observables=["X0"])
        
        # With noise, X expectation should be reduced from 1.0
        assert result.expectation("X0") < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
