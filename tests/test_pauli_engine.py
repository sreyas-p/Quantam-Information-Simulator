"""Tests for the Pauli-based ACES engine."""

import pytest
import numpy as np

from aces.compiler.parser import parse_circuit
from aces.compiler.gate_rules import transform_pauli_string, SINGLE_QUBIT_RULES, CNOT_RULES
from aces.compiler.pauli_transform import CircuitCompiler
from aces.runtime.pauli_state import PauliState
from aces.runtime.pruning_engine import PruningEngine
from aces.runtime.noise_model import DepolarizingNoise, DephasingNoise
from aces.runtime.entanglement_tracker import EntanglementTracker
from aces.analysis.metrics import compute_purity, compression_stats, entanglement_matrix


class TestParser:
    def test_parse_basic(self):
        ops = parse_circuit([{"name": "H", "qubits": [0]}, {"name": "CX", "qubits": [0, 1]}])
        assert len(ops) == 2
        assert ops[0].name == "H"
        assert ops[1].qubits == (0, 1)


class TestGateRules:
    def test_hadamard_rules(self):
        assert SINGLE_QUBIT_RULES["H"]["X"] == (1, "Z")
        assert SINGLE_QUBIT_RULES["H"]["Z"] == (1, "X")
        assert SINGLE_QUBIT_RULES["H"]["Y"] == (-1, "Y")

    def test_hadamard_on_string(self):
        # H on qubit 0 of "ZI": Z→X, so "ZI" → "XI"
        result = transform_pauli_string("ZI", "H", (0,), 2)
        assert result == [(1, "XI")]

    def test_cnot_rules(self):
        assert CNOT_RULES[("X", "I")] == (1, "X", "X")
        assert CNOT_RULES[("I", "Z")] == (1, "Z", "Z")
        assert CNOT_RULES[("X", "Z")] == (-1, "Y", "Y")

    def test_cnot_on_string(self):
        result = transform_pauli_string("XI", "CX", (0, 1), 2)
        assert result == [(1, "XX")]

    def test_rotation_rz(self):
        result = transform_pauli_string("XI", "RZ", (0,), 2, {"theta": np.pi / 2})
        # Rz(π/2): X → cos(π/2)X + sin(π/2)Y ≈ Y
        assert len(result) == 2
        assert np.isclose(result[0][0], 0.0, atol=1e-10)  # cos(π/2) ≈ 0
        assert np.isclose(result[1][0], 1.0, atol=1e-10)  # sin(π/2) ≈ 1
        assert result[1][1] == "YI"


class TestPauliState:
    def test_zero_state(self):
        state = PauliState.zero_state(2)
        # |00⟩ → {II:1, IZ:1, ZI:1, ZZ:1}
        assert state.num_terms == 4
        assert state.coeffs["II"] == 1.0
        assert state.coeffs["ZZ"] == 1.0

    def test_bloch_vector_zero_state(self):
        state = PauliState.zero_state(1)
        rx, ry, rz = state.bloch_vector(0)
        assert np.isclose(rx, 0.0)
        assert np.isclose(ry, 0.0)
        assert np.isclose(rz, 1.0)  # |0⟩ → (0,0,1)

    def test_hadamard_flip(self):
        state = PauliState.zero_state(1)
        from aces.compiler.pauli_transform import CompiledStep
        state.apply_step(CompiledStep("H", (0,)))
        rx, ry, rz = state.bloch_vector(0)
        assert np.isclose(rx, 1.0)  # H|0⟩ = |+⟩ → (1,0,0)
        assert np.isclose(rz, 0.0)

    def test_x_gate_flip(self):
        state = PauliState.zero_state(1)
        from aces.compiler.pauli_transform import CompiledStep
        state.apply_step(CompiledStep("X", (0,)))
        rx, ry, rz = state.bloch_vector(0)
        assert np.isclose(rz, -1.0)  # X|0⟩ = |1⟩ → (0,0,-1)

    def test_bell_state(self):
        state = PauliState.zero_state(2)
        from aces.compiler.pauli_transform import CompiledStep
        state.apply_step(CompiledStep("H", (0,)))
        state.apply_step(CompiledStep("CX", (0, 1)))
        # Bell state: ⟨Z0⟩ = ⟨Z1⟩ = 0, ⟨Z0Z1⟩ = 1
        _, _, z0 = state.bloch_vector(0)
        _, _, z1 = state.bloch_vector(1)
        assert np.isclose(z0, 0.0)
        assert np.isclose(z1, 0.0)
        assert np.isclose(state.expectation("ZZ"), 1.0)

    def test_measurement_probs(self):
        state = PauliState.zero_state(1)
        probs = state.measurement_probabilities(0)
        assert np.isclose(probs["0"], 1.0)
        assert np.isclose(probs["1"], 0.0)

    def test_compiled_circuit(self):
        compiler = CircuitCompiler()
        compiled = compiler.compile(
            [{"name": "H", "qubits": [0]}, {"name": "CX", "qubits": [0, 1]}],
            num_qubits=2,
        )
        state = PauliState.zero_state(2)
        state.run_circuit(compiled)
        assert np.isclose(state.expectation("ZZ"), 1.0)


class TestPruning:
    def test_prune_small_terms(self):
        state = PauliState(num_qubits=2, coeffs={"II": 1.0, "XX": 0.001, "ZZ": 0.5})
        engine = PruningEngine(threshold=0.01)
        result = engine.prune(state)
        assert result.terms_pruned == 1
        assert "XX" not in state.coeffs
        assert "ZZ" in state.coeffs

    def test_never_prune_identity(self):
        state = PauliState(num_qubits=1, coeffs={"I": 0.001, "Z": 0.5})
        engine = PruningEngine(threshold=0.1)
        result = engine.prune(state)
        assert "I" in state.coeffs  # Identity never pruned

    def test_error_tracking(self):
        state = PauliState(num_qubits=2, coeffs={"II": 1.0, "XX": 0.005, "YY": 0.003})
        engine = PruningEngine(threshold=0.01)
        result = engine.prune(state)
        assert np.isclose(result.total_dropped_magnitude, 0.008)


class TestNoise:
    def test_depolarizing(self):
        state = PauliState(num_qubits=1, coeffs={"I": 1.0, "Z": 1.0})
        noise = DepolarizingNoise()
        noise.apply(state, {"p": 0.1})
        assert np.isclose(state.coeffs["I"], 1.0)  # Identity unchanged
        assert np.isclose(state.coeffs["Z"], 0.9)  # Z *= 0.9

    def test_dephasing(self):
        state = PauliState(num_qubits=1, coeffs={"I": 1.0, "X": 0.5, "Z": 0.5})
        noise = DephasingNoise()
        noise.apply(state, {"p": 0.2})
        assert np.isclose(state.coeffs["X"], 0.4)  # X scaled
        assert np.isclose(state.coeffs["Z"], 0.5)  # Z unchanged


class TestEntanglementTracker:
    def test_tracks_growth(self):
        from aces.compiler.pauli_transform import CompiledStep
        state = PauliState.zero_state(2)
        tracker = EntanglementTracker()

        tracker.record(0, "init", state)
        state.apply_step(CompiledStep("H", (0,)))
        tracker.record(1, "H", state)
        state.apply_step(CompiledStep("CX", (0, 1)))
        tracker.record(2, "CX", state)

        strengths = tracker.strength_over_time(0, 1)
        # |00⟩ has ZZ=1.0 (classical correlation), after CX entanglement grows
        assert strengths[2] > strengths[1]  # CX increases entanglement

    def test_entanglement_matrix(self):
        from aces.compiler.pauli_transform import CompiledStep
        state = PauliState.zero_state(2)
        state.apply_step(CompiledStep("H", (0,)))
        state.apply_step(CompiledStep("CX", (0, 1)))

        tracker = EntanglementTracker()
        mat = tracker.entanglement_matrix(state)
        assert mat[0, 1] > 0
        assert mat[0, 1] == mat[1, 0]


class TestMetrics:
    def test_purity_pure_state(self):
        state = PauliState.zero_state(1)
        assert np.isclose(compute_purity(state), 1.0)

    def test_compression_stats(self):
        state = PauliState.zero_state(2)
        stats = compression_stats(state)
        assert stats["naive_size"] == 16  # 4^2
        assert stats["actual_terms"] == 4  # {I,Z}^2


class TestBenchmark:
    def test_small_comparison(self):
        from aces.analysis.benchmarking import ACESBenchmark, random_circuit
        bench = ACESBenchmark()
        circuit = random_circuit(2, 3, seed=42)
        sv = bench.run_statevector(2, circuit)
        aces = bench.run_aces(2, circuit)
        err = bench.fidelity_error(sv, aces)
        assert err < 1e-10  # Clifford gates → exact match


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
