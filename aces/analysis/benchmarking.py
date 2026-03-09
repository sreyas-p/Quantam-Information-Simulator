"""Benchmarking — compare full statevector vs ACES Pauli engine."""

from __future__ import annotations

import time
import tracemalloc
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from aces.compiler.pauli_transform import CircuitCompiler
from aces.runtime.pauli_state import PauliState


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    num_qubits: int
    method: str
    wall_time_s: float
    peak_memory_bytes: int
    z_expectations: Dict[int, float]

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult({self.method}, n={self.num_qubits}, "
            f"time={self.wall_time_s:.4f}s, mem={self.peak_memory_bytes}B)"
        )


def random_circuit(num_qubits: int, depth: int, seed: int = 42) -> List[Dict]:
    """Generate a random Clifford circuit."""
    rng = np.random.RandomState(seed)
    gates = []
    for _ in range(depth):
        q = int(rng.randint(num_qubits))
        gate_name = rng.choice(["H", "X", "S"])
        gates.append({"name": gate_name, "qubits": [q]})
        if num_qubits >= 2:
            q1, q2 = rng.choice(num_qubits, size=2, replace=False)
            gates.append({"name": "CX", "qubits": [int(q1), int(q2)]})
    return gates


class ACESBenchmark:
    """Compare statevector vs ACES Pauli engine."""

    def run_statevector(self, num_qubits: int, gate_sequence: List[Dict]) -> BenchmarkResult:
        """Full 2^n statevector simulation."""
        tracemalloc.start()
        t0 = time.perf_counter()

        state = np.zeros(2**num_qubits, dtype=np.complex128)
        state[0] = 1.0

        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
        gate_map = {"H": H, "X": X, "S": S}

        for spec in gate_sequence:
            name, qubits = spec["name"].upper(), spec["qubits"]
            if len(qubits) == 1 and name in gate_map:
                state = self._apply_1q(state, gate_map[name], qubits[0], num_qubits)
            elif len(qubits) == 2 and name in ("CX", "CNOT"):
                state = self._apply_cnot(state, qubits[0], qubits[1], num_qubits)

        wall_time = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        probs = np.abs(state) ** 2
        z_exp = {}
        for q in range(num_qubits):
            p0 = sum(probs[i] for i in range(2**num_qubits)
                     if not (i >> (num_qubits - 1 - q)) & 1)
            z_exp[q] = 2 * p0 - 1

        return BenchmarkResult(num_qubits, "statevector", wall_time, peak, z_exp)

    def run_aces(self, num_qubits: int, gate_sequence: List[Dict]) -> BenchmarkResult:
        """ACES Pauli dictionary simulation."""
        tracemalloc.start()
        t0 = time.perf_counter()

        compiler = CircuitCompiler()
        compiled = compiler.compile(gate_sequence, num_qubits)
        state = PauliState.zero_state(num_qubits)
        state.run_circuit(compiled)

        wall_time = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        z_exp = {}
        for q in range(num_qubits):
            _, _, rz = state.bloch_vector(q)
            z_exp[q] = rz

        return BenchmarkResult(num_qubits, "aces", wall_time, peak, z_exp)

    def run(self, qubit_range=range(2, 10, 2), depth=10, seed=42):
        """Run comparison across qubit counts."""
        results = []
        for n in qubit_range:
            circuit = random_circuit(n, depth, seed)
            sv = self.run_statevector(n, circuit)
            aces = self.run_aces(n, circuit)
            results.append((sv, aces))
        return results

    def fidelity_error(self, sv: BenchmarkResult, aces: BenchmarkResult) -> float:
        """Max |⟨Z⟩_exact - ⟨Z⟩_aces| across qubits."""
        return max(abs(sv.z_expectations[q] - aces.z_expectations.get(q, 0))
                   for q in sv.z_expectations)

    @staticmethod
    def print_results(results):
        """Print formatted comparison table."""
        print(f"{'Qubits':>6}  {'SV Time':>10}  {'ACES Time':>10}  "
              f"{'SV Mem':>10}  {'ACES Mem':>10}  {'Max |ΔZ|':>10}")
        print("-" * 65)
        bench = ACESBenchmark()
        for sv, aces in results:
            err = bench.fidelity_error(sv, aces)
            print(f"{sv.num_qubits:>6}  {sv.wall_time_s:>10.4f}s  {aces.wall_time_s:>10.4f}s  "
                  f"{sv.peak_memory_bytes:>9}B  {aces.peak_memory_bytes:>9}B  {err:>10.6f}")

    # -- Internal helpers --

    def _apply_1q(self, state, U, qubit, n):
        dim = 2**n
        result = np.zeros(dim, dtype=np.complex128)
        for i in range(dim):
            bit = (i >> (n - 1 - qubit)) & 1
            i_flip = i ^ (1 << (n - 1 - qubit))
            if bit == 0:
                result[i] += U[0, 0] * state[i] + U[0, 1] * state[i_flip]
            else:
                result[i] += U[1, 0] * state[i_flip] + U[1, 1] * state[i]
        return result

    def _apply_cnot(self, state, ctrl, tgt, n):
        result = state.copy()
        for i in range(2**n):
            if (i >> (n - 1 - ctrl)) & 1:
                j = i ^ (1 << (n - 1 - tgt))
                result[i], result[j] = state[j], state[i]
        return result


class PauliPropBenchmark:
    """Benchmark for Heisenberg-picture Pauli Propagation.

    Runs PauliPropagator.simulate() once per observable and records
    total wall time and peak memory.
    """

    def run(
        self,
        compiled_circuit,
        observables: List[str],
        noise_rate: float = 0.0,
        coeff_threshold: float = 1e-6,
        weight_cutoff=None,
    ) -> BenchmarkResult:
        """Run back-propagation for each observable and measure performance.

        Args:
            compiled_circuit: A CompiledCircuit to propagate through.
            observables: List of Pauli-string observables to evaluate.
            noise_rate: Per-qubit depolarizing noise rate.
            coeff_threshold: Coefficient truncation threshold.
            weight_cutoff: Optional Pauli-weight cutoff.

        Returns:
            BenchmarkResult with aggregated timing and memory stats.
            z_expectations maps observable index → expectation value.
        """
        from aces.propagation.pauli_prop import PauliPropagator

        propagator = PauliPropagator()
        num_qubits = len(observables[0]) if observables else 0

        tracemalloc.start()
        t0 = time.perf_counter()

        z_expectations = {}
        for idx, obs in enumerate(observables):
            val = propagator.simulate(
                compiled_circuit,
                obs,
                noise_rate=noise_rate,
                coeff_threshold=coeff_threshold,
                weight_cutoff=weight_cutoff,
            )
            z_expectations[idx] = val

        wall_time = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return BenchmarkResult(
            num_qubits=num_qubits,
            method="pauli_propagation",
            wall_time_s=wall_time,
            peak_memory_bytes=peak,
            z_expectations=z_expectations,
        )
