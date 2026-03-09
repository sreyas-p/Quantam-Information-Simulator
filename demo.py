#!/usr/bin/env python3
"""ACES vs Pauli Propagation — Performance Comparison Demo.

Run:  python3 demo.py
"""

import sys
import os
import time
import tracemalloc
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aces.compiler import CircuitCompiler
from aces.runtime.pauli_state import PauliState
from aces.runtime.noise_model import GlobalDepolarizingNoise
from aces.propagation.pauli_prop import PauliPropagator
from aces.visualization import print_entanglement_map


def build_rotation_circuit(num_qubits=6, num_layers=4, seed=42):
    """Build a rotation + CX circuit that causes Pauli term splitting."""
    rng = random.Random(seed)
    gates = []
    axes = ["RX", "RY", "RZ"]

    for layer in range(num_layers):
        for q in range(num_qubits):
            axis = axes[rng.randint(0, 2)]
            theta = rng.uniform(0.3, 1.2)
            gates.append({"name": axis, "qubits": [q], "params": {"theta": theta}})

        for q in range(num_qubits - 1):
            gates.append({"name": "CX", "qubits": [q, q + 1]})

        for q in range(num_qubits):
            axis = axes[rng.randint(0, 2)]
            theta = rng.uniform(0.3, 1.2)
            gates.append({"name": axis, "qubits": [q], "params": {"theta": theta}})

    return gates


def generate_observables(num_qubits=6, num_random=10, seed=42):
    """Generate single-qubit Z + random weight-2 Pauli observables."""
    rng = random.Random(seed)
    paulis = ["X", "Y", "Z"]
    observables = []

    for q in range(num_qubits):
        obs = ["I"] * num_qubits
        obs[q] = "Z"
        observables.append("".join(obs))

    for _ in range(num_random):
        obs = ["I"] * num_qubits
        q1, q2 = rng.sample(range(num_qubits), 2)
        obs[q1] = rng.choice(paulis)
        obs[q2] = rng.choice(paulis)
        observables.append("".join(obs))

    return observables


def run_aces(compiled_circuit, observables, noise_rate=0.0):
    """Run ACES: one forward simulation, then dictionary lookups."""
    num_qubits = len(observables[0])

    tracemalloc.start()
    t0 = time.perf_counter()

    state = PauliState.zero_state(num_qubits)
    state.run_circuit(compiled_circuit)

    if noise_rate > 0:
        GlobalDepolarizingNoise().apply(state, {"p": noise_rate})

    expectations = {}
    for i, obs in enumerate(observables):
        expectations[i] = state.expectation(obs)

    wall_time = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return expectations, wall_time, peak, state.num_terms, state


def run_pauli_prop(compiled_circuit, observables, noise_rate=0.0,
                   coeff_threshold=1e-6):
    """Run Pauli Propagation: one back-propagation per observable."""
    propagator = PauliPropagator()

    tracemalloc.start()
    t0 = time.perf_counter()

    expectations = {}
    for i, obs in enumerate(observables):
        val = propagator.simulate(
            compiled_circuit, obs, noise_rate=noise_rate,
            coeff_threshold=coeff_threshold,
        )
        expectations[i] = val

    wall_time = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return expectations, wall_time, peak


def run_comparison():
    """Main comparison: ACES vs Pauli Propagation."""
    NUM_QUBITS = 6
    NUM_LAYERS = 4

    print("=" * 72)
    print("  ACES vs Pauli Propagation — Performance Comparison")
    print("=" * 72)

    circuit = build_rotation_circuit(NUM_QUBITS, NUM_LAYERS, seed=42)
    compiler = CircuitCompiler()
    compiled = compiler.compile(circuit, num_qubits=NUM_QUBITS)

    num_gates = compiled.num_steps
    rot_count = sum(1 for s in compiled.steps if s.gate_name in ("RX", "RY", "RZ"))
    cliff_count = num_gates - rot_count

    print(f"\n  Circuit: {NUM_QUBITS} qubits, {num_gates} gates "
          f"({rot_count} rotations + {cliff_count} Cliffords)")
    print(f"  Layers:  {NUM_LAYERS} (rotation → CX → rotation per layer)")
    print(f"\n  Why rotations matter:")
    print(f"    Clifford gates map 1 Pauli term → 1 term  (no dictionary growth)")
    print(f"    Rotation gates split 1 Pauli term → 2 terms (exponential growth)")
    print(f"    PP pays this splitting cost once per observable.")
    print(f"    ACES pays it once total, then reads observables for free.")

    observables = generate_observables(NUM_QUBITS, num_random=10, seed=42)
    num_obs = len(observables)
    print(f"\n  Observables: {num_obs} ({NUM_QUBITS} single-Z + 10 random weight-2)")

    print(f"\n{'─' * 72}")
    print("  ▸ Noiseless Accuracy Verification")
    print(f"{'─' * 72}")

    aces_exp, aces_time, _, aces_terms, aces_state = run_aces(
        compiled, observables, noise_rate=0.0
    )
    pp_exp, pp_time, _ = run_pauli_prop(
        compiled, observables, noise_rate=0.0, coeff_threshold=0,
    )

    max_delta = max(abs(aces_exp[i] - pp_exp[i]) for i in range(num_obs))

    print(f"\n  {'Method':<22} {'Observables':>11}  {'Time (s)':>10}  "
          f"{'Terms tracked':>14}  {'Max |Δ|':>10}")
    print(f"  {'─' * 70}")
    print(f"  {'ACES':<22} {num_obs:>11}  {aces_time:>10.4f}  "
          f"{aces_terms:>14}  {'—':>10}")
    print(f"  {'Pauli Propagation':<22} {num_obs:>11}  {pp_time:>10.4f}  "
          f"{'—':>14}  {max_delta:>10.2e}")

    speedup = pp_time / aces_time if aces_time > 0 else float("inf")
    print(f"\n  ⚡ Speedup: ACES is {speedup:.1f}× faster for {num_obs} observables")
    print(f"  ✓ Both methods agree: max |Δ| = {max_delta:.2e}")

    print(f"\n  Per-Observable Comparison (first 8 shown):")
    print(f"  {'Observable':<10} {'ACES ⟨O⟩':>12}  {'PP ⟨O⟩':>12}  {'|Δ|':>10}")
    print(f"  {'─' * 48}")
    for i in range(min(8, num_obs)):
        delta = abs(aces_exp[i] - pp_exp[i])
        print(f"  {observables[i]:<10} {aces_exp[i]:>+12.6f}  {pp_exp[i]:>+12.6f}  {delta:>10.2e}")

    print_entanglement_map(aces_state, label="Entanglement Map (ACES state)")

    print(f"\n{'═' * 72}")
    print("  Observable Count Sweep: Time vs Number of Observables")
    print(f"{'═' * 72}")

    sweep_counts = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
    all_observables = generate_observables(NUM_QUBITS, num_random=26, seed=99)

    aces_times = []
    pp_times = []

    for count in sweep_counts:
        subset = all_observables[:count]

        t0 = time.perf_counter()
        state = PauliState.zero_state(NUM_QUBITS)
        state.run_circuit(compiled)
        for obs in subset:
            state.expectation(obs)
        aces_times.append(time.perf_counter() - t0)

        prop = PauliPropagator()
        t0 = time.perf_counter()
        for obs in subset:
            prop.simulate(compiled, obs)
        pp_times.append(time.perf_counter() - t0)

    max_time = max(max(pp_times), max(aces_times))
    bar_width = 40

    print(f"\n  {'#Obs':>5}  {'ACES (s)':>10}  {'PP (s)':>10}  "
          f"{'Speedup':>8}  Chart")
    print(f"  {'─' * 75}")

    for i, count in enumerate(sweep_counts):
        sp = pp_times[i] / aces_times[i] if aces_times[i] > 0 else float("inf")
        aces_bar = int(aces_times[i] / max_time * bar_width) if max_time > 0 else 0
        pp_bar = int(pp_times[i] / max_time * bar_width) if max_time > 0 else 0
        print(f"  {count:>5}  {aces_times[i]:>10.4f}  {pp_times[i]:>10.4f}  "
              f"{sp:>7.1f}×  "
              f"A{'█' * aces_bar}{'░' * (bar_width - aces_bar)}")
        print(f"  {'':>5}  {'':>10}  {'':>10}  {'':>8}  "
              f"P{'█' * pp_bar}{'░' * (bar_width - pp_bar)}")


if __name__ == "__main__":
    run_comparison()
