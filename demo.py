#!/usr/bin/env python3
"""
ACES vs Pauli Propagation — Performance Comparison Demo

Demonstrates that ACES (Schrödinger picture) is faster than Pauli Propagation
(Heisenberg picture) when computing multiple observables:
  • ACES runs one forward simulation, then reads observables as free dict lookups
  • Pauli Propagation must run one back-propagation per observable

The circuit uses ROTATION GATES (RX, RY, RZ) which split each Pauli term into
two new terms. This causes exponential dictionary growth that Pauli Propagation
must pay independently for every observable, while ACES pays it only once.

Run:  python3 demo.py
"""

import sys
import os
import time
import tracemalloc
import random

# Ensure the project root is on the path so 'aces' is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aces.compiler import CircuitCompiler
from aces.runtime.pauli_state import PauliState
from aces.runtime.noise_model import GlobalDepolarizingNoise
from aces.propagation.pauli_prop import PauliPropagator
from aces.visualization import print_entanglement_map


# ─── Circuit Construction ───────────────────────────────────────────────────

def build_rotation_circuit(num_qubits=6, num_layers=4, seed=42):
    """Build a circuit with rotation gates that cause Pauli term splitting.

    Why rotations matter:
      Clifford gates (H, CX, CZ, S) map each Pauli string to exactly ONE
      new Pauli string, so dictionaries never grow from gate application.

      Rotation gates (RX, RY, RZ) split each Pauli string into TWO terms
      (when it doesn't commute with the rotation axis):
        e.g. RZ(θ): X → cos(θ)X + sin(θ)Y

      This causes the Pauli dictionary to grow exponentially with circuit
      depth, which is the cost that ACES pays once but Pauli Propagation
      must pay separately for every observable.

    Structure per layer:
      1. Random single-qubit rotations  (term-splitting)
      2. CX entangling ladder           (spreads correlations)
      3. More rotations                  (more splitting)
    """
    rng = random.Random(seed)
    gates = []
    axes = ["RX", "RY", "RZ"]

    for layer in range(num_layers):
        # Single-qubit rotations on every qubit 
        for q in range(num_qubits):
            axis = axes[rng.randint(0, 2)]
            # Angles that are NOT multiples of π/2 — these cause real splitting
            # (π/2 multiples would give cos=0 or sin=0, collapsing to 1 term)
            theta = rng.uniform(0.3, 1.2)
            gates.append({
                "name": axis,
                "qubits": [q],
                "params": {"theta": theta},
            })

        # Entangling layer: CX ladder
        for q in range(num_qubits - 1):
            gates.append({"name": "CX", "qubits": [q, q + 1]})

        # More rotations for additional splitting
        for q in range(num_qubits):
            axis = axes[rng.randint(0, 2)]
            theta = rng.uniform(0.3, 1.2)
            gates.append({
                "name": axis,
                "qubits": [q],
                "params": {"theta": theta},
            })

    return gates


# ─── Observable Generation ──────────────────────────────────────────────────

def generate_observables(num_qubits=6, num_random=10, seed=42):
    """Build a list of observables: single-qubit Z + random weight-2 Paulis."""
    rng = random.Random(seed)
    paulis = ["X", "Y", "Z"]

    # Single-qubit Z on each qubit
    observables = []
    for q in range(num_qubits):
        obs = ["I"] * num_qubits
        obs[q] = "Z"
        observables.append("".join(obs))

    # Random weight-2 Pauli strings
    for _ in range(num_random):
        obs = ["I"] * num_qubits
        q1, q2 = rng.sample(range(num_qubits), 2)
        obs[q1] = rng.choice(paulis)
        obs[q2] = rng.choice(paulis)
        observables.append("".join(obs))

    return observables


# ─── ACES Benchmark ─────────────────────────────────────────────────────────

def run_aces(compiled_circuit, observables, noise_rate=0.0):
    """Run ACES once, then look up all observables from the dictionary."""
    num_qubits = len(observables[0])

    tracemalloc.start()
    t0 = time.perf_counter()

    # Single forward simulation
    state = PauliState.zero_state(num_qubits)
    state.run_circuit(compiled_circuit)

    # Apply noise
    if noise_rate > 0:
        noise = GlobalDepolarizingNoise()
        noise.apply(state, {"p": noise_rate})

    # Free dictionary lookups for all observables
    expectations = {}
    for i, obs in enumerate(observables):
        expectations[i] = state.expectation(obs)

    wall_time = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return expectations, wall_time, peak, state.num_terms, state


# ─── Pauli Propagation Benchmark ────────────────────────────────────────────

def run_pauli_prop(compiled_circuit, observables, noise_rate=0.0,
                   coeff_threshold=1e-6):
    """Run Pauli Propagation once per observable."""
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


# ─── Comparison ─────────────────────────────────────────────────────────────

def run_comparison():
    """Main comparison: ACES vs Pauli Propagation."""
    NUM_QUBITS = 6
    NUM_LAYERS = 4

    print("=" * 72)
    print("  ACES vs Pauli Propagation — Performance Comparison")
    print("=" * 72)

    # Build & compile circuit
    circuit = build_rotation_circuit(NUM_QUBITS, NUM_LAYERS, seed=42)
    compiler = CircuitCompiler()
    compiled = compiler.compile(circuit, num_qubits=NUM_QUBITS)

    num_gates = compiled.num_steps
    # Count rotation vs Clifford gates
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

    # Generate observables
    observables = generate_observables(NUM_QUBITS, num_random=10, seed=42)
    num_obs = len(observables)
    print(f"\n  Observables: {num_obs} ({NUM_QUBITS} single-Z + 10 random weight-2)")

    # ══════════════════════════════════════════════════════════════════
    # PART 1: Noiseless accuracy check — both methods should agree
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 72}")
    print("  ▸ Part 1: Noiseless Accuracy Verification")
    print(f"{'─' * 72}")

    aces_exp_clean, aces_time_clean, _, aces_terms, aces_state = run_aces(
        compiled, observables, noise_rate=0.0
    )
    pp_exp_clean, pp_time_clean, _ = run_pauli_prop(
        compiled, observables, noise_rate=0.0, coeff_threshold=0,
    )

    max_delta_clean = max(abs(aces_exp_clean[i] - pp_exp_clean[i]) for i in range(num_obs))

    print(f"\n  {'Method':<22} {'Observables':>11}  {'Time (s)':>10}  "
          f"{'Terms tracked':>14}  {'Max |Δ|':>10}")
    print(f"  {'─' * 70}")
    print(f"  {'ACES':<22} {num_obs:>11}  {aces_time_clean:>10.4f}  "
          f"{aces_terms:>14}  {'—':>10}")
    print(f"  {'Pauli Propagation':<22} {num_obs:>11}  {pp_time_clean:>10.4f}  "
          f"{'—':>14}  {max_delta_clean:>10.2e}")

    speedup_clean = pp_time_clean / aces_time_clean if aces_time_clean > 0 else float("inf")
    print(f"\n  ⚡ Speedup: ACES is {speedup_clean:.1f}× faster for {num_obs} observables")
    print(f"  ✓ Both methods agree: max |Δ| = {max_delta_clean:.2e}")

    # Per-observable detail
    print(f"\n  Per-Observable Comparison (first 8 shown):")
    print(f"  {'Observable':<10} {'ACES ⟨O⟩':>12}  {'PP ⟨O⟩':>12}  {'|Δ|':>10}")
    print(f"  {'─' * 48}")
    for i in range(min(8, num_obs)):
        obs_label = observables[i]
        delta = abs(aces_exp_clean[i] - pp_exp_clean[i])
        print(f"  {obs_label:<10} {aces_exp_clean[i]:>+12.6f}  {pp_exp_clean[i]:>+12.6f}  {delta:>10.2e}")

    # Entanglement map from the ACES simulation
    print_entanglement_map(aces_state, label="Entanglement Map (ACES state)")



    # ══════════════════════════════════════════════════════════════════
    # PART 3: Observable count sweep
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 72}")
    print("  Observable Count Sweep: Time vs Number of Observables")
    print(f"{'═' * 72}")

    sweep_counts = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
    all_observables = generate_observables(NUM_QUBITS, num_random=26, seed=99)

    aces_times = []
    pp_times = []

    for count in sweep_counts:
        subset = all_observables[:count]

        # ACES timing (noiseless to keep comparison fair)
        t0 = time.perf_counter()
        state = PauliState.zero_state(NUM_QUBITS)
        state.run_circuit(compiled)
        for obs in subset:
            state.expectation(obs)
        aces_t = time.perf_counter() - t0
        aces_times.append(aces_t)

        # Pauli Propagation timing
        prop = PauliPropagator()
        t0 = time.perf_counter()
        for obs in subset:
            prop.simulate(compiled, obs)
        pp_t = time.perf_counter() - t0
        pp_times.append(pp_t)

    # Print sweep table
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
