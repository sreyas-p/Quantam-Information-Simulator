"""
ACES — Full Pipeline Demo (Complex System)

Simulates a 6-qubit quantum circuit with 40+ gates, noise, and pruning.

Input circuit → Compiler → Pauli instructions → Runtime state engine
→ Noise updates → Pruning → Metrics + Visualization

Run: python3 -m aces.main
"""

from aces.compiler import CircuitCompiler
from aces.compiler.pauli_transform import CompiledStep
from aces.runtime import PauliState, EntanglementTracker, PruningEngine, DepolarizingNoise
from aces.runtime.noise_model import GlobalDepolarizingNoise
from aces.analysis import compute_bloch_vectors, compute_purity, compression_stats, entanglement_matrix
from aces.visualization import print_bloch_vectors, print_entanglement_map


def build_complex_circuit():
    """Build a 6-qubit circuit with 48 gates.

    Structure:
      Layer 1: Prepare superpositions (H on all qubits)
      Layer 2: Entangle pairs (CX chain 0→1→2→3→4→5)
      Layer 3: Phase rotations (S gates)
      Layer 4: Cross-entangle (CX across distant qubits)
      Layer 5: Scramble (H + CX mixing)
      Layer 6: More entanglement (CZ gates)
      Layer 7: Final rotations
    """
    gates = []

    # Layer 1 — Superposition on all 6 qubits
    for q in range(6):
        gates.append({"name": "H", "qubits": [q]})

    # Layer 2 — Nearest-neighbor CX chain
    for q in range(5):
        gates.append({"name": "CX", "qubits": [q, q + 1]})

    # Layer 3 — Phase gates
    for q in [0, 2, 4]:
        gates.append({"name": "S", "qubits": [q]})

    # Layer 4 — Long-range entanglement
    gates.append({"name": "CX", "qubits": [0, 3]})
    gates.append({"name": "CX", "qubits": [1, 4]})
    gates.append({"name": "CX", "qubits": [2, 5]})
    gates.append({"name": "CZ", "qubits": [0, 5]})

    # Layer 5 — Scrambling
    for q in [1, 3, 5]:
        gates.append({"name": "H", "qubits": [q]})
    gates.append({"name": "CX", "qubits": [3, 0]})
    gates.append({"name": "CX", "qubits": [4, 1]})
    gates.append({"name": "CX", "qubits": [5, 2]})

    # Layer 6 — CZ layer
    gates.append({"name": "CZ", "qubits": [0, 2]})
    gates.append({"name": "CZ", "qubits": [1, 3]})
    gates.append({"name": "CZ", "qubits": [2, 4]})
    gates.append({"name": "CZ", "qubits": [3, 5]})

    # Layer 7 — Final rotations + entanglement
    for q in range(6):
        gates.append({"name": "H", "qubits": [q]})
    gates.append({"name": "CX", "qubits": [0, 1]})
    gates.append({"name": "CX", "qubits": [2, 3]})
    gates.append({"name": "CX", "qubits": [4, 5]})

    return gates


def main():
    NUM_QUBITS = 6

    print("=" * 65)
    print("  ACES — Adaptive Causal-Entropy Simulation")
    print("  Complex 6-Qubit / 48-Gate Demo")
    print("=" * 65)

    # ── 1. Build & Compile ──────────────────────────────────────────
    circuit = build_complex_circuit()
    compiler = CircuitCompiler()
    compiled = compiler.compile(circuit, num_qubits=NUM_QUBITS, measured_qubits=set(range(NUM_QUBITS)))

    print(f"\n[1] CIRCUIT")
    print(f"    Qubits: {NUM_QUBITS}")
    print(f"    Gates:  {compiled.num_steps}")
    print(f"    Naive statevector size: 2^{NUM_QUBITS} = {2**NUM_QUBITS} amplitudes")

    gate_counts = {}
    for s in compiled.steps:
        gate_counts[s.gate_name] = gate_counts.get(s.gate_name, 0) + 1
    breakdown = ", ".join(f"{k}×{v}" for k, v in sorted(gate_counts.items()))
    print(f"    Gate breakdown: {breakdown}")

    # ── 2. Initialize & Execute with Tracking ───────────────────────
    state = PauliState.zero_state(NUM_QUBITS)
    tracker = EntanglementTracker()
    tracker.record(0, "init", state)

    print(f"\n[2] EXECUTION")
    print(f"    Initial Pauli terms: {state.num_terms}")

    for i, step in enumerate(compiled.steps):
        state.apply_step(step)
        tracker.record(i + 1, step.gate_name, state)

    print(f"    After {compiled.num_steps} gates: {state.num_terms} terms")
    print(f"    Naive Pauli space: 4^{NUM_QUBITS} = {4**NUM_QUBITS}")
    stats = compression_stats(state)
    print(f"    Compression: {stats['actual_terms']:.0f}/{stats['naive_size']:.0f} "
          f"({stats['savings_pct']:.1f}% savings)")

    # ── 3. Bloch Vectors ────────────────────────────────────────────
    print_bloch_vectors(state, label="[3] BLOCH VECTORS (pre-noise)")

    # ── 4. Entanglement Map ─────────────────────────────────────────
    print_entanglement_map(state, label="[4] ENTANGLEMENT MAP")

    # ── 5. Purity ───────────────────────────────────────────────────
    purity = compute_purity(state)
    print(f"\n[5] STATE PURITY")
    print(f"    Purity = {purity:.6f}")
    print(f"    (1.0 = pure, {1/2**NUM_QUBITS:.6f} = maximally mixed)")

    # ── 6. Key Correlations ─────────────────────────────────────────
    print(f"\n[6] SELECTED CORRELATIONS")
    for pair_str in ["ZZIIII", "IZZIII", "ZIIZII", "ZIIIZI", "ZIIIII", "IZZIZZ"]:
        val = state.expectation(pair_str)
        if abs(val) > 0.01:
            print(f"    ⟨{pair_str}⟩ = {val:+.4f}")

    # ── 7. Noise Simulation ─────────────────────────────────────────
    print(f"\n[7] NOISE SIMULATION")
    noiseless = state.copy()

    noise_levels = [0.01, 0.05, 0.10, 0.20]
    print(f"    {'p':>6}  {'Terms':>6}  {'Purity':>8}  {'⟨Z0⟩':>8}  {'|r0|':>8}")
    print(f"    {'-'*42}")

    # Show noiseless baseline
    bv0 = noiseless.bloch_vector(0)
    import numpy as np
    r0 = np.linalg.norm(bv0)
    print(f"    {'0':>6}  {noiseless.num_terms:>6}  {compute_purity(noiseless):>8.4f}  {bv0[2]:>+8.4f}  {r0:>8.4f}")

    for p in noise_levels:
        test_state = noiseless.copy()
        noise = GlobalDepolarizingNoise()
        noise.apply(test_state, {"p": p})
        pruner = PruningEngine(threshold=0.01)
        pruner.prune(test_state)
        bv = test_state.bloch_vector(0)
        r = np.linalg.norm(bv)
        pur = compute_purity(test_state)
        print(f"    {p:>6.2f}  {test_state.num_terms:>6}  {pur:>8.4f}  {bv[2]:>+8.4f}  {r:>8.4f}")

    # ── 8. Pruning Study ────────────────────────────────────────────
    print(f"\n[8] PRUNING STUDY")
    thresholds = [0.001, 0.01, 0.05, 0.10, 0.20]
    print(f"    {'Thresh':>8}  {'Before':>8}  {'After':>8}  {'Pruned':>8}  {'Error≤':>8}  {'Savings':>8}")
    print(f"    {'-'*54}")

    for thresh in thresholds:
        noisy = noiseless.copy()
        GlobalDepolarizingNoise().apply(noisy, {"p": 0.05})
        engine = PruningEngine(threshold=thresh)
        result = engine.prune(noisy)
        savings = 100 * (1 - result.terms_after / result.terms_before) if result.terms_before > 0 else 0
        print(f"    {thresh:>8.3f}  {result.terms_before:>8}  {result.terms_after:>8}  "
              f"{result.terms_pruned:>8}  {result.error_estimate:>8.4f}  {savings:>7.1f}%")

    # ── 9. Entanglement Growth Timeline ─────────────────────────────
    print(f"\n[9] ENTANGLEMENT GROWTH (qubits 0↔5, distant pair)")
    history = tracker.strength_over_time(0, 5)
    # Sample every few steps to keep output readable
    sample_points = list(range(0, len(history), max(1, len(history) // 12)))
    if (len(history) - 1) not in sample_points:
        sample_points.append(len(history) - 1)
    max_h = max(history) if max(history) > 0 else 1
    for i in sample_points:
        bar_len = int(history[i] / max_h * 30) if max_h > 0 else 0
        bar = "█" * bar_len + "░" * (30 - bar_len)
        label = tracker._snapshots[i].gate_name if i < len(tracker._snapshots) else ""
        print(f"    Step {i:>3} ({label:>4}): {history[i]:>6.2f} {bar}")

    # ── 10. Benchmark ───────────────────────────────────────────────
    print(f"\n[10] BENCHMARK: Statevector vs ACES")
    from aces.analysis.benchmarking import ACESBenchmark
    bench = ACESBenchmark()
    sv = bench.run_statevector(NUM_QUBITS, circuit)
    aces = bench.run_aces(NUM_QUBITS, circuit)
    err = bench.fidelity_error(sv, aces)

    print(f"     {'':>12} {'Time':>10}  {'Memory':>10}")
    print(f"     {'Statevector':>12} {sv.wall_time_s:>9.4f}s  {sv.peak_memory_bytes:>9}B")
    print(f"     {'ACES':>12} {aces.wall_time_s:>9.4f}s  {aces.peak_memory_bytes:>9}B")
    print(f"     Max |⟨Z⟩ error|: {err:.2e}")

    print(f"\n{'=' * 65}")
    print(f"  Done. {compiled.num_steps} gates on {NUM_QUBITS} qubits.")
    print(f"  {state.num_terms} Pauli terms tracked (vs {4**NUM_QUBITS} naive).")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
