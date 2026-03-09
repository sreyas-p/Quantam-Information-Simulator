# ACES — Sparse Pauli Dictionary Quantum Circuit Simulator

A Schrödinger-picture quantum circuit simulator that represents quantum states as sparse dictionaries of Pauli strings and coefficients. Includes a Heisenberg-picture Pauli Propagation simulator for comparison.

## Build & Run

### Requirements

- Python ≥ 3.9
- NumPy ≥ 1.21
- pytest ≥ 7.0 (for tests only)

### Installation

```bash
# Clone and install in editable mode
git clone https://github.com/sreyas-p/Quantam-Information-Simulator.git
cd Quantam-Information-Simulator
pip install -e .
```

Or install dependencies directly:

```bash
pip install numpy pytest
```

### Running

```bash
# Full ACES demo (6-qubit / 48-gate circuit with noise, pruning, benchmarking)
python3 -m aces.main

# ACES vs Pauli Propagation comparison demo
python3 demo.py

# Run tests
python3 -m pytest tests/ -v
```

## Project Structure

```
aces/
├── compiler/               # Circuit → Pauli transformation rules
│   ├── parser.py           # Gate sequence parsing
│   ├── gate_rules.py       # Pauli conjugation rules (Clifford + rotation)
│   └── pauli_transform.py  # CircuitCompiler, CompiledCircuit, CompiledStep
├── runtime/                # State evolution engine
│   ├── pauli_state.py      # PauliState: Dict[str, float] state representation
│   ├── noise_model.py      # Depolarizing, dephasing, global depolarizing noise
│   ├── pruning_engine.py   # Coefficient truncation for compression
│   └── entanglement_tracker.py  # Pairwise correlation tracking
├── propagation/            # Heisenberg-picture simulator
│   └── pauli_prop.py       # PauliPropagator: backward observable evolution
├── analysis/               # Metrics and benchmarking
│   ├── metrics.py          # Bloch vectors, purity, compression stats
│   └── benchmarking.py     # ACESBenchmark, PauliPropBenchmark
├── visualization/          # Text-based output
│   ├── bloch_viewer.py     # Bloch vector display
│   └── entanglement_graph.py  # Entanglement heatmap
└── main.py                 # Full ACES pipeline demo
demo.py                     # ACES vs Pauli Propagation comparison
```

## Contributions vs Libraries

### What we built (original code)

Every module in `aces/` is written from scratch. Specifically:

| Component | Description |
|---|---|
| **Pauli state representation** | Quantum states stored as `Dict[str, float]` mapping Pauli strings to coefficients. The density matrix is ρ = (1/2ⁿ) Σ cₚ·P |
| **Gate conjugation rules** | Hand-derived Pauli transformation tables for all Clifford gates (H, X, Y, Z, S, SDG, CX, CZ) and rotation gates (RX, RY, RZ) |
| **Circuit compiler** | Converts gate sequences into compiled Pauli transformation steps |
| **Noise models** | Depolarizing, dephasing, and global depolarizing noise implemented as coefficient scaling in the Pauli basis |
| **Pruning engine** | Coefficient truncation with error tracking for state compression |
| **Pauli Propagation** | Heisenberg-picture back-propagation with adjoint gate handling, per-gate noise, and truncation |
| **Benchmarking** | Statevector simulation (for ground-truth comparison) and timing infrastructure |
| **Visualization** | Text-based Bloch vector display and entanglement heatmaps |

### External libraries used

| Library | What we use it for |
|---|---|
| **NumPy** | Linear algebra in statevector benchmark (`benchmarking.py`), norm calculations in `main.py`, entanglement matrix construction in `metrics.py`. The core Pauli engine itself uses only Python builtins (`dict`, `defaultdict`, `itertools`) |
| **pytest** | Test runner (development only) |

No quantum computing frameworks (Qiskit, Cirq, PennyLane, etc.) are used. All quantum simulation logic is implemented from first principles.

## How It Works

### ACES (Schrödinger Picture)

The state |0⟩⊗ⁿ is initialized as all 2ⁿ combinations of I and Z strings, each with coefficient 1.0. Gates transform the dictionary by applying Pauli conjugation rules: for each term cₚ·P, the gate U maps P → U P U† producing one new term (Clifford) or two terms (rotation). After all gates are applied, any observable ⟨O⟩ is a single dictionary lookup.

### Pauli Propagation (Heisenberg Picture)

Instead of evolving the state forward, we evolve the observable backward. Starting from a single observable O, we apply the adjoint conjugation U†ₖ O Uₖ for each gate in reverse order. The expectation value is then the overlap of the back-propagated dictionary with the |0⟩ state. This gives the same result as ACES, but must be repeated for each observable.

### Why ACES wins for multiple observables

ACES simulates once (~0.1s) then reads N observables as O(1) dictionary lookups. Pauli Propagation runs a full back-propagation per observable (~0.05s × N). The crossover occurs around 2-4 observables; at 32 observables, ACES is ~17× faster.

## References

1. **Shao, Y., Cheng, X., & Liu, T.** "Pauli Propagation: Simulating Quantum Spin Dynamics via Operator Complexity." arXiv:2510.22311 (2025).
   — Introduces scalable Pauli propagation in the Heisenberg picture with error bounds linked to operator complexity.

2. **Gonzalez, A., Bermejo, P., & Müller, M.** "Pauli Propagation: A Computational Framework for Simulating Quantum Systems." arXiv:2505.21606 (2025).
   — Comprehensive account of Pauli propagation from bit-level implementation to high-level applications, including the relationship to Sparse Pauli Dynamics (SPD).

3. **Begušić, T., & Chan, G. K-L.** "Fast classical simulation of evidence for the utility of quantum computing before fault tolerance." arXiv:2306.16372 (2023).
   — Demonstrates that sparse Pauli dynamics can match quantum hardware results for kicked Ising dynamics using Heisenberg-picture evolution of Pauli operators.

4. **Rall, P., Liang, D., Cook, J., & Kretschmer, W.** "Simulation of qubit quantum circuits via Pauli propagation." Physical Review A, 019901 (2019).
   — Early work establishing Pauli propagation as a method for simulating quantum circuit expectation values.

## License

MIT
