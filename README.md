# ACES - Adaptive Causal-Entropy Simulation

A compiled classical emulator for NISQ-era quantum hardware.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from aces import compile_circuit, ACESRuntime
from qiskit import QuantumCircuit

# Define your circuit
qc = QuantumCircuit(10)
qc.h(range(10))
qc.cx(0, 1)
# ... more gates

# Compile with ACES
compiled = compile_circuit(qc, measured_qubits=[0, 1, 2])

# Run with different parameters
runtime = ACESRuntime(compiled)
result = runtime.execute(params={'theta': 0.5})
print(result.expectation_values)
```

## License

MIT
