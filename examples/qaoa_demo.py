"""
QAOA (Quantum Approximate Optimization Algorithm) demo using ACES.

This example shows how to use ACES to efficiently simulate QAOA
for the MaxCut problem on a small graph.
"""

import numpy as np
from typing import List, Tuple


def create_maxcut_qaoa_circuit(
    edges: List[Tuple[int, int]],
    num_qubits: int,
    gamma: float,
    beta: float,
    p: int = 1
):
    """
    Create a QAOA circuit for MaxCut.
    
    Args:
        edges: List of graph edges as (u, v) tuples
        num_qubits: Number of qubits (vertices)
        gamma: Problem unitary parameter
        beta: Mixer unitary parameter
        p: Number of QAOA layers
        
    Returns:
        Qiskit QuantumCircuit
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        raise ImportError("Qiskit required. Install with: pip install qiskit")
    
    qc = QuantumCircuit(num_qubits)
    
    # Initial superposition
    for q in range(num_qubits):
        qc.h(q)
    
    # QAOA layers
    for layer in range(p):
        # Problem unitary: exp(-i * gamma * C)
        # For MaxCut: C = sum_{(u,v) in edges} (1 - Z_u Z_v) / 2
        for (u, v) in edges:
            qc.rzz(2 * gamma, u, v)
        
        # Mixer unitary: exp(-i * beta * B)
        # B = sum_i X_i
        for q in range(num_qubits):
            qc.rx(2 * beta, q)
    
    return qc


def maxcut_energy(
    bitstring: str,
    edges: List[Tuple[int, int]]
) -> float:
    """
    Compute MaxCut energy (negative of cut size) for a bitstring.
    """
    cut_size = 0
    for (u, v) in edges:
        if bitstring[u] != bitstring[v]:
            cut_size += 1
    return -cut_size


def demo_qaoa_aces():
    """
    Demonstrate ACES simulation of QAOA.
    """
    from aces import compile_circuit, ACESRuntime
    from aces.observables.sampler import estimate_counts
    
    # Define a small graph (4-vertex cycle)
    num_qubits = 4
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    
    print("=" * 60)
    print("QAOA MaxCut Demo with ACES")
    print("=" * 60)
    print(f"Graph: {num_qubits} vertices, {len(edges)} edges")
    print(f"Edges: {edges}")
    print()
    
    # Scan over gamma and beta
    best_energy = 0
    best_params = None
    
    gamma_values = np.linspace(0.1, np.pi/2, 5)
    beta_values = np.linspace(0.1, np.pi/2, 5)
    
    print("Scanning QAOA parameters...")
    
    for gamma in gamma_values:
        for beta in beta_values:
            # Create circuit
            qc = create_maxcut_qaoa_circuit(edges, num_qubits, gamma, beta, p=1)
            
            # Compile with ACES
            compiled = compile_circuit(qc, measured_qubits=set(range(num_qubits)))
            
            # Run
            runtime = ACESRuntime(compiled)
            result = runtime.execute(num_samples=100)
            
            # Estimate energy from samples
            if result.samples:
                energies = [maxcut_energy(s, edges) for s in result.samples]
                avg_energy = np.mean(energies)
                
                if avg_energy < best_energy:
                    best_energy = avg_energy
                    best_params = (gamma, beta)
    
    print(f"\nBest parameters: gamma={best_params[0]:.3f}, beta={best_params[1]:.3f}")
    print(f"Best average energy: {best_energy:.3f}")
    print(f"Best cut size: {-best_energy:.1f} / {len(edges)} (maximum possible)")
    
    # Final run with best parameters
    print("\nFinal run with best parameters:")
    qc = create_maxcut_qaoa_circuit(edges, num_qubits, best_params[0], best_params[1], p=1)
    compiled = compile_circuit(qc, measured_qubits=set(range(num_qubits)))
    runtime = ACESRuntime(compiled)
    result = runtime.execute(num_samples=1000)
    
    from collections import Counter
    counts = Counter(result.samples)
    print("\nTop 5 bitstrings:")
    for bitstring, count in counts.most_common(5):
        cut = -maxcut_energy(bitstring, edges)
        print(f"  {bitstring}: count={count}, cut_size={cut}")
    
    print()
    print("=" * 60)

    return best_energy, best_params


def demo_qaoa_sweep():
    """
    Demonstrate parameter sweep using ACES.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from aces import compile_circuit, ACESRuntime
    
    print("\n" + "=" * 60)
    print("QAOA Parameter Sweep Demo")
    print("=" * 60)
    
    # Parametric circuit
    num_qubits = 3
    edges = [(0, 1), (1, 2)]
    
    gamma = Parameter('gamma')
    beta = Parameter('beta')
    
    qc = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        qc.h(q)
    for (u, v) in edges:
        qc.rzz(2 * gamma, u, v)
    for q in range(num_qubits):
        qc.rx(2 * beta, q)
    
    # Compile once
    compiled = compile_circuit(qc, measured_qubits={0, 1, 2})
    runtime = ACESRuntime(compiled)
    
    print(f"Compiled circuit: {compiled.num_gates} gates")
    print(f"Parameters: {compiled.parameter_names}")
    print()
    
    # Sweep gamma at fixed beta
    beta_fixed = 0.5
    gamma_values = np.linspace(0, np.pi, 10)
    
    print("Sweeping gamma at beta=0.5:")
    for g in gamma_values:
        result = runtime.execute(
            params={'gamma': g, 'beta': beta_fixed},
            observables=['Z0', 'Z1', 'Z0Z1']
        )
        print(f"  gamma={g:.2f}: <Z0>={result.expectation('Z0'):.3f}, "
              f"<Z0Z1>={result.expectation('Z0Z1'):.3f}")
    
    print("=" * 60)


if __name__ == "__main__":
    demo_qaoa_aces()
    demo_qaoa_sweep()
