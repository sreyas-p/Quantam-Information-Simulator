"""
VQE (Variational Quantum Eigensolver) demo using ACES.

This example shows how to use ACES to simulate VQE
for finding the ground state energy of a simple Hamiltonian.
"""

import numpy as np
from typing import List, Tuple


def create_vqe_ansatz(
    num_qubits: int,
    params: List[float],
    depth: int = 2
):
    """
    Create a hardware-efficient VQE ansatz.
    
    Args:
        num_qubits: Number of qubits
        params: List of variational parameters
        depth: Number of layers
        
    Returns:
        Qiskit QuantumCircuit
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        raise ImportError("Qiskit required. Install with: pip install qiskit")
    
    qc = QuantumCircuit(num_qubits)
    param_idx = 0
    
    for layer in range(depth):
        # Rotation layer
        for q in range(num_qubits):
            if param_idx < len(params):
                qc.ry(params[param_idx], q)
                param_idx += 1
            if param_idx < len(params):
                qc.rz(params[param_idx], q)
                param_idx += 1
        
        # Entangling layer
        for q in range(0, num_qubits - 1, 2):
            qc.cx(q, q + 1)
        for q in range(1, num_qubits - 1, 2):
            qc.cx(q, q + 1)
    
    return qc


def h2_hamiltonian() -> List[Tuple[float, str]]:
    """
    Simplified H2 Hamiltonian (2 qubits).
    
    H = c0*I + c1*Z0 + c2*Z1 + c3*Z0Z1 + c4*X0X1 + c5*Y0Y1
    """
    # Approximate coefficients for H2 at equilibrium distance
    return [
        (-0.8105, "I"),      # Identity (energy offset)
        (0.1714, "Z0"),
        (0.1714, "Z1"),
        (0.1686, "Z0Z1"),
        (0.0454, "X0X1"),
        (0.0454, "Y0Y1"),
    ]


def compute_hamiltonian_energy(
    hamiltonian: List[Tuple[float, str]],
    result
) -> float:
    """Compute Hamiltonian expectation from ACES result."""
    energy = 0.0
    
    for coeff, term in hamiltonian:
        if term == "I":
            energy += coeff
        else:
            energy += coeff * result.expectation_values.get(term, 0.0)
    
    return energy


def demo_vqe_aces():
    """
    Demonstrate VQE simulation with ACES.
    """
    from aces import compile_circuit, ACESRuntime
    from scipy.optimize import minimize
    
    print("=" * 60)
    print("VQE Demo with ACES")
    print("=" * 60)
    
    num_qubits = 2
    depth = 2
    num_params = 2 * num_qubits * depth
    
    hamiltonian = h2_hamiltonian()
    
    print(f"System: H2 molecule simulation (simplified)")
    print(f"Qubits: {num_qubits}")
    print(f"Ansatz depth: {depth}")
    print(f"Parameters: {num_params}")
    print()
    print("Hamiltonian:")
    for coeff, term in hamiltonian:
        print(f"  {coeff:+.4f} * {term}")
    print()
    
    # Observables needed
    observables = [term for _, term in hamiltonian if term != "I"]
    
    iteration = [0]
    
    def energy_function(params):
        """Objective function for VQE."""
        qc = create_vqe_ansatz(num_qubits, params.tolist(), depth)
        
        compiled = compile_circuit(qc, measured_qubits=set(range(num_qubits)))
        runtime = ACESRuntime(compiled)
        result = runtime.execute(observables=observables)
        
        energy = compute_hamiltonian_energy(hamiltonian, result)
        
        iteration[0] += 1
        if iteration[0] % 10 == 0:
            print(f"  Iteration {iteration[0]}: energy = {energy:.6f}")
        
        return energy
    
    # Initial parameters
    np.random.seed(42)
    init_params = np.random.uniform(-np.pi, np.pi, num_params)
    
    print("Running VQE optimization...")
    print(f"Initial energy: {energy_function(init_params):.6f}")
    print()
    
    # Optimize
    result = minimize(
        energy_function,
        init_params,
        method='COBYLA',
        options={'maxiter': 100}
    )
    
    print(f"\nOptimization complete!")
    print(f"Final energy: {result.fun:.6f}")
    print(f"Iterations: {iteration[0]}")
    
    # Compare with known value
    exact_ground_state = -1.137  # Approximate H2 ground state energy
    print(f"\nReference ground state energy: ~{exact_ground_state:.3f}")
    print(f"Error: {abs(result.fun - exact_ground_state):.4f}")
    
    print("=" * 60)
    
    return result


def demo_vqe_parameter_landscape():
    """
    Visualize VQE energy landscape.
    """
    from aces import compile_circuit, ACESRuntime
    
    print("\n" + "=" * 60)
    print("VQE Energy Landscape")
    print("=" * 60)
    
    num_qubits = 2
    hamiltonian = h2_hamiltonian()
    observables = [term for _, term in hamiltonian if term != "I"]
    
    # Simple 2-parameter ansatz
    def energy_at(theta1, theta2):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.ry(theta1, 0)
        qc.ry(theta2, 1)
        qc.cx(0, 1)
        qc.ry(theta1, 0)
        qc.ry(theta2, 1)
        
        compiled = compile_circuit(qc, measured_qubits={0, 1})
        runtime = ACESRuntime(compiled)
        result = runtime.execute(observables=observables)
        
        return compute_hamiltonian_energy(hamiltonian, result)
    
    # Scan
    theta_range = np.linspace(-np.pi, np.pi, 11)
    
    print("Energy landscape (theta1 x theta2):")
    print()
    print("theta2:  " + "  ".join(f"{t:+.1f}" for t in theta_range[::2]))
    print("-" * 60)
    
    for t1 in theta_range[::2]:
        energies = [energy_at(t1, t2) for t2 in theta_range[::2]]
        row = "  ".join(f"{e:+.2f}" for e in energies)
        print(f"{t1:+.1f} | {row}")
    
    print("=" * 60)


if __name__ == "__main__":
    demo_vqe_aces()
    demo_vqe_parameter_landscape()
