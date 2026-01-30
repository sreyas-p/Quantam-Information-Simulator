import numpy as np

class Qubit:
    def __init__(self, index):
        self.index = index
        self.rho = np.array([[1,0],[0,0]], dtype=complex)  # 1-qubit marginal

class CEG:
    def __init__(self, num_qubits):
        self.qubits = [Qubit(i) for i in range(num_qubits)]
        # Edge weight matrix (mutual information relevance)
        self.edge_weights = np.zeros((num_qubits, num_qubits))
        # Threshold for pruning
        self.threshold = 1e-3


def apply_2qubit_gate(ceg, q1, q2, gate_matrix):
    """
    q1, q2: indices of qubits
    gate_matrix: 4x4 unitary
    """
    # Build 2-qubit marginal
    rho_2 = np.kron(ceg.qubits[q1].rho, ceg.qubits[q2].rho)
    # Apply unitary: ρ -> U ρ U†
    rho_2 = gate_matrix @ rho_2 @ gate_matrix.conj().T
    # Reduce back to 1-qubit marginals (partial trace)
    ceg.qubits[q1].rho = rho_2.reshape(2,2,2,2).trace(axis1=1, axis2=3)
    ceg.qubits[q2].rho = rho_2.reshape(2,2,2,2).trace(axis1=0, axis2=2)
    
    # Update edge weight (mutual information proxy)
    ceg.edge_weights[q1, q2] = np.linalg.norm(rho_2 - np.kron(ceg.qubits[q1].rho, ceg.qubits[q2].rho))
    
    # Prune edges below threshold
    if ceg.edge_weights[q1, q2] < ceg.threshold:
        ceg.edge_weights[q1, q2] = 0

def measure_observable(ceg, qubit_indices, observable):
    """
    qubit_indices: list of qubits to measure
    observable: 2^n x 2^n matrix
    """
    # Build reduced density matrix of selected qubits
    rho = np.array([[1]], dtype=complex)
    for q in qubit_indices:
        rho = np.kron(rho, ceg.qubits[q].rho)
    # Expectation value
    return np.trace(rho @ observable).real


def simulate_circuit(ceg, circuit):
    """
    circuit: list of tuples (gate_matrix, qubit_indices)
    """
    for gate, qubits in circuit:
        if len(qubits) == 1:
            # Apply 1-qubit gate
            q = qubits[0]
            ceg.qubits[q].rho = gate @ ceg.qubits[q].rho @ gate.conj().T
        else:
            apply_2qubit_gate(ceg, qubits[0], qubits[1], gate)