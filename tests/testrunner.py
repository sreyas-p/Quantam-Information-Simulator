"""
ACES Demo — Bell State Preparation & Analysis
"""

import numpy as np
from aces import SingleQubitRDM, TwoQubitRDM, GateLibrary, CausalEntanglementGraph
from aces.core.gates import NoiseLibrary

# -- 1. Prepare a Bell state: H(0) → CNOT(0,1) --

state = {0: SingleQubitRDM.zero_state(0), 1: SingleQubitRDM.zero_state(1)}
correlators = {}

GateLibrary.H.apply_to_rdm(state, correlators, (0,))         # |0⟩ → |+⟩
GateLibrary.CX.apply_to_rdm(state, correlators, (0, 1))      # |+0⟩ → Bell |Φ+⟩

bell = correlators[(0, 1)]
print("Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2")
print(f"  Concurrence: {bell.concurrence():.3f}")             # 1.0 = max entanglement
print(f"  Mutual info:  {bell.mutual_information():.3f} bits") # 2.0 for Bell state
print(f"  Product state? {bell.is_product_state()}")           # False

# -- 2. Individual qubits look random when entangled --

for q in [0, 1]:
    rdm = state[q]
    print(f"\n  Qubit {q}: P(0)={rdm.probability_zero():.2f}, P(1)={rdm.probability_one():.2f}, purity={rdm.purity:.2f}")

# -- 3. Track causal structure --

ceg = CausalEntanglementGraph(num_qubits=2, measured_qubits={0, 1})
ceg.add_edge(0, 1, weight=1.0)
print(f"\nCausal graph: {ceg}")

# -- 4. Add noise and see entanglement degrade --

print("\nDepolarizing noise on qubit 0:")
for p in [0.0, 0.1, 0.5]:
    s = {0: SingleQubitRDM.zero_state(0), 1: SingleQubitRDM.zero_state(1)}
    c = {}
    GateLibrary.H.apply_to_rdm(s, c, (0,))
    GateLibrary.CX.apply_to_rdm(s, c, (0, 1))
    NoiseLibrary.DEPOLARIZING.apply(s[0], {'p': p})
    print(f"  p={p:.1f} → purity(q0)={s[0].purity:.3f}")
