# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:41:33 2024

@author: Brand


Studying how noise impacts vector representations of densitry matrices on 
the Bloch Sphere.
"""



from qiskit.quantum_info import DensityMatrix, Kraus, SuperOp
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_bloch_multivector

# Define the initial state |+> (superposition state)
initial_state = DensityMatrix.from_label('+')

# Define the Kraus operators for phase damping
p_phase_damp = 0.1
K0 = np.array([[1, 0], [0, np.sqrt(1 - p_phase_damp)]])
K1 = np.array([[0, 0], [0, np.sqrt(p_phase_damp)]])
kraus_operators = [K0, K1]

# Apply phase damping using the Kraus representation
kraus = Kraus(kraus_operators)
final_state = kraus @ initial_state

# Visualize the initial and final states on the Bloch sphere
plot_bloch_multivector(initial_state.data)
plt.title("Initial State")
plt.show()

print(final_state)
breakpoint()
plot_bloch_multivector(final_state.data)

plt.title("Final State after Phase Damping")
plt.show()

# Print the density matrices
print("Initial density matrix:")
print(initial_state.data)

print("\nFinal density matrix after phase damping:")
print(final_state.data)
