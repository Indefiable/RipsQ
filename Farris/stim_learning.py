# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 06:42:23 2024

@author: Brand
"""

import stim

circuit = stim.Circuit()

# First, the circuit will initialize a Bell pair.
circuit.append("H", [0])
circuit.append("CNOT", [0, 1])

# Then, the circuit will measure both qubits of the Bell pair in the Z basis.
circuit.append("M", [0, 1])

circuit

sampler = circuit.compile_sampler()

print(sampler.sample(shots=10))

