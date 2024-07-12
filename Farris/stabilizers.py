# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:25:11 2024

@author: Brand

stabilizer state class

"""



import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from gates import *
from states import *
import re
import numpy as np
import cmath

# Define Pauli matrices
I = Identity().matrix()
X = PauliX().matrix()
Y = PauliY().matrix()
Z = PauliZ().matrix()

# Define Clifford gates
H = Hadamard().matrix()
S = Phase().matrix()

class StabState:
    
    def __init__(self, state_name='0', recordState=True):
        self.record=recordState
        self.state_name = state_name
        if(RecordState):
            self.state = self.initialize_state(state_name)
        else:
            self.state=None
            
        self.stabilizer_group = self.initialize_stabilizer_group(state_name)
    
    
    
    def initialize_state(self, state_name):
        
        if state_name == '0':
            return np.array([1, 0], dtype=complex)
        elif state_name == '1':
            return np.array([0, 1], dtype=complex)
        elif state_name == '+':
            return np.array([1, 1], dtype=complex) / np.sqrt(2)
        elif state_name == '-':
            return np.array([1, -1], dtype=complex) / np.sqrt(2)
        elif state_name == '+i':
            return np.array([1, 1j], dtype=complex) / np.sqrt(2)
        elif state_name == '-i':
            return np.array([1, -1j], dtype=complex) / np.sqrt(2)
        else:
            raise ValueError("Invalid state name")
    
    
    def initialize_stabilizer_group(self, state_name):
        if state_name == '0':
            return [Z]
        elif state_name == '1':
            return [-Z]
        elif state_name == '+':
            return [X]
        elif state_name == '-':
            return [-X]
        elif state_name == '+i':
            return [Y]
        elif state_name == '-i':
            return [-Y]
        else:
            raise ValueError("Invalid state name")
    
    def apply_clifford(self, gate):
        # Apply the gate to the state
        if self.record:
            self.state = gate @ self.state
        
        # Update the stabilizer group
        self.stabilizer_group = [gate @ stab @ np.linalg.inv(gate) for stab in self.stabilizer_group]
    
    def __str__(self):
        return f"State: {self.state_name}\nStabilizer Group: {[stab for stab in self.stabilizer_group]}"
    
# Example usage
state = StabState('+')
print(state)
print("Applying Hadamard gate...")
state.apply_clifford(H)
print(state)


