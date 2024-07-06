# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:42:58 2024

@author: Farris

class based structure for BlochSphere.py
"""

import qutip as qt
import numpy as np
import sympy as sp
import re
import cmath
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from gates import *
from states import *


class Qubit:
   
    def __init__(self, state_vector=None, create=False):
       
        if create:
            weights_input = input("Enter the entries making up the vector linear combo of |0> and |1>: ").split()
            weights = np.array([self.parse_input(v) for v in weights_input])
            
            self.state_vector = weights
            print(self.state_vector)
            print('==="')
            
            if np.linalg.norm(weights) != 1:
                self.normalize()
            return
        
        if state_vector is None:
            self.state_vector = None
        else:
            self.state_vector = state_vector
        
    
    def normalize(self):
        if self.state_vector is None:
            raise ValueError("Qubit state vector is not initialized.")
        
        norm = np.linalg.norm(self.state_vector)
        if norm == 0:
            raise ValueError("Cannot normalize the zero vector.")
        
        self.state_vector /= norm
        
        return 
    
    
    def set_state_vector(self, state_vector):
        self.state_vector = state_vector
    
    
    def show_bloch_sphere(self):
        if self.state_vector is None:
            raise ValueError("Qubit state vector is not initialized.")
        
        b = qt.Bloch()
        ket0 = qt.basis(2, 0)
        ket1 = qt.basis(2, 1)
        
        state_vector_qt = self.state_vector[0] * ket0 + self.state_vector[1] * ket1
        b.add_states(state_vector_qt)
        b.show()
        
    def parse_input(self, s):
        s = s.replace('i', 'j')  # Replace 'i' with 'j'
        s = re.sub(r'sqrt\((\d+)\)', r'sp.sqrt(\1)', s)  # Replace sqrt(x) with sp.sqrt(x)
        s = re.sub(r'(\d+)/(\d+)', r'sp.Rational(\1, \2)', s)  # Handle fractions like 1/2
        return complex(eval(s))
        
    

class Simulator:
   def __init__(self):
       print('nothing yet')




if __name__ == "__main__":
    qubit = Qubit(create=True)
   
    qubit.show_bloch_sphere()
