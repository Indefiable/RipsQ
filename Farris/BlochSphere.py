# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 18:50:08 2024


Bloch Sphere

@author: Brand
"""

import qutip as qt
import numpy as np
import qiskit.quantum_info as qi
import sympy as sp
import re
import cmath
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from gates import *
from states import *



def normalize(ket):
    norm = int(np.dot(ket.conj(), ket))
    print('current norm was ', norm)
    return (ket/ cmath.sqrt(norm))


def parse_input(s):
    s = s.replace('i', 'j')  # Replace 'i' with 'j'
    s = re.sub(r'sqrt\((\d+)\)', r'sp.sqrt(\1)', s)  # Replace sqrt(x) with sp.sqrt(x)
    s = re.sub(r'(\d+)/(\d+)', r'sp.Rational(\1, \2)', s)  # Handle fractions like 1/2
    return complex(eval(s))

b = qt.Bloch()


ket0 = qt.basis(2,0)
ket1 = qt.basis(2,1)


weights = input("Enter the entries making up the vector linear combo of |0> and |1>.").split()
weights = np.array([parse_input(v) for v in weights])


if(np.linalg.norm(weights) != 1):
    print('=============')
    print(weights, ' is not normal. Normalizing now.')
    print('=============')
    weights = normalize(weights)
   
    
    
state_vector = weights[0] * ket0 + weights[1] * ket1


b.add_states(state_vector)

b.show()

"""1
up = qutip.basis(2, 0)
down = qutip.basis(2, 1)
b.add_states(up)
b.add_states(down)
b.render()
b.show()

plus=1/np.sqrt(2)*(up+down)
minus = 1/np.sqrt(2)*up -1/np.sqrt(2) *down
b = qutip.Bloch()
b.add_states(plus)
b.add_states(minus)
b.render()
b.show()
"""