# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:32:35 2024

@author: Brand

code for generating all 24 1-qubit clifford gates via brute force

general algorithm goes as such:

hard code in {X,Y,Z,H,S}

for C in cliffords:
    for K in cliffords:
        if CK not in cliffords up to phase shift, add CK to cliffords
        
        if |cliffords| >= 24
            stop

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



def equal(C, K):
    
    K1 = np.linalg.inv(K)
    matt = np.matmul(C,K1)
    
    # this is why the code is so slow. estimating equality of matrices up to phase
    # by iterating through all possible phases, [0,2pi] / 800 and checking for 
    #equality up to error 1e^-1. 
    values = np.linspace(0, 2*np.pi, num=800, endpoint=True)
    
    for theta in values:
        phase = Identity().scale(cmath.exp(1j*theta)).matrix()
        
        if (np.allclose(matt, phase, atol=1e-1)):
            return True
    
    return False
        
    """
    Check if two 2x2 complex matrices C and K are equal up to a global phase e^{iθ}.
    
    Parameters:
    C, K (numpy.array): 2x2 numpy arrays of complex numbers.
    
    Returns:
    bool: True if C and k are equal up to a global phase, False otherwise.
   
    
    if C.shape != (2, 2) or K.shape != (2, 2):
       raise ValueError("Both C and K must be 2x2 matrices.")
       
    # Compute the determinant to find the global phase factor.
    det_C = np.linalg.det(C)
    det_K = np.linalg.det(K)
    
    if det_C == 0 or det_K == 0:
        raise ValueError("Matrices should be non-singular.")
    
    # Normalize matrices by their determinant's phase
    phase_C = np.angle(det_C)
    phase_K = np.angle(det_K)
    
        
    C_normalized = C / np.exp(1j * phase_C / 2)
    K_normalized = K / np.exp(1j * phase_K / 2)
    
    return np.allclose(C_normalized, K_normalized, atol=1e-8)
   """



I = Identity().matrix()
X = PauliX().matrix()
Y = PauliY().matrix()
Z = PauliZ().matrix()
H = Hadamard().matrix()
S = Phase().matrix()

cliffs = [X,Y,Z,H,S]

for i in range(len(cliffs)):
    print(i)
    C = cliffs[i]
    for K in cliffs:
        newMat = np.matmul(C,K)
        alreadyIn = False
        
        for cliff in cliffs:
            if equal(cliff,newMat):
                alreadyIn = True
                break
            
        if not alreadyIn:
            cliffs.append(newMat)
            print(len(cliffs))
            
    print('=======')


print(len(cliffs))

print('===================')
for cliff in cliffs:
    print(cliff)
    print('=====')

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        