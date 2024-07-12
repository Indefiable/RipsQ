# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:08:55 2024

@author: Brand
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


"""

We have ±P= {±X, ±Y, ±Z}. Conjugation must preserve the structure of P*, thus the
action of U ∈ C1 is completely determined by the images of X and Z. Moreover, UXU† and
UZU† must anti-commute. Thus X can go to any element of ±P*, but Z can only go to ±P*/±UXU†
Hence |C1| = 6 · 4 = 24.
"""

def equal(C, K):
    
    K1 = np.linalg.inv(K)
    matt = np.matmul(C,K1)
    
    # this is why the code is so slow. estimating equality of matrices up to phase
    # by iterating through all possible phases, [0,2pi] / 800 and checking for 
    #equality up to error 1e^-1. 
    values = np.linspace(0, 2*np.pi, num=300, endpoint=True)
    
    for theta in values:
        phase = Identity().scale(cmath.exp(1j*theta)).matrix()
        
        if (np.allclose(matt, phase, atol=1e-2)):
            return True
    
    return False


filename = 'C:\\Users\\Brand\\Documents\\IPAM\\Research\\Problem_1\\farris.npy'

jad = 'C:\\Users\\Brand\\Documents\\IPAM\\Research\\Problem_1\\cliff.npy'

jad = np.load(jad)

jad = [row.reshape(2, 2) for row in jad]

farris = np.load(filename)


counts = np.zeros(len(jad))
for i in range(len(jad)):
    mat = jad[i]
    
    for matt in farris:
        
        if equal(mat,matt):
            counts[i] +=1


print(sum(counts))

print(counts)





























