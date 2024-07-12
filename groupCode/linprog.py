# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:24:10 2024

@author: Brand
"""
import numpy as np
from scipy.optimize import linprog
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from gates import *
from states import *
import math
qtype=np.complex64


# Define the given 2x2 complex matrix K
K = np.array([[1+1j, 2+2j],
              [3+3j, 4+4j]])

# Define the set of given 2x2 complex matrices P_i
P1 = Hadamard().matrix()
P2 = Identity().matrix()
P3 = PauliX().matrix()
P4 = PauliY().matrix()
P5 = PauliZ().matrix()
P6 = Phase().matrix()
#more gates?

P = np.array([P1, P2, P3, P4, P5, P6])  # Stack matrices P_i


# Number of matrices
n = P.shape[0]


# Separate real and imaginary parts of K and P matrices
K_real = K.real.flatten()
K_imag = K.imag.flatten()
P_real = np.array([Pi.real.flatten() for Pi in P])
P_imag = np.array([Pi.imag.flatten() for Pi in P])


# Objective function: minimize the sum of auxiliary variables for real and imaginary parts
c = np.hstack([np.zeros(2*n), np.ones(2*n)])


# Inequality constraints to handle absolute values of c_i_real and c_i_imag
A_ub = np.vstack([
    np.hstack([np.eye(2*n), -np.eye(2*n)]),
    np.hstack([-np.eye(2*n), -np.eye(2*n)])
])
b_ub = np.zeros(4 * n)


# Equality constraints for the decomposition K = sum(c_i * P_i)
A_eq_real = np.hstack([P_real.T, np.zeros((P_real.shape[1], 2*n))])
A_eq_imag = np.hstack([P_imag.T, np.zeros((P_imag.shape[1], 2*n))])


A_eq = np.vstack([A_eq_real, A_eq_imag])
b_eq = np.hstack([K_real, K_imag])

# Bounds for variables c_i_real, c_i_imag, and u_i (u_i are non-negative)
bounds = [(None, None)] * (2*n) + [(0, None)] * (2*n)

# Solving the linear programming problem
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

# Printing the results
if res.success:
    c_opt_real = res.x[:n]
    c_opt_imag = res.x[n:2*n]
    c_opt = c_opt_real + 1j * c_opt_imag
    print("Optimal coefficients:", c_opt)
    print("Reconstructed K:")
    K_reconstructed = sum(c_opt[i] * P[i] for i in range(n))
    print(K_reconstructed)
else:
    print("No solution found:", res.message)

