# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 08:27:33 2024

@author: Brand

Unitary decomposition implementation


"""

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from gates import *
from states import *
import re
import numpy as np





def parse_input(s):
    s = s.replace('i', 'j')  # Replace 'i' with 'j'
    s = re.sub(r'sqrt\((\d+)\)', r'sp.sqrt(\1)', s)  # Replace sqrt(x) with sp.sqrt(x)
    s = re.sub(r'(\d+)/(\d+)', r'sp.Rational(\1, \2)', s)  # Handle fractions like 1/2
    return complex(eval(s))



I = Identity()
X = PauliX()
Y = PauliY()
Z = PauliZ()

basis = [I, X, Y, Z,
         I.scale(-1), X.scale(-1), Y.scale(-1), Z.scale(-1),
         I.scale(1j), X.scale(1j), Y.scale(1j), Z.scale(1j),
         I.scale(-1j),X.scale(-1j),Y.scale(-1j),Z.scale(-1j)]



"""
e1 = I.add(Z).scale(0.5)
e2 = X.add(Y.scale(1j)).scale(0.5)
e3 = X.add(Y.scale(-1j)).scale(0.5)
e4 = I.add(Z.scale(-1)).scale(0.5)


U=[a,b] = 1/2 [ (a+d)I + (b+c)X + (b+c)iY + (a-d)Z  ]
  [c,d]
  
  
For manual input to test various matrix decompositions, use these three lines
#weights = input("Enter the four elements of an Imaginary matrix").split()
#weights = [parse_input(v) for v in weights]
#K = Gate(matrix = np.array([[weights[0],weights[1]], [weights[2],weights[3]]]))


For phase damping, you have these two K matrices based on an input p
K0 = √pI and K1 = √1 − pZ
"""

#p = float(input('enter the probability used for K0 = √pI and K1 = √(1 − p)Z: '))
#delta = float(input('enter the delta (error): '))
p=.2
delta=.2

K0 = I.scale(np.sqrt(p))
K1 = Z.scale(np.sqrt(1-p))
K3 = Gate(matrix=np.array([[1+1j, 0],[0,2]]))


rights = []
diffs = []
matss = []

U=K3
print('===== U is ====')
print(U.matrix())
print('==========')
a = U.matrix()[0][0]
b = U.matrix()[0][1]
c = U.matrix()[1][0]
d = U.matrix()[1][1]

#U=[a,b] = 1/2 [ (a+d)I + (b+c)X + (b+c)iY + (a-d)Z  ]
#  [c,d]

for M in basis:
    a = np.trace(np.dot(np.conj(M.matrix().T), U)) / np.trace(np.dot(np.conj(M.matrix().T), M.matrix()))
    c.append(a)
    

remade = Gate()

print('=== c is ===')
print(c)
print('======')

for i in range(len(c)):
    remade=remade.add(basis[i].scale(c[i]))

breakpoint()
#c = np.array([(a+d)/2,(b+c)/2, ((b+c)/2)*1j, (a-d) / 2], dtype=qtype)

#c = np.array([1+2j, 0, 1j, 0])
mats = [I,X,Y,Z]

Cnorm = np.linalg.norm(c)
#print(Cnorm)

k = (Cnorm / delta) ** 2
k = math.ceil(k)

#print('==========')
#print('we will be sampling k = (', Cnorm,' / ', delta, ')^2 = ', k, ' matrices')
#print('==========')
#randomly sampling k generator indices with weights |c[i]| / Cnorm
    
indices = np.arange(0, len(c))  # Creates an index array [0, a+1, ..., len(c)-1]

probabilities = [(np.abs(c[i]) / Cnorm) for i in range(len(c))]
probabilities = probabilities / (sum(probabilities))

print('=== probabilities are ===')
print(probabilities)
print('======')

for i in range(200):
    w = np.random.choice(indices, size=k, p=probabilities)
    
    #print('=== w is ===')
    #print(w)
    #print('======')
    
    omega = Gate()
    
    for index in w:
        omega = omega.add(mats[index].scale(Cnorm/k))
    
    diff = U.matrix() - omega.matrix()
    
    normDiff = np.linalg.norm(diff, ord=1)
    right = 1 + (Cnorm / delta) ** 2
    
    matss.append(omega.matrix())
    diffs.append(normDiff)
    rights.append(right)
    
    
    
"""
print('original matrix:')
print(U.matrix())
print('random sampling process gave us:')
print('====')
print(omega.matrix())
print('====')
print('with error : ' , normDiff)
print('========')
print(k, ' <= ', right)
"""

avgMat = sum(matss) / len(matss)
avg = sum(diffs) / len(diffs)

print('avg', avg)

avgR = sum(rights) / len(rights)

print('avg matrix: ' ) 
print(avgMat)

print('average right hand side', avgR)







































