# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 18:27:17 2024

@author: Brand



stabilizer formalism with krauss noise channels

"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from gates import *
from states import *
from stabilizers import StabState
import re
import numpy as np
import cmath
import MILP
import random
import tqdm


# Define Pauli matrices
I = Identity().matrix()
X = PauliX().matrix()
Y = PauliY().matrix()
Z = PauliZ().matrix()

# Define Clifford gates
H = Hadamard().matrix()
S = Phase().matrix()


def measure(stabs, ket):
    measures = []
    for stab in stabs:
        inner = np.dot(ket, stab.state) * stab.scalar
        measures.append(inner)
    return measures


def measure2(stabs, ket):
    measures = []
    
    for stab in stabs:
        inner = np.dot(ket, stab.state)
        norm = np.abs(inner)
        measures.append(norm**2)
        
    return measures


def measure3(stabs, ket):
    measures = []
    for stab in stabs:
        inner = np.dot(ket, stab.state) * stab.scalar
        measures.append(inner)
        
    norm = np.abs( sum(measures) )
     
    return norm**2


amplitude_dampening = lambda p: [np.array([1+0j, 0+0j, 0+0j, np.sqrt(1-p)+0j]),
                              np.array([0+0j, 0+0j,np.sqrt(p)+0j, 0+0j])]

dephasing = lambda p: [np.sqrt(p)*I,
                          np.sqrt(1-p)*Z]

partial_depolarizing = lambda p: [np.sqrt(1-(3*p)/4)*I,np.sqrt(p/4)*X,np.sqrt(p/4)*Y,np.sqrt(p/4)*Z]

pd5 =partial_depolarizing(.5)
ad25 =amplitude_dampening(.25)
d33 = dephasing(.33)
ad75=amplitude_dampening(.75)


stab = StabState(state_name='+')

cliffs = 'C:\\Users\\Brand\\Documents\\IPAM\\Research\\RipsQ\\groupCode\\cliffords.txt'




allmeasures=[]

Ks = [pd5, ad25, d33,ad75]


"""
for channel in ops:
    chs = []
    index = random.randint(0,len(channel)-1)
    #print(index)
    Ks.append(channel[index].flatten())

"""


for i in range(len(Ks)):
    noise_channel = Ks[i]
    
    for j in range(len(noise_channel)):
        krauss = noise_channel[j]
      
        milf = MILP.MILP(cliffs)
        
        milf.solve(krauss.flatten(),lambda_weight=1/3,scaling_factor=50)
        
        Ks[i][j] = milf
   


#Jad's answer     
#0.0325

#initial 0.023



    
   # print('&'*20)
    
    
#milf = MILP.MILP(cliffs)
for i in tqdm.tqdm(range(5000)):
    
    states = []

    states.append(stab)
    for noise_channel in Ks:
        
        random_krauss = random.randint(0,len(noise_channel)-1)
        
        milf = noise_channel[random_krauss]
        
       
        
       # milf.solve(krauss.flatten(),lambda_weight=1/3,scaling_factor=50)
    
        cliffs = milf.get_cliffords_of_approx()
        
       # print('len cliffs ', len(cliffs))
       # print('len states ', len(states))
        
        cs = milf.get_coeff_of_approx()
        newStab = []
       
        for i in range(len(cliffs)):
            clif = cliffs[i]
            c = cs[i]
            
            newlist = []
            for stab in states:
                cstab = StabState(stab=stab)
                
                cstab.apply_clifford(clif)
                
                cstab.scalar=c*cstab.scalar
                
                newlist.append(cstab)
                   
            newStab.extend(newlist)
            
        states=newStab
       # print('len new states ', len(states))
       # print('*'*20)
    
    
    
    measures0 = measure3(states, np.array([1,0]))
    
    if not (np.isclose(measures0,0)):
        allmeasures.append(measures0)


#print(measures0)

   # allmeasures.append(measures0)



avg = sum(allmeasures)/len(allmeasures)
print('')
print('avg ', avg)



















