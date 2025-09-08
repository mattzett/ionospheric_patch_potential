#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 19:23:01 2025

@author: zettergm
"""

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Calculate the first N terms of the series solution for radial part of the 
#   equation
###############################################################################
def radial_soln(rho,a0,L,m,N):
    from scipy.special import factorial
    from numpy import ones
    
    R=ones(rho.shape)
    for j in range(1,N+1):    # 1 to N
        prod=ones(rho.shape)
        for k in range(0,j):     # 0 to j-1
            prod *= m**2 - k**2
        R+= rho**j * L**j/factorial(j) * prod
        
    return R
###############################################################################


m=1
a0=1.0
L=2.0
rhomin=0.0
rhomax=4.0
lrho=96

rho=np.linspace(rhomin,rhomax,lrho)
N=10

plt.figure()
for j in range(0,N):
    R=radial_soln(rho,a0,L,m,j)
    print("Min/max R:  ",R.min(), R.max())
    plt.plot(rho,R)
    