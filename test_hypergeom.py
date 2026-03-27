# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.special import hyp1f1
from scipy.special import hyperu
from scipy.special import gamma
import matplotlib.pyplot as plt

#### Physical problem parameters
mrho=128
ell=10e3
rhomin=ell
rhomax=2*ell
rho=np.linspace(rhomin,rhomax,mrho)

plt.subplots(1,2,dpi=150)
for m in range(1,2):
    #### Scaled parameters
    m2=m**2
    a=1/ell
    b=m2
#    z=-rho/ell
    z=rho/ell
    c1=2*m+1
    c2=m

    #### Functions
    M=hyp1f1(c1,c2,z)
    U=hyperu(c1,c2,z)
    R1=rho**m*M
    R2=rho**m*U
        
    plt.subplot(1,2,1)
    plt.plot(rho,M)
    plt.plot(rho,U)
    plt.legend(("$M=_1F_1$","$U$"))
    plt.title("Confluent Hypergeometric Functions")
    #plt.xlabel("$\rho$")
    
    plt.subplot(1,2,2)
    plt.plot(rho,R1/R1.max())
    plt.plot(rho,R2/R2.max())
    plt.legend(("$R_1$","$R_2$"))
    plt.title("Radial Solution Functions")
    #plt.xlabel("$\rho$")
