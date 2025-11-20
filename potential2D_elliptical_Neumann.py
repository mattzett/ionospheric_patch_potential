#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 12:01:49 2025

@author: zettergm
"""


###############################################################################
def plot_grad_region(param,x,rho0,drho):
    minEy=np.min(param)
    maxEy=np.max(param)
    plt.plot([rho0,rho0],[minEy,maxEy],'--')
    plt.plot([rho0+drho,rho0+drho],[minEy,maxEy],'--')
    
    plt.plot([-rho0,-rho0],[minEy,maxEy],'--')
    plt.plot([-rho0-drho,-rho0-drho],[minEy,maxEy],'--')    
###############################################################################


###############################################################################
# Main program to simulate one elliptical patch
import numpy as np
from potential_parameterize_ellipse import solve_elliptic_neumann, plot_results
import matplotlib.pyplot as plt

# parameters of problem:
Ey0=-0.05              # background field in which the object is immersed
Ex0=0.0
n0=4e11                # density at center of structure
n1=2e11                # background density
a=50e3                 # radius of structure along semiminor axis
d=8.0                  # ratio of semimajor to semiminor axes
b=a*d                  # semimajor axis
c=np.sqrt(b**2-a**2)   # elliptic ecccentricity
L=50e3                 # gradient scale length at structure edge

edgedist=4*a-a           # tests suggest this boundary is sufficiently far away from structure edge along semimajor axis
xmax=4*a;                # x extent
ymax=a*d + edgedist;     # y extent
lx=128
ly=256

# Run the solve
x,y,Phi,Ex,Ey,n,ddist = solve_elliptic_neumann(xmax,ymax,lx,ly,a,b,Ex0,Ey0,n0,n1,L)
plot_results(x,y,Ex0,Ey0,n,Phi,Ex,Ey,a,b)
###############################################################################


###############################################################################
# Evaluate gradient scale lengths
[dndx,dndy]=np.gradient(n,x,y)
maggradn=np.sqrt(dndx**2+dndy**2)
for i in range(0,lx):
    for j in range(0,ly):
        if maggradn[i,j] < 1e4:
            maggradn[i,j]=np.nan      # if |gradient| is too small NaN the data
Lprime=n/maggradn
            
plt.figure()
plt.pcolormesh(x,y,Lprime.transpose())
plt.title("n/| grad n |")
plt.clim([10e3, 60e3])
plt.colorbar()
ax=plt.gca()
ax.set_aspect("equal")
###############################################################################


###############################################################################
# Compute a velocity field assuming B is in -z direction (NH)
B=50000e-9
vx=-Ey/B
vy=Ex/B

plt.subplots(1,2)

plt.subplot(1,2,1)
plt.pcolormesh(x,y,vx.transpose(),shading='auto')
plt.colorbar()
plt.title('$v_x$')

plt.subplot(1,2,2)
plt.pcolormesh(x,y,vy.transpose(),shading='auto')
plt.colorbar()
plt.title('$v_y$')


# Extract behavior along the y=0 line
Eyctrline=(Ey[:,ly//2]-Ey0)
plt.figure()
plt.plot(x,Eyctrline)
plot_grad_region(Eyctrline,x,a,ddist)

vxctrline=vx[:,ly//2]
plt.figure()
plt.plot(x,vxctrline)
plot_grad_region(vxctrline,x,a,ddist)

nctrline=n[:,ly//2]
plt.figure()
plt.plot(x,nctrline)
plot_grad_region(nctrline,x,a,ddist)

###############################################################################


###############################################################################
# Compute terms contributing to variations in density gradient with time
[dndx,dndy]=np.gradient(n,x,y)
maggradn=np.sqrt(dndx**2+dndy**2)

fluxx=n*vx
fluxy=n*vy
[dfluxxdx,_]=np.gradient(fluxx,x,y)
[_,dfluxydy]=np.gradient(fluxy,x,y)
divflux=dfluxxdx + dfluxydy

[graddivfluxx,graddivfluxy]=np.gradient(divflux,x,y)

dgradndt=-1/maggradn * (dndx*graddivfluxx + dndy*graddivfluxy)

for i in range(0,lx):
    for j in range(0,ly):
        if maggradn[i,j] < 1e4:
            dgradndt[i,j]=np.nan      # if |gradient| is too small NaN the data

plt.figure()
plt.pcolormesh(x,y,dgradndt.transpose(),shading='auto')
plt.colorbar()
plt.title("$\partial / \partial t ( grad n )$ in observer frame")



fluxx=n*(vx+Ey0/B)
fluxy=n*vy
[dfluxxdx,_]=np.gradient(fluxx,x,y)
[_,dfluxydy]=np.gradient(fluxy,x,y)
divflux=dfluxxdx + dfluxydy

[graddivfluxx,graddivfluxy]=np.gradient(divflux,x,y)

dgradndt=-1/maggradn * (dndx*graddivfluxx + dndy*graddivfluxy)

for i in range(0,lx):
    for j in range(0,ly):
        if maggradn[i,j] < 1e4:
            dgradndt[i,j]=np.nan      # blank region where |gradient| is too small to be of interest

plt.figure()
plt.pcolormesh(x,y,dgradndt.transpose(),shading='auto')
plt.colorbar()
plt.title("$\partial / \partial t ( grad n )$ in local plasma frame")
plt.clim((-60000,60000))
###############################################################################
