# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 09:31:06 2026

@author: willi
"""

#%% Question 1a 1D heat equation
import numpy as np
import matplotlib.pyplot as plt 

#parameters 
Nx = 100
L = 1.0
dt = 0.1
t_final = 61 
alpha = 2.3e-4

# Grids
x = np.linspace(0,L,Nx)
dx = x[1] - x[0]

t = np.arange(0, t_final +dt, dt)
Nt = len(t)

# check numerical stabilty 
# using Caurant stabilty criterion from notes 

k = alpha * dt / dx**2

if k  > 0.5:
    raise ValueError(
        f"Unstable choice of dt and dx: k = {k:.4f} > 0.5. "
        "Choose a smaller dt or larger dx."
        )
print(f"dx = {dx:.6f}")
print(f"dt = {dt:.6f}") 
print(f"k  = {k:.6f}  --> stable")
    
# Solution array: rows = time, columns = space
u = np.zeros((Nt,Nx))

#intial condition 
u[0,:] = 20 + 30 * np.exp(-100*(x - 0.5)**2)

# apply boundary conditions 
u[0, 0] = 20 
u[0, -1] = 20

#time stepping 
for n in range(Nt - 1):
    # Update only interior points using slicing
   u[n+1, 1:-1] = u[n, 1:-1] + k * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])
   
   #each slice is a shifted verision of the grid
   #u[n,2:] right neighbors 
   #u[n,1;-1] current point
   #u[n,:-2] left neighbors
   
   
    # fixed noundary conditions
   u[n +1 , 0] = 20 
   u[n +1, -1 ] = 20 

#times to plot 
# 0, 5, 10, 20, 30, 60 

times_to_plot = [0,5,10,20,30,60]

plt.figure(figsize=(8,5))

for ts in times_to_plot :
    idx =int(ts/dt)
    plt.plot(x, u[idx,:], label = f'time = {t[idx]: .1f}')

plt.xlabel('Rod length (m)', fontsize=18)
plt.ylabel('Temperature (C)', fontsize=18)
plt.title('1D Heat Equation Solution', fontsize=18)

plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.show()

# the graph show how temp gradients decrease over time
#at t=0 the heat is in the center, it flows from high to low regions,

#%% Question 2 Poissons Equation

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as lg 

def source_function(X, Y):
    return np.cos(10*X) - np.sin(5*Y - np.pi/4)

# parameters
Nx = 200
Ny = 100
x_min, x_max = 0.0, 2.0
y_min, y_max = 0.0, 1.0
tol = 1e-5
max_iter = 50000

x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)

dx = x[1] - x[0]
dy = y[1] - y[0]

X, Y = np.meshgrid(x, y)
F = source_function(X, Y)

# initial guess
phi = np.zeros((Ny, Nx))
phi_new = np.zeros((Ny, Nx))

error = 1.0
iteration = 0

while error > tol and iteration < max_iter:

    # Jacobi update
    #Each point is updated using the average of its neighbors + source term
    phi_new[1:-1, 1:-1] = (
        dy**2 * (phi[1:-1, 2:] + phi[1:-1, :-2])
        + dx**2 * (phi[2:, 1:-1] + phi[:-2, 1:-1])
        - dx**2 * dy**2 * F[1:-1, 1:-1]
    ) / (2 * (dx**2 + dy**2))

# (neighbors - 4*center) / grid spacing
    # boundary conditions
    # phi is 0 on all edges 
    phi_new[0, :] = 0
    phi_new[-1, :] = 0
    phi_new[:, 0] = 0
    phi_new[:, -1] = 0

    # convergence test
    error = lg.norm(phi_new - phi) / lg.norm(phi_new)

    # update old solution
    phi[:, :] = phi_new[:, :]

    iteration += 1

print(f"dx = {dx:.6f}")
print(f"dy = {dy:.6f}")
print(f"Iterations = {iteration}")
print(f"Final error = {error:.2e}")

if iteration == max_iter and error > tol:
    print("Warning: maximum iterations reached before convergence.")

plt.figure(figsize=(9, 4.5))
im = plt.imshow(
    phi,
    extent=[x_min, x_max, y_min, y_max],
    origin='lower',
    aspect='auto',
    cmap='hot'
)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution to 2D Poisson Equation')
cbar = plt.colorbar(im)
cbar.set_label(r'$\phi(x,y)$')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.show()

# A smooth field shaped by the source function
# Boundaries force the edges to zero

#%% Question 3  Poissons equation on peridic grid 

import numpy as np 
import matplotlib.pyplot as plt 
import time 

n = 800 
#dont include endpoint since it is periodic, 0 is the same as 2 *pi
x = np.linspace(0,2*np.pi, n, endpoint=False)

y = x.copy()
h = x[1] - x[0] #grid spacing

X,Y = np.meshgrid(x,y) #create full 2d grid of coords 

def source_term(X, Y):
# forcing function, it creates waves in the domain 
    return np.cos(3*X + 4*Y) - np.cos(5*X - 2*Y) 

# solve using FFT
#FFT converts function from spatial domain to frequency domain

t_start= time.time() # see how fast the fft is 

f = source_term(X,Y) #input for poissons eqn 

f_hat = np.fft.fft2(f) #convert funciton into the sum of waves, values represent frequencys 

# build indices for Fourier modes (k,l) , (k,l) reprents different waves 
k = np.arange(n)
l = np.arange(n)

K ,L = np.meshgrid(k,l,indexing='ij')

#how each wave bheaves when lapacian is applied 
denom = np.cos(2*np.pi*K/n) + np.cos(2*np.pi*L/n) -2 


phi_hat = np.zeros_like(f_hat) #solution 

#avoid dividing by zero 
#skip k,l= 0,0
mask = (K != 0) | (L != 0)

phi_hat[mask] = (h**2 /2) * f_hat[mask] / denom[mask]

#convert back to real space
#np.real remove imagenary numbers
phi = np.real(np.fft.ifft2(phi_hat))

t_end = time.time()
print(f"FFT solve time: {t_end - t_start:.4f} s")

  #plot            
fig, ax = plt.subplots(figsize=(7, 7))

img = ax.imshow(
    phi,
    extent=[0, 2*np.pi, 0, 2*np.pi],       
    origin='lower',             
    cmap='hot',                 
    aspect='equal'             
)

cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'Re $\phi(x,y)$', fontsize=16)

ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('y', fontsize=15)
ax.set_title("Q3 - Poisson equation (periodic, FFT)", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.show()

#Repeating wave patterns because of periodic boundaries cos and sin



