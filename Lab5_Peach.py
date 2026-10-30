# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 21:13:22 2026

@author: willi
"""

#%% Question 1 Goal interpolate RUNGE function

import numpy as np 
import matplotlib.pyplot as plt

#define runge function 

def funcR(x):
    return 1.0/(1.0+25.0 * x**2)

def chebyshev_nodes(a,b,n):
   
    x = np.zeros(n)
    for j in range(n):
        x[j] = 0.5*(a + b) - 0.5*(b - a)*np.cos(j*np.pi/(n - 1))
    return x
    
def generatedata(n, f, spacing="equid", a=-1.0, b=1.0):
    
    #xnodes = node points in [a,b]
    #ynodes = f(xnodes)
    # equid evenly spaced nodes
    #cheb chebyshev nodes 
    
 
    
    if spacing.lower() == "equid":
        xnodes = np.linspace(a, b, n) #for evenly spaced nodes
    elif spacing.lower() in ["cheb", "chebyshev"]:
        xnodes = chebyshev_nodes(a, b, n)
        xnodes = np.sort(xnodes)   # good for plotting
    else:
        raise ValueError("spacing must be 'equid' or 'cheb'")

    ynodes = f(xnodes)
    return xnodes, ynodes


#lagrange interpolation 
def lagrange_eval_one(xnodes,ynodes, x):
   #evaulate the lagrange interpolation polynomail at one point x 
   
   # p(x) = sum_{k=0}^{n-1} y_k * L_k(x)
   #L_k(x) = product_{j!=k} (x - x_j)/(x_k - x_j)
   
   n = len(xnodes)
   p = 0.0
   
   for k in range(n):
       #computing L_k(x)
       Lk =1.0 
       for j in range(n):
           if j != k:
               Lk = Lk * (x - xnodes[j]) / (xnodes[k] - xnodes[j])
               
       p = p + ynodes[k] *Lk
       
   return p 

def lagrange_eval_grid(xnodes,ynodes,xgrid):
    #evaulate the lagrange polynomial at many points on the grid
    
    ygrid = np.zeros_like(xgrid, dtype = float)
    for i in range(len(xgrid)):
        ygrid[i] = lagrange_eval_one(xnodes, ynodes, xgrid[i])
    return ygrid 

a , b = -1.0, 1.0 
n = 15
itot = 400 

x_new = np.linspace(a , b , itot)

#chebyshev node case
x_cheb, y_cheb = generatedata(n, funcR, spacing="cheb", a=a, b=b)
y_cheb_new = lagrange_eval_grid(x_cheb, y_cheb, x_new)

#equidistant nodes case 
x_equi, y_equi = generatedata(n, funcR, spacing="equid", a=a, b=b)
y_equi_new = lagrange_eval_grid(x_equi, y_equi, x_new)

#plot
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

axs[0].plot(x_new, y_cheb_new, label="Lagrange Interpolant")
axs[0].plot(x_cheb, y_cheb, "o", label="Nodes")
axs[0].set_title("Chebyshev Nodes (n=15)")
axs[0].set_xlabel("x")
axs[0].set_ylabel("f(x)")
axs[0].legend()

axs[1].plot(x_new, y_equi_new, label="Lagrange Interpolant")
axs[1].plot(x_equi, y_equi, "o", label="Nodes")
axs[1].set_title("Equidistant Nodes (n=15)")
axs[1].set_xlabel("x")
axs[1].legend()

plt.tight_layout()
plt.show()

#%% Question 2 

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.linalg import solve 

# Given data from the lab 
datax = np.array([2.5, 2.8, 3.0, 3.3, 3.7, 4.1, 4.3, 4.7,
                  5.0, 5.2, 5.6, 6.1, 6.5, 6.7, 7.1, 7.2, 7.5])

datay = np.array([1.084e-17, 6.7041e-11, 1.125e-07, 2.359e-04,
                  5.749e-02, 5.188e-01, 7.865e-01, 9.919e-01,
                  1.000e+00, 9.984e-01, 8.784e-01, 2.312e-01,
                  6.329e-03, 2.359e-04, 3.579e-09, 6.704e-11,
                  1.084e-17])
      
#make the 500 desired points for plotting 
itot = 500 
xmin, xmax = 2.5, 7.5 
x_new = np.linspace(xmin, xmax, itot)

#using the same lagrange from question 1 
def lagrange_eval_one(xnodes, ynodes, x):
    n = len(xnodes)
    p = 0.0

    for k in range(n):
       Lk = 1.0
       for j in range(n):
           if j != k:
               Lk *= (x - xnodes[j]) / (xnodes[k] - xnodes[j])
       p += ynodes[k] * Lk

    return p


def lagrange_eval_grid(xnodes, ynodes, xgrid):
    #evaualte lagrange polynomial at many x values 
    
    ygrid = np.zeros_like(xgrid, dtype=float)
    for i in range(len(xgrid)):
        ygrid[i] = lagrange_eval_one(xnodes, ynodes, xgrid[i])
    return ygrid


# Evaluate Lagrange interpolant on the 500 points
y_lagr = lagrange_eval_grid(datax, datay, x_new)
    
#natural cubic spline interpolation 

# Natural spline means:
#   second derivative at the endpoints = 0
#   M0 = 0 and M_{n-1} = 0
# where M_i means second derivative at node i




def spline_second_derivatives(xnodes, ynodes):
    x = np.asarray(xnodes, dtype=float)
    y = np.asarray(ynodes, dtype=float)
    n = len(x)

    h = x[1:] - x[:-1]

    A = np.zeros((n, n), dtype=float)
    rhs = np.zeros(n, dtype=float)

    # natural boundary conditions
    A[0, 0] = 1.0
    rhs[0] = 0.0
    A[n-1, n-1] = 1.0
    rhs[n-1] = 0.0

    # interior equations
    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i, i]   = 2.0*(h[i-1] + h[i])
        A[i, i+1] = h[i]
        rhs[i] = 6.0 * ((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])

    M = solve(A, rhs)
    return M


def natural_spline_eval(xnodes, ynodes, M, xeval):
    x = np.asarray(xnodes, dtype=float)
    y = np.asarray(ynodes, dtype=float)
    xeval = np.asarray(xeval, dtype=float)

    n = len(x)
    yout = np.zeros_like(xeval, dtype=float)

    for k in range(len(xeval)):
        xv = xeval[k]

        i = np.searchsorted(x, xv) - 1
        if i < 0:
            i = 0
        if i > n-2:
            i = n-2

        h = x[i+1] - x[i]

        term1 = M[i]   * (x[i+1] - xv)**3 / (6.0*h)
        term2 = M[i+1] * (xv - x[i])**3   / (6.0*h)
        term3 = (y[i]   - M[i]*h**2/6.0)  * (x[i+1] - xv) / h
        term4 = (y[i+1] - M[i+1]*h**2/6.0)* (xv - x[i]) / h

        yout[k] = term1 + term2 + term3 + term4

    return yout

# Compute second derivatives for the spline

# Evaluate spline on the 500 points

M = spline_second_derivatives(datax, datay)

y_spline = natural_spline_eval(datax, datay, M, x_new)


fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

# Lagrange plot
axs[0].plot(x_new, y_lagr, label="Lagrange")
axs[0].plot(datax, datay, "o", label="Data")
axs[0].set_title("Lagrange Interpolation")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].legend()

# Spline plot
axs[1].plot(x_new, y_spline, label="Cubic Spline")
axs[1].plot(datax, datay, "o", label="Data")
axs[1].set_title("Natural Cubic Spline")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].legend()

plt.tight_layout()
plt.show()

#%% Question 3 trig interpolation 
import numpy as np 
import matplotlib.pyplot as plt 


#define the function 
def f(x):
    return np.exp(np.sin(2.0*x))

#build equally spaced nodes on [0,2pi]

def periodic_nodes(n): #pick evenly spaced nodes on one period
    #equally spaced node o to 2pi 
    x = np.zeros(n)
    for j in range(n):
        x[j] = 2.0*np.pi*j/n
    return x 

#trig interpolation 


def trig_interp(xnodes, ynodes, xeval):
    n = len(xnodes)
    c = np.fft.fft(ynodes) / n   # scale once

    # frequency indices: 0,1,...,n//2, -(n//2-1),..., -1  (for odd n)
    k = np.arange(n)
    k[k > n//2] -= n

    # Evaluate: p(x) = sum_k c_k * exp(i*k*x)
    # Vectorized: shape (len(xeval), n)
    E = np.exp(1j * np.outer(xeval, k))
    yout = E @ c

    return yout.real

#Make a fine grid for plotting

x_plot = np.linspace(0.0, 2.0*np.pi, 500)
y_true = f(x_plot) 

#Run for n = 11 and n = 51

for n in [11, 51]:

    # Build nodes and sample the function
    xnodes = periodic_nodes(n)
    ynodes = f(xnodes)

    # Trig interpolation
    y_interp = trig_interp(xnodes, ynodes, x_plot)

    #  Plot
    plt.figure(figsize=(7, 4.5))
    plt.plot(x_plot, y_interp, label=f"Trig Interp (n={n})")
    plt.plot(xnodes, ynodes, "o", label="Points")
    plt.plot(x_plot, y_true, "--", label="True f(x)")  # optional but helpful
    plt.title(f"Trigonometric Interpolation on [0, 2π] (n={n})")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.tight_layout()
    plt.show()
       
           
    
    
               
               
      