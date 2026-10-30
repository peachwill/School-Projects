# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 07:42:20 2026

@author: willi
"""

#%% Question 1a 
import numpy as np 

def make_A(n):
    
    A =  np.zeros((n,n))#create matrix n,n of all zeros
    
    np.fill_diagonal(A,n) #set diagonal entries to n  
                     
    row , col = np.triu_indices(n, k=1)#fill the upper corner with 1s 
    #fill upper triangle with -1
    A[row, col] = -1 
    return A 
        
def make_b(n):
        b = np.ones(n) #all one vector
        
        b[1::2] = -1 #-1,-3,-5 the indices are negative
        
        return b 
  
def euclidean_norm(A): #compure norm using the eqn in assignment
        return np.sqrt(np.sum(A**2))
    
n = 4 
A = make_A(n)
b = make_b(n) 
norm_A = euclidean_norm(A)
    
print("A =")
print(A)
    
print("\nb =")
print(b) 
    
print("\n||A||=", norm_A)
#%% Question 1b 

import matplotlib.pyplot as plt 

ns = np.arange(3,18)#n = 3 to 18, doesnt inlcude the last guy 

x0 =[]
x1 =[]
x2= []
#place to store the original solutions 

x0p =[]
x1p =[]
x2p=[]
#place to store the changed soltuions, going to call them the prime ones
ks =[]
for n in ns :
    A = make_A(n)
    b= make_b(n)
    
    x = np.linalg.solve(A, b) #solves Ax = b for current n value
    #save first three xs 
    x0.append(x[0])
    x1.append(x[1])
    x2.append(x[2])              
    
    Ainv = np.linalg.inv(A)
    kappa = euclidean_norm(A) * euclidean_norm(Ainv)
    ks.append(kappa)
    #create a preturbed version of A
    Ap = np.copy(A)
    Ap[n-1,0] = Ap[n-1,0] - 1e-5
    
    xp = np.linalg.solve(Ap,b)
    
    x0p.append(xp[0])
    x1p.append(xp[1])
    x2p.append(xp[2])

#convert to array for plotting 
x0 = np.array(x0)
x1 = np.array(x1)
x2 = np.array(x2)
#plotting the arrays/data and more to fit the style of the provided graph 
x0p = np.array(x0p)
x1p = np.array(x1p)
x2p = np.array(x2p)
ks = np.array(ks)

fig, ax1 = plt.subplots()

ax1.plot(ns, x0, marker='o', label='x[0]')
ax1.plot(ns, x1, marker='o', label='x[1]')
ax1.plot(ns, x2, marker='o', linestyle='-.', label='x[2]')

ax1.plot(ns, x0p, marker='x', linestyle='--', label='x[0] pert')
ax1.plot(ns, x1p, marker='x', linestyle='--', label='x[1] pert')
ax1.plot(ns, x2p, marker='x', linestyle='--', label='x[2] pert')

ax1.set_xlabel("matrix nxn size: n")
ax1.set_ylabel("solution: x[i]")
ax1.set_xscale('log')

ax2 = ax1.twinx()
ax2.plot(ns, ks, color='red', linewidth=2, label='cond number')
ax2.set_yscale('log')
ax2.set_ylabel("cond number", color='red')
ax2.tick_params(axis='y', colors='red')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper left')

plt.title("Q1(b): sensitivity of solution and condition number vs n")
plt.show()
#%% Question 2a 
import numpy as np 
from timeit import default_timer

def make_test(n):
    #create number 21,22,.....
    A = np.arange(21,21 + n*n,dtype = float).reshape(n,n)
    #triu keeps everyhting above the diagonal 
    U = np.triu(A)
    
    b =U[0,:].copy()
    
    return U, b

def backsub1(U, bs):
    # Back substitution for an upper triangular system Ux = b
    # This version follows the lecture notes exactly (deliberately slow)

    n = bs.size
    xs = np.zeros(n)

    # Solve the last equation first
    xs[n-1] = bs[n-1] / U[n-1, n-1]

    # Work upwards from row n-2 to row 0
    for i in range(n-2, -1, -1):
        bb = 0.0
        for j in range(i+1, n):
            bb += U[i, j] * xs[j]
        xs[i] = (bs[i] - bb) / U[i, i]

    return xs   

def backsub2(U,bs):
    #same as backsub1 but uses vectorization instead of loops 
    #backsub for the upper triangle matrix
    
    
    n = bs.size 
    xs = np.zeros(n)
    
    xs[n-1] = bs[n-1] / U[n-1, n-1]
    
    for i in range(n-2,-1,-1):
        #compute teh sum using vectorization 
        bb = U[i,i+1:] @ xs[i+1:] # matrix multiplication using @ 
    
        xs[i] = (bs[i]- bb )/ U[i,i]
    return xs
    
#check n =4 
n=4
U4, b4 = make_test(n)
print("U (n=4):\n",U4)
print("b (n=4):", b4 )

x1_4 =backsub1(U4,b4)
x2_4 =backsub2(U4,b4)

print("x from backsub1:", x1_4)
print("x from backsub2:", x2_4)
print("U @ x (should equal b):", U4 @ x2_4)
print()
       
#timing o'clock, compare the speeds of backsub 1 and 2 n=5000

n=5000

U,b = make_test(n)

t0 = default_timer()
x1 = backsub1(U, b)
t1 = default_timer() 
time1 = t1 -t0 

t0 = default_timer() 
x2 = backsub2(U,b)
t1 = default_timer()
time2 = t1 - t0

print("backsub1 time:", time1)
print("backsub2 time:", time2)
print("speed factor:", time1 / time2)

print("max |difference|:", np.max(np.abs(x1 - x2)))
  
#backsub1 is ~8.59s 
#backsub2 is ~0.026s 
#backsub2 using vectorization is way faster
#%% Question 2b 
import numpy as np 

def gauss_elim(inA,inbs, pivot = False):
    #using the inclass example with some changes
    #inlcuding partial pivitong
    
    #make copies so the originals remain unchanged
    A = np.copy(inA).astype(float)
    bs = np.copy(inbs).astype(float)
    
    n = bs.size 
    
    for j in range(n-1):
        
        if pivot :
        #partial pivoting
        #find row below j with largest K and chnage postions/swap
            k = j + np.argmax(np.abs(A[j:,j]))
            
            if A[k,j] ==0:
                raise ValueError("Matrix is singular zero pivot")
            if k != j :
                A[[j,k], :] = A[[k,j], :]
                bs[[j,k]] = bs[[k,j]]
                
        if A[j,j]==0:
            raise ZeroDivisionError("Zero pivot encountered,Chagne pivot = true")
        for i in range(j+1, n):
            coeff = A[j,j]/A[i,j]
            
            A[i,j:]= A[i,j:] - coeff * A[j, j:]
            
            bs[i] = bs[i] - coeff * bs[j]
                
        #back sub with backsub2 
        #after forward elim trying to solve the upper triangle as done in earlier
        xs = backsub2(A,bs)
        return xs 
    
    A5 = np.array([[2,1,1], [1,1,-2], [1,2,1]], dtype = float)
    
    b5 = np.array([8,-2,2], dtype = float )
    
    x5 = gauss_elim(A5, b5,pivot=False)
    print("Problem (5) solution x =", x5)
    print("Check A@x =", A5 @ x5)
    print("Expected x = [4, -2, 2]\n")
                   
    #pivoting required 
    A6 = np.array([[2, 1, 1],
               [2, 1, -4],
               [1, 2, 1]], dtype=float)

    b6 = np.array([8, -2, 2], dtype=float)

    x6 = gauss_elim(A6, b6, pivot=True)
    print("Problem (6) solution x =", x6)
    print("Check A@x =", A6 @ x6)
    print("Expected x = [4, -2, 2]")
        
  
                
        
    
    
    

