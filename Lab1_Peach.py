# -*- coding: utf-8 -*-
"""
ENPH 213 Assignment 1 
1/10/26
Will Peach 

"""


#%%Question 1a Symbolic Derivitive and Plotting
import sympy as sm
from sympy.abc import x 
import numpy as np
import math as m 
import matplotlib.pyplot as plt

f = sm.exp(sm.sin(2*x)) #define the function sybolically 
df = sm.diff(f,x)
print(sm.diff(f, x)) #print first derivitive 
d2f = sm.diff(df,x)
print (sm.diff(df,x)) #print second derivitive 

#lambda functions, creates a fucntion that can take arrays and turn them into arrays 
func= sm.lambdify(x, f,   'numpy')
dfunc = sm.lambdify(x, df,   'numpy')
d2func = sm.lambdify(x, d2f,   'numpy')

xs = np.linspace(0, 2*m.pi, 200)

#print(len(xs), xs[0], xs[len(xs)-1] - 2*m.pi) #print array
val_f = func(xs)
val_df = dfunc(xs)
val_d2f = d2func(xs)

#plot the array

plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(10,6))

#plot arrays making changes to the provided code  
fig = plt.figure(figsize=(8,5))
lw=2
ax = fig.add_axes([.15, .15, .75, .75])

#plot f,f',and f" change colours and create a label, using the code from onq with small changes
plt.plot(xs, val_f, 'r',linewidth=lw, label=r"$f(x)$")
plt.plot(xs, val_df, 'b',linewidth=lw, label=r"$f'(x)$")
plt.plot(xs, val_d2f, 'g',linewidth=lw, label=r"$f''(x)$")

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend(loc='best') 
ax.set_xlim(0,2*m.pi)
plt.title(r"$f(x)=e^{\sin(2x)}$ and its first and second derivatives")
plt.show()


#%% Question 1b 
import numpy as np 
import matplotlib.pyplot as plt 
import math as m 

#forward differencing function returns derivitive
def forward(f, x, h):
    return (f(x+h) - f(x))/h
#central differencing derivitive using formula from class
def central(f,x,h):
    return (f(x+h)-f(x-h))/(2*h)

#populate forward derivative using the forward fuction defined earlier

def populate_Forward(f, x_array, h):
    d = np.zeros(len(x_array), dtype=float)
    
    for i in range(len(x_array)):
        x = x_array[i]
        
        if x + h <= x_array[-1]:
            d[i] = forward(f, x, h)
        else:
            d[i] = np.nan          
    return d

#populate central derivitve
def populate_central(f, x_array, h):
    d = np.zeros(len(x_array), dtype=float)
    
    for i in range(len(x_array)):
        x = x_array[i]
        
        if (x - h >= x_array[0]) and (x + h <= x_array[-1]):
            d[i] = central(f, x, h) #using the central function defined earlier
        else:
            d[i] = np.nan
            
    return d
hs = [0.4,0.1]
for h in hs:
    xs_h = np.arange(0, 2*np.pi + h,h)
    
    df_true = dfunc(xs_h) #derivative 
    
    df_fwd = populate_Forward(func, xs_h,h)
    df_cen = populate_central(func, xs_h, h)
    lw=2
    plt.figure(figsize=(10,6))
    plt.plot(xs_h, df_true, linewidth=lw, label="Analytical f'(x)")
    plt.plot(xs_h, df_fwd, linewidth=lw, label=f"Forward diff h={h}")
    plt.plot(xs_h, df_cen, linewidth=lw, label=f"Central diff h={h}")
    
    plt.xlabel("x")
    plt.ylabel("f'(x)")
    plt.title(f"Finite-difference derivative vs analytical (h = {h})")
    plt.legend()
    plt.xlim(0, 2*np.pi)
    plt.show()
#the graphing code is mainly the provided graphing code with changes to variables,colour, and titles

#%% Question 2  
#Using built in legendre fucntion in SciPy
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre



nsteps = 200 
xs = [i/nsteps for i in range(-nsteps+1, nsteps)]

#looking at the 8th and 9th derivitive we get the legendre polynomial functions 

p8 = legendre(8)
p9 = legendre(9)

#evauate p8 and p9 at xs 

val_P8 = p8(xs)
val_P9 = p9(xs)

#print what the values are 
print("len(xs) =", len(xs), "xs[0] =", xs[0], "xs[-1] =", xs[-1])

plt.figure()
plt.plot(xs, val_P8, label= "P8(x)")
plt.plot(xs,val_P9,label="P9(x)")
plt.xlabel("x")
plt.ylabel("P_n(x)")
plt.legend(["SciPy P8(x)", "SciPy P9(x)"])
plt.title("Legrande Functions of 8th and 9th Derivitive")


#%% Question 2 Rodrigues Formula
import math as m 

#function that represent (x^2 -1)^n that is given
#Dont't use f since it has already been used earlier, going to use g(n,x)
def g(n,x):
    return (x**2 -1)**n

#using the function set out above to find the derivitive/
def central_1(g,n,x,h):
    return (g(n , x + h/2) - g(n, x - h/2 ))/h  

def central_k ( g, n, x, h, k): # where k represents the kth derivitive just a symbol 
    if k == 0 : #k=0, return the function value
        return g(n,x)
    return (central_k(g, n, x +h/2, h, k-1) -central_k(g, n, x-h/2, h, k-1))/h

def P_rod(n, xs, h):
    coeff = 1.0/(2**n * m.factorial(n))
    ys = []
#change the variable from x to xi, as x has already been used
    for xi in xs:
        dn = central_k(g, n, xi, h, k=n)
        ys.append(coeff * dn)

    return ys
#function returns the rodrigues formula
    
h= 0.01
P8_rod = P_rod(8, xs, h)
P9_rod = P_rod(9, xs, h)
#compute solution to Rodrigeus at n=8 and n=9 when h= 0.01

#plot the two functions, Legrande and Rodrigeus, 
plt.figure()
plt.plot (xs, val_P8, label="Legrande $P_8(x)$")
plt.plot (xs, val_P9, label="Legrande $P_9(x)$")

plt.plot(xs, P8_rod, "--",label= f"$Rodrigues P_8(x)$" )
plt.plot(xs, P9_rod, "--",label=  f"$Rodrigues P_9(x)$" )

plt.xlabel("x")
plt.ylabel("$P_n(x)$")
plt.title("Legendre Polynomials P8 and P9: SciPy vs Rodrigues (h = 0.01)")
plt.legend(fontsize= 8)
plt.show()




