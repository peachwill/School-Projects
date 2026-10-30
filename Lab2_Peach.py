# -*- Lab2_Peach -*-
"""
Created on Sat Jan 17 12:51:42 2026

@author: willi
"""

#%% Question 1 

import numpy as np 
import matplotlib.pyplot as plt 
import math as m 
import mpmath as mp 

#%%  Question 1a 

#rectangular method 

def rectangular_rule(f,a,b,n):
    if n <2 :
        raise  ValueError ("n must be least 2")
        # use ValueError rather than print(".....) and then return none 
     #if b == a :
      #  return 0.0
    #make n number of spaces evenly spaced between a and b, bulding a grid between a and b of n spaces 
    x = np.linspace(a, b, n)
    #spacing between/ step size 
    h= (b-a)/(n-1)
    #left rule use everything but end points, x-1 make sure you dont go all the way to the end 
    area = h * np.sum(f(x[:-1]))
    return area 

#%% Simpson time, approximate integral using 1/3 rule 
def simpson_rule(f, a, b, n):
    if n < 3 :
        raise ValueError("n must be at least 3") 
    if n % 2 ==0:
        raise ValueError("n must be odd")
  
#make n number of spaced points from a to b 
    x = np.linspace(a, b, n)
    h = (b - a) / ( n - 1)
    #evaulate at all the grid points
    fx = f(x)
    # coeffs/weights at all points for simposons rule, 1 4 2 4 ....1 
    weighted_sum = fx[0] + fx[-1]
    weighted_sum = weighted_sum + 4 * np.sum(fx[1:-1:2])   # odd indices
    weighted_sum = weighted_sum + 2 * np.sum(fx[2:-1:2])   # even indices
    
    area = (h/3) * weighted_sum
    return area
   

    
#%%erf(1) Evaulating the Error in each function 

def rel_error (num, exact):
#where num is numerical answer, exact is the exact number
#compairing the two will give us a percentage 
    return abs((num - exact)/exact) * 100.0
f_erf = lambda x: np.exp(-x**2)
#evaulate error fumction at 1 
z =1.0
exact =m.erf(z)

#approximates the error function using rectuale rule, of erf(1) with 100 grid points
ans_rect_100 = (2/ np.sqrt(np.pi)) * rectangular_rule(f_erf, 0, z, 100)
#increase n by 1 to see how the error changes
ans_rect_101 = (2/ np.sqrt(np.pi)) * rectangular_rule(f_erf, 0, z, 101)

#approximates erf using simpson's 1/3 rule with n = 101 
simp_101 = (2/ np.sqrt(np.pi)) * simpson_rule(f_erf, 0, z, 101)

#error for rectangle using n = 100 , 101 as well as simpson rule at n = 101
err_rect_100 = rel_error(ans_rect_100, exact)
err_rect_101 = rel_error(ans_rect_101, exact)
err_simp_101 = rel_error(simp_101, exact)

print("Exact erf(1) is", exact)
print("Rectanlge rule n = 100:", ans_rect_100 , "\n" , "percent error =", err_rect_100, "\n")
print("Rectangle rule n = 101:", ans_rect_101,"\n","perecent error", err_rect_101, "\n")
print("Simpson n = 101:", simp_101,"\n","percent error", err_simp_101, "\n")
#%% Question 1b 
def erf_num(z, n=101, method="simpson"):
 #for z>= 0 intefrat from 0 to z for z< 0 integrate from z to 0 and apply ultply by neg 1 
    f = lambda x: np.exp(-x**2)
#choose bounds, lower lim is always less than upper lim 
    if z >= 0:
        a, b = 0.0, z
        sign = 1.0
    else:
        a, b = z, 0.0
        sign = -1.0
#compute integral
    if method == "rect":
        integral = rectangular_rule(f, a, b, n)

    else:
        integral = simpson_rule(f, a, b, n)

    return sign * (2 / np.sqrt(np.pi)) * integral



z_val = np.linspace (-4,4,400) #-4,4 evenly with 400 spaced evenly 
erf_simp = np.array ([erf_num(z, n =101, method="simpson")for z in z_val])
erf_exact = np.array ([m.erf(z) for z in z_val])

plt.figure()
plt.plot(z_val, erf_simp, label="Numerical (Simpson, n=101)")
plt.plot(z_val, erf_exact, "--", label="Exact (math.erf)")
plt.xlabel("z")
plt.ylabel("erf(z)")
plt.title("Comparison of Simposon 1/3 method vs Exact")
plt.legend()
plt.show()

#%% Question 2a plot the function 

def func2d(x,y):#define the 2d function so can be called later
    return np.sqrt(x**2 +y)*np.sin(x)*np.cos(y)
#define ranges
x = np.linspace(0, np.pi, 200)
y = np.linspace(0, np.pi/2, 200)

# X = np.meshgrid(x)
# Y = np.meshgrid(y)
X, Y = np.meshgrid(x, y)
#using vectorization
Z = func2d(X,Y)

plt.figure()
plt.contourf(X, Y, Z, levels=40)
plt.colorbar(label="func2d(x, y)")
plt.xlabel("x (rad)")
plt.ylabel("y (rad)")
plt.title("2D Function Contour Plot")
plt.show()

#%% Question 2b 

def simpsonf2d (func2d, a, b , c , d , n , m):

    if n < 3 or m < 3:
        raise ValueError("n and m must be  atleast greater than 3 for Simpson's rule")
    if n % 2 == 0:
        raise ValueError("n must be odd")
    if m % 2 == 0:
        raise ValueError ("m must be odd")
    # build x and y grid in 1D
    xs = np.linspace(a,b,n)
    ys = np.linspace(c,d,m)
    
    hx = (b-a)/(n-1)
    hy = (d-c)/(m-1)   
    
    #create 2D arrays of x and y positions
    X,Y = np.meshgrid(xs,ys)
    
    #evaulate the grid
    F = func2d(X,Y)
    
    #simpson weight       
    wx = np.ones(n)
    wx[1:-1:2] = 4
    wx[2:-1:2] = 2
    
    wy = np.ones(m)
    wy[1:-1:2] = 4
    wy[2:-1:2] = 2
    
    #2d weighted matrix done by combining the two, with shape (m,n)
    W = np.outer(wy,wx)
    
    integral = (hx/3) * (hy/3) * np.sum(W * F) 
    return integral

A_101 = simpsonf2d(func2d, 0, np.pi, 0, np.pi/2, 101, 101)
print("A (n,m) = (101,101):", A_101)
    
A_big = simpsonf2d(func2d, 0, np.pi, 0, np.pi/2, 1001, 2001)
print("A (n.m) =(1001,2001):", A_big)
    
    #checking with quad function, it is a high accuarcy function
f_mp = lambda y, x: mp.sqrt(x**2 +y) * mp.sin(x) * mp.cos(y)
    

A_quad = mp.quad(lambda x: mp.quad(lambda y: f_mp(y, x), [0, mp.pi/2]), [0, mp.pi])
print("A (quad):", A_quad)
    

   