# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 21:06:37 2026

@author: willi
"""
#%% Q1a  bisection

def bisection (f_q1 , x0 , x1 , nmax = 200 , eps = 1.0*(10**-8)) :
    #check to see if the signs are opposite 
    if f_q1(x0)*f_q1(x1) >=0 :
        raise ValueError("f(x0) and f(x1) must have opposite signs")
        
    #loop
    for i in range ( nmax ) :
        #midpoint of the interval
        x2 = ( x0 + x1 ) / 2
        
        #keep the half interval that still brackets the sign change
        #decide which half interval sitll contains teh sign change
        if f_q1( x0 ) * f_q1( x2 ) < 0 :
            x1 = x2
        else :
            x0 = x2
        #convergence check   
        x2new = ( x0 + x1 ) / 2
        xdiff = abs ( x2new - x2 )
        #if approximations get small enought stop
        if abs ( xdiff / x2new ) < eps :
            break
    
    return x2new

def f_q1(x):
    return 1.0 / (x - 3.0)

root = bisection(f_q1, 0.0, 5.0, nmax=200, eps=1e-9)

print("Bisection returned:", root)
print("f(root_est) =", f_q1(root))
#this result is not a true root since f(x) has a vertical asymptote at x=3

#%% Q1b controlling convergence 
import numpy as np 

def f(x):
    #e^(x-sqrtx) -x
    return np.exp(x - np.sqrt(x)) -x 

def df(x):
    #the first derivitive using of f(x)
    #e^(x-sqrtx)-x * (1 - 1/2sqrtx)
    return np.exp(x - np.sqrt(x))*(1-1/(2*np.sqrt(x)))-1

#define the function that is given u(x)
#delfated function 
def u(x,a):
    return f(x)/ (x-a)
    #first derivitve of u(x)
def du(x,a) :
    #using quotinent rule
    # d/dx(f/g) = (f'*g - g*f)/g2
    denominator = (x-a)**2
    numerator = df(x)*(x-a) - f(x)
    return numerator/denominator 

#newtons method applied to the deflated function u(x)
def newton(x0,a, nmax=50, tol=1e-9):
    #first guess
    x=x0 
    
    #newton iteration 
    for n in range(nmax):
        
        #inside in the root must be positive 
        if x <= 0:
            raise ValueError("x is less that zero, sqrt(x) not real")
            
        if abs(x-a) < 1e-12:
            raise ValueError("x is too close to a, u(x) blows up")
           
        #evaluate deflated function 
        ux = u(x,a)
        dux = du(x,a)
        
        #avoid by dividing by small derivitive
        if abs(dux) <1e-14:
            raise ValueError("u'(x) is too small, netwon doesnt work")
        
        #newtwon correction step 
        step = ux/dux
        
        #update soln 
        x_new = x - step 
        
        if abs(x_new - x) < tol:
            return x_new, n+1 
        
        x = x_new #update for the next iteration 
        
        print(n, x, ux, step)
    raise RuntimeError("Newton did no converge with max iterations")
a = 1.0
x0_list = [2.0, 0.5, 4.0, 0.1]

for x0 in x0_list:
    root, iters = newton(x0, a, nmax=50, tol=1e-9)
    print("x0 =", x0, " -> root =", root, " iterations =", iters)            
#%% Q1c  Archimedes principal
#Fb = Fw

#sphere density between (0,1)
rho_sphere = 0.49


# R =1.0 
# rho_w = 1.0 not sure if i need to define all the variables

def f_h(h):
    #rho_water(g)(v(h))= mg 
    #V(h)= 4/3pi(R^3)(rho_sphere)
    #rho_water(g)(V(h)) = mg 
    #rho_water(g)V(h)= 4/3pi(R^3)(rho_sphere) = mg
    
    #sub in sub merged volume 
    #1/3pi(3Rh^2-h^3) = 4/3piR^3rho_sphere
    
    #cancel 1/3pi, R=1 
    #3h^2-h^3 = 4rho_sphere
    eqn = 3*h**2 - h**3 - 4*rho_sphere

    return eqn 
h_root = bisection(f_h, 0.0, 2.0, eps=1e-9)

print ("Equilibrium depth h =", h_root)
#%% Q2a Solving non linear equations 
import math as m 
import numpy as np 

#writing out the differnt system of linear eqns 
def fs1(xs):
    
    #input vectors xs into variables 
    x0, x1 = xs
    
    #define eqn 1 
    f0 = x0**2 - 2*x0 + x1**4 - 2*(x1**2) + x1
    
    #define eqn
    f1 = x0**2 + x0 + 2*(x1**3) - 2*(x1**2) - 1.5*x1 - 0.05
    #return the system of eqns as a vector 
    return np.array([f0, f1], dtype=float)


def fs2(xs):
    #input vector xs as variables 
    x0, x1, x2 = xs 
    
    #define eqn 1
    f0 = 2*x0 - x1*(np.cos(x2)) -3 
    
    #define eqn 2 
    f1 = x0**2 -25*(x1-2)**2 + np.sin(x2) - np.pi/10
    
    #define eqn 3 
    f2 = 7*x0*np.exp(x1) - 17*x2 + 8*np.pi 
    
    #return the system as a vector 
    return np.array([f0,f1,f2], dtype = float)

#%% Q2b 
def jacobian(f,x,h=1e-4):
    x = np.array(x, dtype = float )
    fx = f(x)
    
    #number of rows 
    m = fx.size 
    #number of columns 
    n = x.size 
    
    #create an empty jacobian matrix 
    J= np.zeros((m,n), dtype = float) 
    
    #loop over each variable
    for j in range(n):
        
        #make copies so the original x isnt changed 
        xp = x.copy()
        xm = x.copy()
        #preturb only the jth varible forward and backward 
        xp[j] = xp[j] + h 
        xm[j] = xm[j] - h 
        
        #evaulate preturbed points 
        fp = f(xp)
        fm = f(xm)
        
        #central differnt approximation 
        #fills column j of the jacobian 
        J[:, j] = (fp - fm)/ (2*h)
        
    return J , fx 

J1, f1 = jacobian(fs1, [1.0, 1.0])
print("fs1 f(x) =", f1)
print("fs1 J shape =", J1.shape)
print(J1)

J2, f2 = jacobian(fs2, [1.0, 1.0, 1.0])
print("fs2 f(x) =", f2)
print("fs2 J shape =", J2.shape)
print(J2)
#%%Q2c

def newton_sys(f,x0,tol=1e-8, h= 1e-4, nmax = 50):
    #intial guess to numpy array so can do vector math 
    x = np.array(x0,dtype = float)
    
    for k in range(nmax):
        #create jacobian matrix 
        J, fx = jacobian(f, x, h)
        
        #solve linear system, this gives the newton step
        dx = np.linalg.solve(J,-fx)
        
        #update the estimate
        x_new = x + dx 
        
        if np.linalg.norm(dx, ord=np.inf)< tol:
            return x_new 
        
        #move onto the next iteration 
        x = x_new 
        #stop if the correction step very small 
raise RuntimeError("Newton method did not converge")
    
x_sol1 = newton_sys(fs1, [1.0, 1.0])
print("Solution to system 1:", x_sol1)
print("fs1(x) =", fs1(x_sol1))

x_sol2 = newton_sys(fs2, [1.0, 1.0, 1.0])
print("Solution to system 2:", x_sol2)
print("fs2(x) =", fs2(x_sol2))

        
        
        
                    

    

