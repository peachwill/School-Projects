# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:09:44 2026

@author: willi
"""

#%% Imports and Functions 

import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib import animation

plt.rcParams.update({'font.size': 14})
plt.rcParams['figure.dpi'] = 110

#provided functions
def feval(funcName, *args):
    return eval(funcName)(*args)

# vectorized forward Euler with 1d numpy arrays
def euler(f, y0, t, h): # Vectorized forward Euler (so no need to loop, even for coupled ODEs) 
    k1 = h*f(y0, t)                     
    y1 = y0+k1
    return y1 

def backwardeuler(f,y0,t ,h):
    #general backward euler step:
    #solve this eqn, y1 =y0 + h*f(y1, t+h)
    
    # residual function that should equal zero
    def residual(y1):
        return y1 - y0 -h*f(y1,t+h)
    
    # fsolve finds the root of the residual equation
    y1 = fsolve(residual,y0,maxfev =1000)
    
    return y1 

def rk4(f,y0, t,h):

    # Instead of using only one slope (Euler),
    # RK4 calculates FOUR slope estimates and averages them.

 # k1 = slope at start
 # k2 = slope halfway using k1
 # k3 = slope halfway using k2
 # k4 = slope at end

 #Final update:

 #y1 = y0 + (1/6)*(k1 + 2k2 + 2k3 + k4)
    
 
    k1 = h* f(y0,t)
    k2 = h* f(y0 + 0.5*h*k1, t + 0.5*h)
    k3 = h* f(y0 + 0.5*h*k2, t + 0.5*h)
    k4 = h* f(y0 + h*k3, t + h)
    
    y1 = y0 + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return y1 

def odestepper(odesolver, deriv, y0,t):
    
# steps through the whole time array using the chose solver  
# The solver functions (Euler, RK4, etc.) compute ONE time step.
# odestepper repeatedly applies that step to march forward in time.

# convert intial conditions to numpy array 
    y0 = np.atleast_1d(y0)
#create array to store the solution 
    y = np.zeros((t.size, y0.size))
    y[0,:] = y0
#step size computation
    h = t[1] - t[0]
    y_next = y0
    
    for i in range (1,len(t)):
        y_next = feval(odesolver, deriv, y_next, t[i-1],h)
        y[i,:] = y_next  
    return y 

#%% Question 1 a Exponential Decay 

def decay_deriv(y,t):
    return -10*y 

def decay_exact(t):
    return np.exp(-10*t)

def run_q1a(n):
    #time range
    a = 0
    b = 0.6
    y0 = 1
    #time grid 
    t = np.linspace(a,b,n)
    
#numberical soln 
    y_fe = odestepper('euler', decay_deriv, y0, t)[:,0]
    y_be = odestepper('backwardeuler', decay_deriv, y0, t)[:,0]
    #exact solution 
    y_ex = decay_exact(t)
#plotting
    plt.figure()

    plt.plot(t, y_ex, 'k-', label='Exact')
    plt.plot(t, y_fe, 'gs', label='F-Euler')
    plt.plot(t, y_be, 'bo', label='B-Euler')

    plt.xlabel('t')
    plt.ylabel('y')
    plt.title(f'Q1(a): Decay ODE (n = {n})')

    plt.legend()
    plt.grid()
    plt.show()

# run for the two different grid sizes 
run_q1a(10)
run_q1a(20)
    
#%% Q1(b) simple harmonic oscillator f

def sho_deriv(y,t):
    
    x = y[0]
    v = y[1]
    
    dxdt = v 
    dvdt = -x 
    
    return np.array([dxdt,dvdt])

def sho_exact(t):
    x = np.cos(t)
    v = -np.sin(t)

    return x,v

def run_q1b(dt):
    #oscillator period 
    T0 = 2*np.pi
    
    #run for 10 periods
    tmax = 10*T0
    
    #intial conditions and time grid
    t = np.arange(0,tmax + dt, dt)
    
    y0 = np.array([1.0,0.0])
    
    y_euler = odestepper('euler', sho_deriv, y0, t)
    y_rk4 = odestepper('rk4', sho_deriv, y0, t)

    x_euler = y_euler[:,0]
    v_euler = y_euler[:,1]

    x_rk4 = y_rk4[:,0]
    v_rk4 = y_rk4[:,1]

#exact solns 
    x_exact, v_exact = sho_exact(t)
    
    # phase space
    plt.figure()
    plt.plot(v_rk4, x_rk4, label='RK4')
    plt.plot(v_euler, x_euler, '--', label='Euler')

    plt.xlabel('v')
    plt.ylabel('x')
    plt.title(f'Q1(b): Phase Space (dt = {dt})')

    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

    # x vs time
    plt.figure()
    plt.plot(t/T0, x_rk4, label='RK4')
    plt.plot(t/T0, x_euler, '--', label='Euler')
    plt.plot(t/T0, x_exact, ':', label='Exact')

    plt.xlabel('t/T0')
    plt.ylabel('x')
    plt.title(f'Q1(b): Position vs Time (dt = {dt})')

    plt.legend()
    plt.grid()
    plt.show()
    
    # velocity  vs time
    plt.figure()
    plt.plot(t/T0, v_rk4, label='RK4')
    plt.plot(t/T0, v_euler, '--', label='Euler')
    plt.plot(t/T0, v_exact, ':', label='Exact')
    plt.xlabel('t/T0')
    plt.ylabel('v')
    plt.title(f'Q1(b): Velocity vs Time (dt = {dt})')
    plt.legend()
    plt.show()


run_q1b(0.01)
run_q1b(0.005)
    
#%% Question 2a Lorenz system

def lorenz_deriv(y,t):
    #dx/dt = sigma*(y-x)
    #dy/dt = r*x - y -x*z
    #dz/dt = x*y - b*z
    
    sigma = 10 
    r = 28 
    b = 8/3 
    
    x = y[0]
    yv = y[1]
    z = y[2]
    
    dxdt = sigma*(yv - x)
    dydt = r*x -yv - x*z
    dzdt = x*yv - b*z 
    
    return np.array([dxdt, dydt, dzdt])

def solve_lorenz(dt=0.01):
    
    T0 = 2 * np.pi 
    tmax = 4 * T0
    t = np.arange(0, tmax + dt,dt )
    
    ic1 = np.array([1.0,1.0,1.0])
    ic2 = np.array([1.0,1.0,1.001]) #very small difference in z 
    
    sol1 = odestepper('rk4', lorenz_deriv , ic1, t)
    sol2 = odestepper('rk4', lorenz_deriv,ic2,t )
    
    return t, sol1, sol2

#plot lorenz trajectoreies 
t_lor, sol1, sol2 = solve_lorenz()

fig = plt.figure(figsize=(9, 7))
ax  = fig.add_subplot(projection='3d')
ax.view_init(azim=20, elev=29)
 
ax.plot(sol1[:, 0], sol1[:, 1], sol1[:, 2], '-',  lw=0.6, color='red',  label='y0=[1,1,1]')
ax.plot(sol2[:, 0], sol2[:, 1], sol2[:, 2], '--', lw=0.6, color='blue', label='y0=[1,1,1.001]')
 
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('Q2(a): Lorenz Attractor')
ax.legend()
plt.tight_layout()
plt.show()

#Question 2b Animated Lorenz Attractor 
# Solve over a longer time window so the animation has enough to show
#Create an animation showing the trajectory evolving over time.

dt_anim =0.01
t_anim = np.arange(0,25,dt_anim)
sol_anim = odestepper('rk4', lorenz_deriv, np.array([1.0,1.0,1.0]), t_anim)

fig_a = plt.figure(figsize=(9, 7))
ax_a  = fig_a.add_subplot(projection='3d')
ax_a.view_init(azim=20, elev=29)
ax_a.set_xlim(-25, 25); ax_a.set_ylim(-35, 35); ax_a.set_zlim(0, 55)
ax_a.set_xlabel('x'); ax_a.set_ylabel('y'); ax_a.set_zlabel('z')
ax_a.set_title('Q2(b): Lorenz Attractor Animation')
 
line_anim, = ax_a.plot([], [], [], '-',  lw=0.7, color='royalblue')
dot_anim,  = ax_a.plot([], [], [], 'o',  ms=5,   color='red')   # current position marker

stride = 8 # number of trajectory points added per frame - increase to speed up

def init_anim():
    """Initialize both artists with empty data."""
    line_anim.set_data([], []);      line_anim.set_3d_properties([])
    dot_anim.set_data([], []);       dot_anim.set_3d_properties([])
    return line_anim, dot_anim 

def update_anim(frame):
    """Reveal the trajectory up to the current frame index."""
    end = frame * stride + 1
    xs, ys, zs = sol_anim[:end, 0], sol_anim[:end, 1], sol_anim[:end, 2]
    line_anim.set_data(xs, ys);      line_anim.set_3d_properties(zs)
    dot_anim.set_data([xs[-1]], [ys[-1]]); dot_anim.set_3d_properties([zs[-1]])
    
    return line_anim, dot_anim

anim = animation.FuncAnimation(
    fig_a, update_anim,
    frames=len(t_anim) // stride,
    init_func=init_anim,
    interval=20,    
    blit=False      # blit=True causes issues with 3D axes
)
plt.tight_layout()
plt.show()



