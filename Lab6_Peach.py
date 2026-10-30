
#%% functions
import numpy as np
import matplotlib.pyplot as plt

def dft(y):
    
    #Compute the Discrete Fourier Transform of a 1D array y
    

    n = len(y)

    # Clean exit if n is not even
    if n % 2 != 0:
        sys.exit("Error: n must be even for this lab.")

    # j = time/sample index, k = frequency index
    j = np.arange(n)
    k = j.reshape((n, 1))  # k is a column vector

    # Build the full DFT matrix M_{k,j} = exp(-2*pi*i*k*j/n)
    M = np.exp(-2j * np.pi * k * j / n)

    # Matrix multiplication performs all sums at once
    yw = M @ y

    return yw


def idft(yw):
    
    #Compute the Inverse Discrete Fourier Transform
    

    n = len(yw)

    # Clean exit if n is not even
    if n % 2 != 0:
        sys.exit("Error: n must be even for this lab.")

    j = np.arange(n)
    k = j.reshape((n, 1))  # column vector

    # Build inverse transform matrix
    M = np.exp(2j * np.pi * k * j / n)

    # Divide by n because IDFT formula includes 1/n
    y = (M @ yw) / n

    return y


def sine_signal(t, a0, w0, a1, w1, a2, w2):
    
    #Returns the sinusoidal signal used in Question 1 and Question 3
    
    return a0*np.sin(w0*t) + a1*np.sin(w1*t) + a2*np.sin(w2*t)


def gaussian_pulse(t, sigma, wp):
    
    #Gaussian pulse used in Question 2
    
    return np.exp(-(t**2)/(sigma**2)) * np.cos(wp*t)


def omega_no_shift(n, dt):
    
    #Frequency array for the raw DFT (no shift)
    
    k = np.arange(n)
    return 2*np.pi*k/(n*dt)


def omega_shifted(n, dt):
    
    #Frequency array with fftshift applied
    
    return 2*np.pi*np.fft.fftshift(np.fft.fftfreq(n, d=dt))


#%% Question 1 a
import numpy as np
import matplotlib.pyplot as plt


# parameters given in lab
a0, w0 = 3.0, 1.0
a1, w1 = 1.0, 4.0
a2, w2 = 0.5, 7.0

# create subplot grid
fig, ax = plt.subplots(2,2, figsize=(11,7))

for row, n in enumerate([30,60]):

    #time array on interval [0,2pi]
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    dt = t[1] - t[0]

    #build signal
    y = sine_signal(t, a0, w0, a1, w1, a2, w2)

    #compute dft
    yw = dft(y)

    #Frequency array for the raw, unshifted DFT
    w = omega_no_shift(n,dt)

    # time domain plot
    ax[row,0].plot(t,y)

    ax[row,0].set_xlabel("t", fontsize=14)
    ax[row,0].set_ylabel("y", fontsize=14)
    ax[row,0].set_title(f"Time signal n={n}", fontsize=16)

    ax[row,0].tick_params(axis='both', labelsize=12)
    # frequency domain plot
    ax[row,1].stem(w, np.abs(yw), linefmt='b-', markerfmt=' ', basefmt='-b')

    ax[row,1].set_xlabel("omega", fontsize=14)
    ax[row,1].set_ylabel("|y_tilde|", fontsize=14)
    ax[row,1].set_title(f"DFT magnitude n={n}", fontsize=16)

    ax[row,1].tick_params(axis='both', labelsize=12)


plt.show()


#%% Question 1b  (IDFT reconstruction)
import numpy as np
import matplotlib.pyplot as plt
import sys

n = 60
t = np.linspace(0,2*np.pi,n,endpoint=False)

y = sine_signal(t,a0,w0,a1,w1,a2,w2)

# compute DFT
yw = dft(y)

# reconstruct signal using IDFT
y_recovered = np.real(idft(yw))

plt.figure()

plt.plot(t,y,"r",label="original")
plt.plot(t,y_recovered,"b--",label="reconstructed")

plt.xlabel("t", fontsize=14)
plt.ylabel("y", fontsize=14)

plt.legend(fontsize=12)
plt.tick_params(axis='both', labelsize=12)

plt.show()


#%% Q2 Gaussian Pulse Shifted Spectrum
import numpy as np
import matplotlib.pyplot as plt

sigma = 0.5
wp = 0
n = 60 

# time range t = [-pi, pi]
t = np.linspace(-np.pi, np.pi, n, endpoint=False)
dt = t[1]-t[0]

# build the pulse
y = gaussian_pulse(t,sigma,wp)

# compute DFT
yw = dft(y)

# frequency arrays
w_noshift = omega_no_shift(n,dt)

w_shift = omega_shifted(n,dt)
yw_shift = np.fft.fftshift(yw)

fig, ax = plt.subplots(1,2, figsize=(11,4))

# time plot
ax[0].plot(t,y)

ax[0].set_xlabel("t", fontsize=14)
ax[0].set_ylabel("y(t)", fontsize=14)
ax[0].set_title("Gaussian Pulse", fontsize=16)

ax[0].tick_params(axis='both', labelsize=12)

# frequency plot
ax[1].plot(w_noshift,np.abs(yw),label="no shift")
ax[1].plot(w_shift,np.abs(yw_shift),'--',label="shift")

ax[1].set_xlabel("omega", fontsize=14)
ax[1].set_ylabel("|y_tilde|", fontsize=14)

ax[1].legend(fontsize=12)
ax[1].tick_params(axis='both', labelsize=12)
plt.show()


#%% Question 3 Filter noisy data
import numpy as np
import matplotlib.pyplot as plt


n = 200
t = np.linspace(0,8*np.pi,n,endpoint=False)
dt = t[1]-t[0]

# noisy signal
y_noisy  = sine_signal(t, 3.0,1.0, 1.0, 10.0, 0.0, 0.0)

# DFT of signal
yw_noisy= dft(y_noisy)

# shifted spectrum
w_shift = omega_shifted(n,dt)
yw_shift = np.fft.fftshift(yw_noisy)

# copy spectrum for filtering
yw_filtered_shift = yw_shift.copy()

# remove frequencies around ±10
tol = 0.6
mask = np.abs(np.abs(w_shift) - 10) < tol

yw_filtered_shift[mask] = 0

# shift back
yw_filtered = np.fft.ifftshift(yw_filtered_shift)

# recover filtered signal
y_filtered = np.real(idft(yw_filtered))

fig, ax = plt.subplots(1,2, figsize=(11,4))

# time 
ax[0].plot(t,y_noisy,"r--",label="unfiltered")
ax[0].plot(t,y_filtered,"b",label="filtered")

ax[0].set_xlabel("t", fontsize=14)
ax[0].set_ylabel("y", fontsize=14)

ax[0].legend(fontsize=12)
ax[0].tick_params(axis='both', labelsize=12)

# frequency 
ax[1].plot(w_shift,np.abs(yw_shift),"r--",label="unfiltered")
ax[1].plot(w_shift,np.abs(yw_filtered_shift),"b",label="filtered")

ax[1].set_xlabel("omega", fontsize=14)
ax[1].set_ylabel("|y_tilde|", fontsize=14)

ax[1].legend(fontsize=12)
ax[1].tick_params(axis='both', labelsize=12)


plt.show()
