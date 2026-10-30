# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Make plots a bit easier to read
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 11

#%% Q1(a) Monte Carlo computing pi by stone throwing

def compute_pi_stone_throwing(N, r=0.4):
    # Estimate pi by throwing N random 'stones' into a unit square.

    # Generate N random (x, y) points in the square
    x = np.random.uniform(-0.5, 0.5, N)
    y = np.random.uniform(-0.5, 0.5, N)

    # Check which points fall inside the circle: x^2 + y^2 <= r^2
    dist_sq = x**2 + y**2
    inside = dist_sq <= r**2

    # pi ≈ (fraction inside) / r^2
    pi_estimate = np.sum(inside) / N / r**2

    return pi_estimate, x, y, inside

# Run with 1000 stones
N_stones = 1000
r = 0.4
pi_est, x, y, inside = compute_pi_stone_throwing(N_stones, r)

print(f"Q1(a): Estimated pi = {pi_est:.4f}  (exact = {np.pi:.4f})")

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x[inside],  y[inside],  s=5, color='royalblue', label='Inside circle')
ax.scatter(x[~inside], y[~inside], s=5, color='tomato',    label='Outside circle')

# Draw the circle outline
theta = np.linspace(0, 2*np.pi, 300)
ax.plot(r*np.cos(theta), r*np.sin(theta), 'k-', linewidth=1.5)

ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_aspect('equal')
ax.set_title(f'Q1(a): Stone Throwing - estimated pi = {pi_est:.4f}, N = {N_stones}')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc='upper right', markerscale=3)
plt.tight_layout()
plt.savefig('Q1a_pi_stone_throwing.png', dpi=150)
plt.show()


#%% Q1(b) Monte Carlo Integration

def mc_integrate(func, a, b, N):
    x_samples = np.random.uniform(a, b, N)
    f_mean = np.mean(func(x_samples))
    return (b - a) * f_mean

# Set up 500 log-spaced samples
N_values = np.logspace(2, 6, 500).astype(int)

I_exact = np.e - 1

# Compute MC estimate for each N
I_mc = np.array([mc_integrate(np.exp, 0, 1, N) for N in N_values])

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(N_values, I_mc, s=8, color='royalblue', label='Numerical', zorder=3)
ax.axhline(I_exact, color='orange', linewidth=2, label='Exact')
ax.set_xscale('log')
ax.set_xlabel('N Sampling Points', fontsize=13)
ax.set_ylabel(r'$I = \int_0^1 dx\, e^x$', fontsize=13)
ax.set_title('Q1(b): Monte Carlo Integration of $e^x$', fontsize=14)
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('Q1b_mc_integration.png', dpi=150)
plt.show()


#%% Q2 Ising Model

def initialize_chain(N, p_up=0.9):
    # Assign +1 with probability p_up, else -1
    spin = np.where(np.random.rand(N) < p_up, 1, -1)

    # Compute initial energy with periodic BCs
    E = -np.sum(spin * np.roll(spin, -1))

    # Magnetization
    M = np.sum(spin)

    return spin, E, M


def metropolis_step(spin, N, kT):
    # Pick a random spin index
    i = np.random.randint(0, N)

    # Neighbours with periodic boundary conditions
    left  = spin[(i - 1) % N]
    right = spin[(i + 1) % N]
    s_old = spin[i]

    # Energy change if we flip spin i (pre-flip value)
    dE = 2 * s_old * (left + right)

    # Accept or reject the flip
    if dE < 0:
        # Always accept if energy decreases
        spin[i] = -spin[i]
        dM = -2 * s_old
        return spin, dE, dM
    else:
        # Accept with Boltzmann probability
        if np.random.rand() < np.exp(-dE / kT):
            spin[i] = -spin[i]
            dM = -2 * s_old
            return spin, dE, dM
        else:
            return spin, 0, 0   # rejected, no change


def run_ising(N, kT, n_steps, p_up=0.9):
    spin, E, M = initialize_chain(N, p_up)

    snapshots = []
    E_history = []
    M_history = []

    for step in range(n_steps):
        spin, dE, dM = metropolis_step(spin, N, kT)
        E += dE
        M += dM

        # Save a snapshot once every N steps (one "sweep")
        if (step + 1) % N == 0:
            snapshots.append(spin.copy())
            E_history.append(E / N)
            M_history.append(M / N)

    return np.array(snapshots), np.array(E_history), np.array(M_history)


def energy_analytical(N, kT, eps=1.0):
    # <E>/N = -eps * tanh(eps / kT)
    beta = 1.0 / kT
    return -eps * np.tanh(beta * eps)


# Run for three temperatures
N_spins   = 50
n_eq      = 40 * N_spins
p_up_warm = 0.6
p_up_cold = 0.95

temperatures = [0.1, 0.5, 1.0]

print("\nQ2: Running Metropolis algorithm for three temperatures...")

for kT in temperatures:
    E_an = energy_analytical(N_spins, kT)

    # Warm start
    snap_w, E_w, M_w = run_ising(N_spins, kT, n_eq, p_up=p_up_warm)
    # Cold start
    snap_c, E_c, M_c = run_ising(N_spins, kT, n_eq, p_up=p_up_cold)

    n_sweeps  = len(E_w)
    iter_axis = np.arange(n_sweeps)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    fig.suptitle(f'Q2: Spin evolution - kT = {kT}  (N={N_spins})', fontsize=14)

    for col, (snap, E_hist, M_hist, label) in enumerate([
        (snap_w, E_w, M_w, f'Warm start (p_up={p_up_warm})'),
        (snap_c, E_c, M_c, f'Cold start (p_up={p_up_cold})')
    ]):
        # Top row: spin evolution image
        axes[0, col].imshow(snap.T, aspect='auto', cmap='plasma',
                            extent=[0, n_sweeps, 0, N_spins],
                            vmin=-1, vmax=1, origin='lower')
        axes[0, col].set_title(f'Spin evolution - {label}')
        axes[0, col].set_ylabel('N Spins')
        axes[0, col].set_xlabel('Iteration/N')

        # Bottom row: energy history
        axes[1, col].plot(iter_axis, E_hist, color='steelblue', lw=1, label='E')
        axes[1, col].axhline(E_an, color='red', lw=2, linestyle='--',
                             label=r'$\langle E \rangle$ an')
        axes[1, col].set_ylabel(r'Energy/N$\epsilon$')
        axes[1, col].set_xlabel('Iteration/N')
        axes[1, col].legend()

    plt.tight_layout()
    plt.savefig(f'Q2_spin_evolution_kT_{kT}.png', dpi=150)
    plt.show()

    print(f"kT = {kT}: "
          f"<E>/N (analytical) = {E_an:.4f} | "
          f"<E>/N (MC, warm) = {np.mean(E_w[-n_sweeps//2:]):.4f} | "
          f"<E>/N (MC, cold) = {np.mean(E_c[-n_sweeps//2:]):.4f}")


#%% Q3 Equilibrium Energy and Magnetization vs Temperature

print("\nQ3: Computing equilibrium properties vs temperature...")

N_spins = 50
n_eq_3  = 40 * N_spins   # equilibration phase
n_avg_3 = 40 * N_spins   # averaging phase

kT_values = np.linspace(0.1, 6.0, 100)

E_mc_avg = np.zeros(len(kT_values))
M_mc_avg = np.zeros(len(kT_values))

for idx, kT in enumerate(kT_values):

    
    # all magnetization points to be on one side of the plot.
    spin, E, M = initialize_chain(N_spins, p_up=0.5)  # balanced random start

    # Equilibration phase (results discarded)
    for _ in range(n_eq_3):
        spin, dE, dM = metropolis_step(spin, N_spins, kT)
        E += dE

    # Recompute E and M cleanly from the spin array after equilibration
    # so any accumulated floating point drift is removed
    E = -np.sum(spin * np.roll(spin, -1))
    M = np.sum(spin)

    # Averaging phase
    E_sum = 0.0
    M_sum = 0.0
    count = 0
    for _ in range(n_avg_3):
        spin, dE, dM = metropolis_step(spin, N_spins, kT)
        E += dE
        M += dM
        E_sum += E / N_spins   # accumulate per-spin energy
        M_sum += M / N_spins   # accumulate per-spin magnetization
        count += 1

    E_mc_avg[idx] = E_sum / count
    M_mc_avg[idx] = M_sum / count

# Analytical solutions
E_analytical = np.array([energy_analytical(N_spins, kT) for kT in kT_values])

# Plot Energy vs T
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(kT_values, E_mc_avg, s=15, color='royalblue', label='MC Numerical', zorder=3)
ax.plot(kT_values, E_analytical, color='orange', lw=2, label='Analytical')
ax.set_xlabel(r'$kT/\epsilon$', fontsize=13)
ax.set_ylabel(r'$\langle E \rangle / N\epsilon$', fontsize=13)
ax.set_title(f'Q3: Energy vs Temperature  (N={N_spins} spins)', fontsize=14)
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('Q3_energy_vs_T.png', dpi=150)
plt.show()

# Plot Magnetization vs T
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(kT_values, M_mc_avg, s=15, color='royalblue', label='MC Numerical', zorder=3)
ax.axhline(0, color='orange', lw=2, label='Analytical (<M> = 0)')
ax.set_xlabel(r'$kT/\epsilon$', fontsize=13)
ax.set_ylabel(r'$\langle M \rangle / N$', fontsize=13)
ax.set_title(f'Q3: Magnetization vs Temperature  (N={N_spins} spins)', fontsize=14)
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('Q3_magnetization_vs_T.png', dpi=150)
plt.show()

# Print commentary
print("Energy: The MC simulation agrees well with the analytical solution")
print("  -N*eps*tanh(beta*eps). At low kT, energy approaches -N*eps (all spins")
print("  aligned). At high kT the system becomes disordered and E -> 0.")
print("  Scatter increases at intermediate kT where thermal fluctuations are largest.")
print()
print("Magnetization: The MC result fluctuates around 0, matching <M>_an = 0.")
print("  At low kT you may see the MC stuck near +N or -N (all-up or all-down)")
print("  because the chain rarely flips its global alignment in a short run.")
print("  This is a finite-size / finite-time effect: the true average is 0.")
