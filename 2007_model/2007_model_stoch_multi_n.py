import numpy as np
import matplotlib.pyplot as plt

from utils import Utils

starting_position = 0
tau_r = 1
tau_s = 0.1
dt = 0.0001
sim_seconds = 1
num_iterations = sim_seconds * int(1 / dt)
repeat_neurons = 1000
M = 25 # Take every Mth spike (spike thinning factor)


xi = Utils.xi
eta = Utils.eta
r0 = Utils.r0

plot_ep = True
num_seeds = 1
eye_position_all = np.zeros((num_seeds, num_iterations))  # Store all trajectories



for seed in range(num_seeds):
    print(f"\rSeed {seed+1}/{num_seeds}", end="", flush=True)
    # ---------- initial values - iteration "0"  ----------
    # setting the vectors
    np.random.seed(seed)
    ones = np.ones((len(xi), ))
    rR = np.zeros((len(xi), num_iterations))
    rL = np.zeros((len(xi), num_iterations))
    SR = np.zeros((len(xi), num_iterations))
    SL = np.zeros((len(xi), num_iterations))
    spiking_R = np.zeros((len(xi), num_iterations))
    spiking_L = np.zeros((len(xi), num_iterations))
    spike_counters_R = np.zeros((len(xi), repeat_neurons))  # Individual spike counters for each repeated neuron
    spike_counters_L = np.zeros((len(xi), repeat_neurons))  # Individual spike counters for each repeated neuron
    eye_position = np.zeros((1, num_iterations))  # Current seed's eye position

    # setting the initial values
    rR[:, 0] = starting_position * xi + r0
    rL[:, 0] = (-starting_position) * xi + r0
    rR[:, 0] = np.maximum(rR[:, 0], 0)
    rL[:, 0] = np.maximum(rL[:, 0], 0)
    rR_activated = Utils.traditional(rR[:, 0])
    rL_activated = Utils.traditional(rL[:, 0])

    # synaptic output from each population
    SR[:, 0] = rR_activated
    SL[:, 0] = rL_activated

    eye_position[:, 0] = np.dot((SR[:, 0] - SL[:, 0]), eta)

    # ---------- run simulation  ----------
    
    for i in range(1, num_iterations):
        if i % 10 == 0:
            print(f"\rSeed {seed+1}/{num_seeds} - Iteration {i}/{num_iterations}", end="", flush=True)
        
        # Implementing equation (23) in Nadav's article
        rR[:, i] = xi * (eye_position[:, i-1]) + r0
        rL[:, i] = -xi * (eye_position[:, i-1]) + r0
        rR[:, i] = np.maximum(rR[:, i], 0)
        rL[:, i] = np.maximum(rL[:, i], 0)
        
        # ----------------------- Spikes calculation with repeated neurons and thinning -----------------------
        # Generate spikes for each repeated neuron
        raw_spikes_R = np.random.rand(len(xi), repeat_neurons) < (dt * rR[:, i] * M)[:, np.newaxis]
        raw_spikes_L = np.random.rand(len(xi), repeat_neurons) < (dt * rL[:, i] * M)[:, np.newaxis]
        
        # calculate spikes with thinning:
        # Update spike counters and determine which spikes to keep
        spike_counters_R += raw_spikes_R
        spike_counters_L += raw_spikes_L
        
        # Keep every Mth spike
        thinned_spikes_R = (spike_counters_R % M == 0) & raw_spikes_R
        thinned_spikes_L = (spike_counters_L % M == 0) & raw_spikes_L
        
        # Reset counters when they reach M
        spike_counters_R[thinned_spikes_R] = 0
        spike_counters_L[thinned_spikes_L] = 0
        
        # Calculate average firing rate across repeated neurons
        spiking_R[:, i] = np.sum(thinned_spikes_R, axis=1) / repeat_neurons
        spiking_L[:, i] = np.sum(thinned_spikes_L, axis=1) / repeat_neurons
        
        # -----------------------
        
        rR_activated = Utils.traditional_by_spike(rR[:, i], spiking_R[:, i])
        rL_activated = Utils.traditional_by_spike(rL[:, i], spiking_L[:, i])

        # Implementing equation (24) in Nadav's article
        SR[:, i] = SR[:, i-1] + (1 / tau_s) * ((dt * -SR[:, i-1]) + rR_activated)
        SL[:, i] = SL[:, i-1] + (1 / tau_s) * ((dt * -SL[:, i-1]) + rL_activated)
        
        # Implementing equation (25) in Nadav's article
        eye_position[:, i] = np.dot((SR[:, i] - SL[:, i]), eta)

    # Store this seed's trajectory in the main array
    eye_position_all[seed, :] = eye_position[0, :]

    if plot_ep:
        # ---------- plot eye position  ----------
        time = np.arange(0, num_iterations) * dt
        plt.figure(figsize=(10, 6))
        plt.plot(time, eye_position[0, :])
        plt.xlabel('Time (s)')
        plt.ylabel('Eye Position (deg)')
        plt.title('Eye Position Over Time')
        plt.grid(True)
        plt.ticklabel_format(style='plain', axis='both')
        plt.show()

        # ---------- plot rR for a specific neuron ----------

        neuron_index = 15  # Choose which neuron to plot (adjust as needed)
        plt.figure(figsize=(10, 6))
        plt.plot(time, rR[neuron_index, :])
        plt.xlabel('Time (s)')
        plt.ylabel(f'rR[{neuron_index}]')
        plt.title(f'Right Side Neuron {neuron_index} Activity Over Time')
        plt.grid(True)
        plt.show()

        # ---------- plot spiking_R for a specific neuron ----------

        plt.figure(figsize=(10, 6))
        spike_counts_R = np.sum(spiking_R, axis=0) * repeat_neurons
        spike_counts_L = np.sum(spiking_L, axis=0) * repeat_neurons
        plt.plot(time[1:], spike_counts_R[1:], label='Right Side Neurons')
        plt.plot(time[1:], spike_counts_L[1:], 'r', label='Left Side Neurons')
        plt.xlabel('Time (s)')
        plt.ylabel('Number of Spiking Neurons\n(Out of {})'.format(len(xi) * repeat_neurons))
        plt.title('Number of Spiking Neurons Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
        

# MSD
def compute_msd(x):
    N = len(x)
    msd = np.zeros(N)
    for dt in range(1, N):  # lag
        diffs = x[dt:] - x[:-dt]
        msd[dt] = np.mean(diffs**2)
    return msd


# Option 2: Average MSD across all seeds (uncomment if preferred)
all_msds = np.array([compute_msd(eye_position_all[seed, :]) for seed in range(num_seeds)])
msd = np.mean(all_msds, axis=0)

lag_time = np.arange(1, len(msd)) * dt  # Convert lag to seconds

# Linear regression on the middle portion of the data
start_idx = int(0.10 * len(lag_time))  # Skip first 10%
end_idx = int(0.8 * len(lag_time))     # Skip last 20%

# Extract the fitting region
lag_fit = lag_time[start_idx:end_idx]
msd_fit = msd[1:][start_idx:end_idx]  # msd[1:] to match lag_time length

# Perform linear regression: MSD = 2*D*t + intercept where D is the diffusion coefficient
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(lag_fit, msd_fit)

# Diffusion coefficient is slope/2 for 1D random walk
diffusion_coefficient = slope / 2

# Generate fitted line for plotting
fitted_line = slope * lag_time + intercept

plt.figure(figsize=(10, 6))
plt.plot(lag_time, msd[1:], 'b-', label='MSD data', alpha=0.7)
plt.plot(lag_time, fitted_line, 'r--', linewidth=2, label=f'Linear fit (D = {diffusion_coefficient:.6f} deg²/s)')
plt.axvline(lag_fit[0], color='gray', linestyle=':', alpha=0.5, label='Fit region')
plt.axvline(lag_fit[-1], color='gray', linestyle=':', alpha=0.5)
plt.xlabel('Lag (s)', fontsize=12)
plt.ylabel('Mean Squared Displacement (deg²)', fontsize=12)
plt.title(f'Mean Squared Displacement vs Lag\nDiffusion Coefficient: {diffusion_coefficient:.6f} deg²/s (R² = {r_value**2:.4f})', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print(f"Linear regression results:")
print(f"Slope: {slope:.6f} deg²/s")
print(f"Intercept: {intercept:.6f} deg²")
print(f"Diffusion coefficient (D): {diffusion_coefficient:.6f} deg²/s")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.2e}")
print(f"Standard error: {std_err:.6f}")


