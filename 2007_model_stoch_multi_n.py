import numpy as np
import matplotlib.pyplot as plt

from simulating_36_n import Utils

starting_position = 0
tau_r = 1
tau_s = 0.1
dt = 0.0001
redIO = 1
sim_seconds = 1
# num_iterations = sim_seconds * int(1 / dt)
num_iterations = 100000
repeat_neurons = 1000
M = 25  # Take every Mth spike (spike thinning factor)
np.random.seed(7)

xi = Utils.xi
eta = Utils.eta
r0 = Utils.r0

# ---------- initial values - iteration "0"  ----------

# settings the vectors
ones = np.ones((len(xi), ))
rR = np.zeros((len(xi), num_iterations))
rL = np.zeros((len(xi), num_iterations))
SR = np.zeros((len(xi), num_iterations))
SL = np.zeros((len(xi), num_iterations))
spiking_R = np.zeros((len(xi), num_iterations))
spiking_L = np.zeros((len(xi), num_iterations))
spike_counters_R = np.zeros((len(xi), repeat_neurons))  # Individual spike counters for each repeated neuron
spike_counters_L = np.zeros((len(xi), repeat_neurons))  # Individual spike counters for each repeated neuron
eye_position = np.zeros((1, num_iterations))

# in Nadav implementation, we don't use spikes for starting values.

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

see_vals = np.zeros((len(xi), num_iterations))  # For debugging: to see rR*dt values

for i in range(1, num_iterations):
    if i % 10 == 0:
        print(f"\r{i}/{num_iterations}", end="", flush=True)
    
    # Implementing equation (23) in Nadav's article
    rR[:, i] = xi * (eye_position[:, i-1]) + r0
    rL[:, i] = -xi * (eye_position[:, i-1]) + r0
    rR[:, i] = np.maximum(rR[:, i], 0)
    rL[:, i] = np.maximum(rL[:, i], 0)
    
    
    # calculate spikes with no thinning 
    # spiking_R[:, i] = np.sum(np.random.rand(len(xi), repeat_neurons) < (dt * rR[:, i])[:, np.newaxis], axis=1) / repeat_neurons
    # spiking_L[:, i] = np.sum(np.random.rand(len(xi), repeat_neurons) < (dt * rL[:, i])[:, np.newaxis], axis=1) / repeat_neurons
    
    # -----------------------
    # calculate spikes with thinning
    # Generate spikes for each repeated neuron
    raw_spikes_R = np.random.rand(len(xi), repeat_neurons) < (dt * rR[:, i] * M)[:, np.newaxis]
    raw_spikes_L = np.random.rand(len(xi), repeat_neurons) < (dt * rL[:, i] * M)[:, np.newaxis]
    
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

# MSD
def compute_msd(x):
    N = len(x)
    msd = np.zeros(N)
    for dt in range(1, N):  # lag
        diffs = x[dt:] - x[:-dt]
        msd[dt] = np.mean(diffs**2)
    return msd

msd = compute_msd(eye_position[0])
lag_time = np.arange(1, len(msd)) * dt  # Convert lag to seconds

# Linear regression on the middle portion of the data
start_idx = int(0.05 * len(lag_time))  # Skip first 5%
end_idx = int(0.8 * len(lag_time))     # Skip last 20%

# Extract the fitting region
lag_fit = lag_time[start_idx:end_idx]
msd_fit = msd[1:][start_idx:end_idx]  # msd[1:] to match lag_time length

# Perform linear regression: MSD = 2*D*t + intercept
# where D is the diffusion coefficient
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

exit()


# ---------- plot eye position  ----------

time = np.arange(0, num_iterations) * dt
plt.figure(figsize=(10, 6))
plt.plot(time, eye_position[0, :])
plt.xlabel('Time (s)')
plt.ylabel('Eye Position')
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


# Analysis for all neurons (R and L)

# Calculate average rR and rL values for all neurons
rR_averages = np.mean(rR, axis=1)
rL_averages = np.mean(rL, axis=1)

# Count total spikes for all neurons
spike_counts_R = np.sum(spiking_R, axis=1)
spike_counts_L = np.sum(spiking_L, axis=1)

# Calculate differences between spike counts and average r values
diff_R = (spike_counts_R / sim_seconds) - rR_averages
diff_L = (spike_counts_L / sim_seconds) - rL_averages

# Calculate average of these differences
avg_diff_R = np.mean(diff_R)
avg_diff_L = np.mean(diff_L)

print("Analysis for all neurons:")
print(f"Average difference (spike count - avg rR) for R neurons: {avg_diff_R:.4f}")
print(f"Average difference (spike count - avg rL) for L neurons: {avg_diff_L:.4f}")

# Additional statistics
print(f"\nAdditional statistics:")
print(f"R neurons - Mean spike count: {np.mean(spike_counts_R):.4f}, Mean avg rR: {np.mean(rR_averages):.4f}")
print(f"L neurons - Mean spike count: {np.mean(spike_counts_L):.4f}, Mean avg rL: {np.mean(rL_averages):.4f}")