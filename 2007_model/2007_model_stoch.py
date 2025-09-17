import numpy as np
import matplotlib.pyplot as plt

from utils import Utils

starting_position = 10
tau_r = 1
tau_s = 0.01
dt = 0.0001
redIO = 1
sim_seconds = 1
num_iterations = sim_seconds * int(1 / dt)
highest_firing_rate = 100  # in Hz
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
spike_cnt_R = np.zeros((len(xi), ))
spike_cnt_L = np.zeros((len(xi), ))
eye_position = np.zeros((1, num_iterations))

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
    
    # in Nadav's paper, we actually don't have a differential equation for r, but only for S. Using his logic:
    rR[:, i] = xi * (eye_position[:, i-1]) + r0
    rL[:, i] = -xi * (eye_position[:, i-1]) + r0
    rR[:, i] = np.maximum(rR[:, i], 0)
    rL[:, i] = np.maximum(rL[:, i], 0)
    
    
    spiking_R[:, i] = np.random.rand(len(xi)) < (dt * rR[:, i])
    spiking_L[:, i] = np.random.rand(len(xi)) < (dt * rL[:, i])
    rR_activated = Utils.traditional_by_spike(rR[:, i], spiking_R[:, i])
    rL_activated = Utils.traditional_by_spike(rL[:, i], spiking_L[:, i])

    SR[:, i] = SR[:, i-1] + (1 / tau_s) * ((dt * -SR[:, i-1]) + rR_activated)
    SL[:, i] = SL[:, i-1] + (1 / tau_s) * ((dt * -SL[:, i-1]) + rL_activated)
    
    eye_position[:, i] = np.dot((SR[:, i] - SL[:, i]), eta)

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
spike_counts_R = np.sum(spiking_R, axis=0)
spike_counts_L = np.sum(spiking_L, axis=0)
plt.plot(time, spike_counts_R, label='Right Side Neurons')
plt.plot(time, spike_counts_L, 'r', label='Left Side Neurons')
plt.xlabel('Time (s)')
plt.ylabel('Number of Spiking Neurons')
plt.title('Number of Spiking Neurons Over Time')
plt.legend()
plt.grid(True)
plt.show()