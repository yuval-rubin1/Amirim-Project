import numpy as np
import matplotlib.pyplot as plt

from simulating_36_n import Utils

starting_position = 5
tau_r = 0.1
tau_s = 0.001
dt = 0.0001
num_iterations = 10000
np.random.seed(7)

# ---------- saccade mechanism  ----------

iterations_since_last_burst = 0
burst_value = 0
burst_duration_steps = int(0.005 / dt)  # 0.05 seconds converted to steps

def start_burst(burst_val=0.18):
    global iterations_since_last_burst, burst_value
    iterations_since_last_burst = 0
    burst_value = burst_val

def get_outside_input():
    global iterations_since_last_burst
    if iterations_since_last_burst > burst_duration_steps:
        return 0
    iterations_since_last_burst += 1
    return burst_value

# ---------- initial values - iteration "0"  ----------

# settings the vectors
ones = np.ones((len(Utils.xi), ))
rR = np.zeros((len(Utils.xi), num_iterations))
rL = np.zeros((len(Utils.xi), num_iterations))
SR = np.zeros((1, num_iterations))
SL = np.zeros((1, num_iterations))
eye_position = np.zeros((1, num_iterations))

# setting the initial values
rR[:, 0] = starting_position * Utils.xi + Utils.r0
rL[:, 0] = (-starting_position) * Utils.xi + Utils.r0
rR[:, 0] = np.maximum(rR[:, 0], 0)
rL[:, 0] = np.maximum(rL[:, 0], 0)
rR_activated = Utils.traditional(rR[:, 0])
rL_activated = Utils.traditional(rL[:, 0])

SR[:, 0] = np.dot(rR_activated, Utils.eta)
SL[:, 0] = np.dot(rL_activated, Utils.eta)

eye_position[:, 0] = (SR[:, 0] - SL[:, 0])

# ---------- run simulation  ----------

for i in range(1, num_iterations):

    if i % int(0.5 / dt) == 0:  # Every 5 seconds (converted to steps)
        burst_val = np.random.choice([-1, 1]) * np.random.uniform(0.16, 0.22)
        start_burst(burst_val)
    
    do_burst = False
    # representing B(t) in equation (5)
    outside_input = get_outside_input() if do_burst else 0
    
    # implementing equation (5) for r_i using euler approximation, when T_i is r0.
    # rR[:, i] = rR[:, i-1] + dt*(1 / tau_r)*(-rR[:, i-1] + (SR[:, i-1]-SL[:, i-1]+ outside_input)*Utils.xi + Utils.r0)
    # rL[:, i] = rL[:, i-1] + dt*(1 / tau_r)*(-rL[:, i-1] - (SR[:, i-1]-SL[:, i-1] + outside_input)*Utils.xi + Utils.r0)
    
    # in Nadav implementation, we actually don't have a differential equation for r, but only for S. Using his logic:
    rR[:, i] = Utils.xi * (SR[:, i-1] - SL[:, i-1]) + Utils.r0
    rL[:, i] = -Utils.xi * (SR[:, i-1] - SL[:, i-1]) + Utils.r0
    rR[:, i] = np.maximum(rR[:, i], 0)
    rL[:, i] = np.maximum(rL[:, i], 0)
    rR_activated = Utils.traditional(rR[:, i])
    rL_activated = Utils.traditional(rL[:, i])
    
    
    # implementing equation (5) for SL, SR using euler approximation, with activated r values
    SR[:, i] = SR[:, i-1] + dt*(1 / tau_s)*(-SR[:, i-1] + np.dot(rR_activated, Utils.eta))
    SL[:, i] = SL[:, i-1] + dt*(1 / tau_s)*(-SL[:, i-1] + np.dot(rL_activated, Utils.eta))
    
    eye_position[:, i] = (SR[:, i] - SL[:, i])

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