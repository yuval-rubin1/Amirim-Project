import numpy as np
import matplotlib.pyplot as plt

from simulating_36_n import Utils

starting_position = 0
tau_r = 1
tau_s = 0.01
dt = 0.001
num_iterations = 1000

# Burst mechanism variables
iterations_since_last_burst = 0
burst_value = 0
burst_duration_steps = int(0.05 / dt)  # 0.05 seconds converted to steps

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



rR = np.zeros((len(Utils.xi), num_iterations))
rL = np.zeros((len(Utils.xi), num_iterations))
SR = np.zeros((1, num_iterations))
SL = np.zeros((1, num_iterations))
eye_position = np.zeros((1, num_iterations))

ones = np.ones((len(Utils.xi), ))

# Set initial values (iteration 0)
rR[:, 0] = starting_position * Utils.xi + Utils.r0
rR[rR[:, 0] < 0, 0] = 0
rL[:, 0] = (-starting_position) * Utils.xi + Utils.r0
rL[rL[:, 0] < 0, 0] = 0


SR[:, 0] = np.dot(Utils.traditional(rR[:, 0]), Utils.eta)
SL[:, 0] = np.dot(Utils.traditional(rL[:, 0]), Utils.eta)

eye_position[:, 0] = (SR[:, 0] - SL[:, 0])

for i in range(1, num_iterations):
    
    # Check if we should start a new burst (similar to simulating_36_n.py)
    if i % int(2.0 / dt) == 0 and i > 0:  # Every 2 seconds (converted to steps)
        burst_val = np.random.choice([-1, 1]) * np.random.uniform(0.16, 0.22)
        start_burst(burst_val)
    
    outside_input = get_outside_input()

    eye_position[:, i] = (SR[:, i] - SL[:, i])
    
    # implementing equation (5) for r_i using euler approximation and using ReLU
    # Added outside_input to the equations
    rR[:, i] = rR[:, i-1] + dt*(-rR[:, i-1] + (SR[:, i-1]-SL[:, i-1])*Utils.xi + Utils.r0 + outside_input)
    rR[rR[:, i] < 0, i] = 0
    rL[:, i] = rL[:, i-1] + dt*(-rL[:, i-1] - (SR[:, i-1]-SL[:, i-1])*Utils.xi + Utils.r0 + outside_input)
    rL[rL[:, i] < 0, i] = 0
    
    # implementing equation (5) for SL, SR using euler approximation
    SR[:, i] = SR[:, i-1] + dt*(-SR[:, i-1] + np.dot(Utils.traditional(rR[:, i]), Utils.eta))
    SL[:, i] = SL[:, i-1] + dt*(-SL[:, i-1] + np.dot(Utils.traditional(rL[:, i]), Utils.eta))
    

    # Plot eye position over time
time = np.arange(0, num_iterations) * dt
plt.figure(figsize=(10, 6))
plt.plot(time, eye_position[0, :])
plt.xlabel('Time (s)')
plt.ylabel('Eye Position')
plt.title('Eye Position Over Time')
plt.grid(True)
plt.show()