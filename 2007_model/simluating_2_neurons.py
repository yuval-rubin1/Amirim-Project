import matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools
import numpy as np

# Global simulation state.
s_1_values = [4]  # Initial value for s₁ at t = 0
s_2_values = [9]  # Initial value for s₂ at t = 0
time_steps  = [0]   # Starting time step
last_updated_frame = 0  # To ensure state is updated only once per frame

# Simulation parameter.
step_size = 0.01
tau = 20
b = 1

def eval_val(my_val, other_val, noise_mean=0):
    """Update function for the simulation state."""
    noise = np.random.normal(noise_mean, 0.001)  # Mean provided, std 0.01 noise
    b_noise = np.random.normal(0, 0.001)  # Mean 0, std 0.01 noise
    return my_val + ((1 / tau) * step_size * (b + b_noise - my_val - other_val)) + noise

K = s_2_values[0] - s_1_values[0]


def accurate_sol_s_1(t):
    # return b_minus_2K + (s_1_values[0] - b_minus_2K) * np.exp((-t * 4) / tau)
    # b_noise = 0
    # if t % 5 == 0:
    #     b_noise = np.random.normal(0, 0.5)  # Mean 0, std 0.01 noise
    
    half_b_minus_K = (b - K) / 2
    return half_b_minus_K + (s_1_values[0] - half_b_minus_K) * np.exp((-t * 2) / tau)

def accurate_sol_s_2(t):
    
    return accurate_sol_s_1(t) + K

def update_state(frame):
    """
    Update the global simulation state only if this frame is new.
    This prevents multiple updates if both animations are running.
    """
    global last_updated_frame
    if frame > last_updated_frame:
        # Compute new simulation values.
        # new_s_1 = eval_val(s_1_values[-1], s_2_values[-1], noise_mean=0.01)
        # new_s_2 = eval_val(s_2_values[-1], s_1_values[-1])
        new_s_1 = accurate_sol_s_1(frame)
        new_s_2 = accurate_sol_s_2(frame)
        # Append the new values.
        s_1_values.append(new_s_1)
        s_2_values.append(new_s_2)
        time_steps.append(frame)
        last_updated_frame = frame

# --------------------------------------------------
# Function to run time-series animation (s₁ and s₂ vs. time)
def run_time_series_animation():
    fig, ax = plt.subplots()
    # Two line objects: blue for s₁ and green for s₂.
    line1, = ax.plot(time_steps, s_1_values, 'bo-', label="s₁")
    line2, = ax.plot(time_steps, s_2_values, 'go-', label="s₂")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.set_title("Evolution of s₁ and s₂ over time")
    ax.legend()
    ax.grid(True)
    
    def update_time_series(frame):
        # Update simulation state (only once per new frame).
        update_state(frame)
        # Update the data for the time-series plot.
        line1.set_data(time_steps, s_1_values)
        line2.set_data(time_steps, s_2_values)
        ax.relim()
        ax.autoscale_view()
        return line1, line2

    # Create an animation that updates every 50 milliseconds.
    ani = animation.FuncAnimation(fig, update_time_series, frames=itertools.count(1),
                                  interval=50, blit=True)
    plt.show()

# --------------------------------------------------
# Function to run scatter plot animation (s₁ vs. s₂)
def run_scatter_animation():
    fig, ax = plt.subplots()
    # Updated axis labels: s₁ on x-axis, s₂ on y-axis.
    ax.set_xlabel("s₁")
    ax.set_ylabel("s₂")
    ax.set_title("Scatter Plot: s₁ (x-axis) vs s₂ (y-axis)")
    
    def update_scatter(frame):
        # Update simulation state (only once per new frame).
        update_state(frame)
        # Clear the axes to redraw the scatter plot.
        ax.cla()
        ax.set_xlabel("s₁")
        ax.set_ylabel("s₂")
        ax.set_title("Scatter Plot: s₁ (x-axis) vs s₂ (y-axis)")
        
        # Create a color list: all points green except the last one red.
        colors = ['green'] * (len(s_1_values) - 1) + ['red']
        # Switch the axis: now s₁ is x and s₂ is y.
        ax.scatter(s_1_values, s_2_values, c=colors)
        return ax.collections

    # Create an animation that updates every 500 milliseconds.
    ani = animation.FuncAnimation(fig, update_scatter, frames=itertools.count(1),
                                  interval=250, blit=True)
    plt.show()

# --------------------------------------------------
# To run one or both animations, uncomment the corresponding function calls:

# run_time_series_animation()  # Time series of s₁ and s₂ vs. time.
run_scatter_animation()      # Scatter plot of s₁ (x-axis) vs. s₂ (y-axis).
