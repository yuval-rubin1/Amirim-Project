import numpy as np
import matplotlib.pyplot as plt

first_t = 0
first_y = 0.8
alpha = 2
#  dy/dt + 2y = t

def eval_y_at_t(t):
    step_size = (t - first_t) / 100
    curr_t = first_t
    curr_y = first_y
    while curr_t < t:
        # curr_y = curr_y + (step_size * (curr_t - (alpha * curr_y)))
        curr_y = curr_y + (step_size * ((curr_t ** 2) - (alpha * curr_y)))
        curr_t += step_size
        
    return curr_y

def draw_and_eval():
    t = 50
    step_size = 0.01
    y_values = [first_y]
    t_values = [first_t]
    curr_t = first_t
    curr_y = first_y
    while curr_t < t:
        # curr_y = curr_y + (step_size * (curr_t - (alpha * curr_y)))
        # curr_y = curr_y + (step_size * ((curr_t ** 2) - (alpha * curr_y)))
        curr_y = curr_y + (step_size * (np.sin(curr_t) - (alpha * curr_y)))
        curr_t += step_size
        y_values.append(curr_y)
        t_values.append(curr_t)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, y_values, 'b-', linewidth=2)
    plt.title('Graph of y = f(t)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid(True)
    plt.savefig('function_graph.png')
    plt.show()

def draw_graph():
    
    # Create an array of t values from 0 to 100 with step size 1
    t_values = np.arange(0, 101, 1)
    
    # Calculate y values for each t using eval_y_at_t
    y_values = [eval_y_at_t(t) for t in t_values]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, y_values, 'b-', linewidth=2)
    plt.title('Graph of y = f(t)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid(True)
    plt.savefig('function_graph.png')
    plt.show()
        



if __name__ == '__main__':
    # draw_graph()
    draw_and_eval()
    
    