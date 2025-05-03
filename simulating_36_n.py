import numpy as np
import matplotlib.pyplot as plt

tau_r = 1
tau_s = 0.01


class Utils:
    time_step = 0.002
    
    etas_source = 'traditional_etas_with_thresh.csv'
    old_etas_source = 'traditional_etas.csv'
    traditional_etas = np.loadtxt(etas_source, delimiter=',')
    slopes_thresh = np.genfromtxt('./slopes_thresholds.csv', delimiter=',')
    ksi = slopes_thresh[:, 0]
    thresholds = slopes_thresh[:, 2]
    start_r = slopes_thresh[:, 1]
    
    @staticmethod    
    def euler_approx(y_t_n, dy_dt_t_n, h=time_step):
        return y_t_n + h * dy_dt_t_n

    @staticmethod
    def ReLU(x):
        return max(0, x)
    
    @staticmethod
    def get_ReLU_by_r(r):
        def ReLU(x):
            return max(0, x - r)
        return ReLU
    
    @staticmethod
    def traditional(x):
        return x / (20 + x)
    
    @staticmethod
    def get_traditional_by_threshold(thresh):
        def res(x):
            if x <= thresh:
                return 0
            return Utils.traditional(x - thresh)
        return res


class Neuron:
    
    def __init__(self, ksi, eta, activation_function, start_r=0):
        self._ksi = ksi
        self._eta = eta
        self._activation_function = activation_function
        self._r_values = [start_r]
    
    def get_start_r(self):
        return self._r_values[0]
    
    def update_r(self, S_R: float, S_L: float, outside_input: float, tonic_input: float) -> float:
        """update the r value of the neuron using the Euler approximation.

        Args:
            S_R (float): the S_R value of the network
            S_L (float): the S_L value of the network
            outside_input (float): the outside input to the neuron

        Returns:
            float: the new r value of the neuron
        """
        r = self._r_values[-1]
        dr_dt = (1 / tau_r) * (-r + (self._ksi * (S_R - S_L)) + (self._ksi * outside_input) + tonic_input) # multiply outside by ksi because of paragraph that talks about it
        new_r = Utils.euler_approx(r, dr_dt)
        self._r_values.append(new_r)
        return new_r
    
    def get_r(self):
        return self._r_values[-1]
    
    def contribute_to_network(self):
        return self._activation_function(self.get_r()) * self._eta


class Network:
    
    def __init__(self, right_neurons, left_neurons):
        self._right_neurons = right_neurons
        self._left_neurons = left_neurons
        # self._right_eta_vector = np.array([neuron._eta for neuron in right_neurons])
        # self._left_eta_vector = np.array([neuron._eta for neuron in left_neurons])
        
        self._S_R_values = [sum([neuron.contribute_to_network() for neuron in self._right_neurons])]
        self._S_L_values = [sum([neuron.contribute_to_network() for neuron in self._left_neurons])]
        
    def add_right_neuron(self, neuron):
        self._right_neurons.append(neuron)
        
    def add_left_neuron(self, neuron):
        self._left_neurons.append(neuron)
        
    def _calc_S_R_inf(self):
        return sum([neuron.contribute_to_network() for neuron in self._right_neurons])
    
    def update_S_R(self):
        curr_S_R = self._S_R_values[-1]
        dS_R_dt = (1 / tau_s) * (-curr_S_R + self._calc_S_R_inf())
        new_S_R = Utils.euler_approx(curr_S_R, dS_R_dt)
        self._S_R_values.append(new_S_R)
        return new_S_R
    
    def _calc_S_L_inf(self):
        return sum([neuron.contribute_to_network() for neuron in self._left_neurons])
    
    def update_S_L(self):
        curr_S_L = self._S_L_values[-1]
        dS_L_dt = (1 / tau_s) * (-curr_S_L + self._calc_S_L_inf())
        new_S_L = Utils.euler_approx(curr_S_L, dS_L_dt)
        self._S_L_values.append(new_S_L)
        return new_S_L

    def get_S_R(self):
        return self._S_R_values[-1]
    
    def get_S_L(self):
        return self._S_L_values[-1]
    
    def update(self, outside_input=0):
        for neuron in self._right_neurons:
            neuron.update_r(self.get_S_R(), self.get_S_L(), outside_input, neuron.get_start_r())
        for neuron in self._left_neurons:
            neuron.update_r(self.get_S_R(), self.get_S_L(), outside_input, neuron.get_start_r())
        self.update_S_R()
        self.update_S_L()
        
    def print_state(self):
        print("Right neurons:")
        for neuron in self._right_neurons:
            print(neuron.get_r())
        print("Left neurons:")
        for neuron in self._left_neurons:
            print(neuron.get_r())
    
    def plot_S_values(self):
        """
        Plots a scatter plot of the S_R and S_L values.
        S_R is on the X-axis and S_L is on the Y-axis.
        Only every 10th sample is plotted, and the color of the points
        transitions gradually from a very dark blue to a brighter blue.
        """
        # Subsample every 10th value from the stored S_R and S_L values.
        sample_S_R = np.array(self._S_R_values)[::10]
        sample_S_L = np.array(self._S_L_values)[::10]
        num_points = len(sample_S_R)
        
        # Create a gradient array for colors that increases with the sample index.
        colors = np.linspace(0, 1, num_points)
        
        plt.figure(figsize=(8, 6))
        # Use a reversed Blues colormap so that the earliest points are dark and later points are bright.
        scatter = plt.scatter(sample_S_R, sample_S_L, c=colors, cmap='Blues_r')
        plt.xlabel("S_R")
        plt.ylabel("S_L")
        plt.title("Scatter plot of S_L vs S_R")
        plt.colorbar(scatter, label="Progression")
        plt.show()

    def plot_S_R_minus_S_L(self):
        """
        Plots the difference between S_R and S_L over time.
        The X-axis represents time, and the Y-axis represents the difference S_R - S_L.
        Only every 10th sample is plotted, and the color of the points
        transitions gradually from a very dark blue to a brighter blue.
        """
        # Create a time axis based on the time step and number of S_R values.
        time = np.arange(len(self._S_R_values)) * Utils.time_step
        # Subsample every 4th value from the stored S_R and S_L values.
        sample_time = time[::4]
        sample_diff = (np.array(self._S_R_values) - np.array(self._S_L_values))[::4]
        
        plt.figure(figsize=(8, 6))
        plt.plot(sample_time, sample_diff, color='blue')
        plt.xlabel("Time")
        plt.ylabel("S_R - S_L")
        plt.title("Difference between S_R and S_L over time")
        plt.show()
    
    def plot_right_neurons_r(self):
        import matplotlib.pyplot as plt
        # Select every fourth right neuron.
        jump = 4
        neurons = self._right_neurons[::jump]
        num_neurons = len(neurons)
        # Generate a distinct color for each neuron using a colormap.
        colors = plt.cm.viridis(np.linspace(0, 1, num_neurons))
        
        plt.figure(figsize=(8, 6))
        for idx, neuron in enumerate(neurons):
            r_values = np.array(neuron._r_values)
            # Create a time axis based on the time step and number of r values.
            time = np.arange(len(r_values)) * Utils.time_step
            # Subsample every 10th value.
            sample_time = time[::10]
            sample_r = r_values[::10]
            plt.scatter(sample_time, sample_r, color=colors[idx], label=f"Neuron {idx*jump}")
        
        plt.xlabel("Time")
        plt.ylabel("r value")
        plt.title("r values of right neurons over time")
        plt.legend()
        plt.show()
    

class Manager:
    
    def __init__(self):
        
        right_neurons = [Neuron(Utils.ksi[i], 
                                Utils.traditional_etas[i], 
                                Utils.get_traditional_by_threshold(Utils.thresholds[i]),
                                Utils.start_r[i])
                         for i in range(len(Utils.traditional_etas))]
        
        left_neurons = [Neuron(-Utils.ksi[i],
                               Utils.traditional_etas[i], 
                               Utils.get_traditional_by_threshold(Utils.thresholds[i]),
                               Utils.start_r[i])
                         for i in range(len(Utils.traditional_etas))]
        
        
        self._network = Network(right_neurons, left_neurons)
        
    iterations_since_last_burst = 0
    burst_value = 0
    
    def start_burst(self, burst_value=0.18):
        self.iterations_since_last_burst = 0
        self.burst_value = burst_value
    
    def get_outside_input(self):
        if self.iterations_since_last_burst > (0.05 / Utils.time_step):
            return 0
        self.iterations_since_last_burst += 1
        return self.burst_value
    
    def run(self, num_steps):
        for t in range(num_steps):
            outside_input = 0
            if t % 300 == 0:
                burst_value = np.random.choice([-1, 1]) * np.random.uniform(0.16, 0.22)
                self.start_burst(burst_value)
            
            outside_input = self.get_outside_input()
            self._network.update(outside_input) 
        self._network.plot_S_R_minus_S_L()



if __name__ == '__main__':
    Manager().run(4000)
        
    
    