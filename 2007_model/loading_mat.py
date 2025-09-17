import numpy as np

import scipy.io

mat_file_to_load = './slopes_thresholds.mat'
data = scipy.io.loadmat(mat_file_to_load)
print(data.keys())
a_R = data['a_R']
b_R = data['b_R']
T_R = data['T_R']

# print("Contents of a_R:\n", a_R)
# print("Contents of b_R:\n", b_R)
# print("Contents of T_R\n:", T_R)

combined_array = np.column_stack((a_R.flatten(), b_R.flatten(), T_R.flatten()))
# print("Combined array:\n", combined_array)
# np.savetxt('slopes_thresholds.csv', combined_array, delimiter=',', header='a_R,b_R,T_R', comments='')

import matplotlib.pyplot as plt

# Define an x-range. Using the min and max of T_R (third column) with some margin.
x_min = -20
x_max = 20
x_vals = np.linspace(x_min, x_max, 200)

# Plot each line and scatter the corresponding T_R value at y=0.
for a, b, T in combined_array:
    plt.plot(x_vals, a * x_vals + b, color='blue', alpha=0.5)
    plt.scatter(T, 0, color='red')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Lines Defined by a_R, b_R and T_R')
plt.grid(True)
plt.ylim(bottom=0)
plt.xlim(left=x_min, right=x_max)
for line in plt.gca().get_lines():
    line.set_color("red")
plt.show()
