import numpy as np
from simulating_36_n import Utils
from sklearn.linear_model import LinearRegression, Ridge
from scipy.optimize import lsq_linear


# Load the CSV data into a numpy array named "slopes_thresh"
slopes_thresh = np.genfromtxt('./slopes_thresholds.csv', delimiter=',')

# activation_function = Utils.ReLU
# activation_function = lambda x: max(0, x / (20 + x))
# activation_function = lambda x: x / (20 + x)
# activation_function = Utils.ReLU

def calc_one_sample(eye_pos, slope, intercept, thresh=0):
    activation_function = Utils.get_traditional_by_threshold(thresh)
    right_part = activation_function((slope * eye_pos) + intercept)
    left_part = activation_function(((-slope) * eye_pos) + intercept)
    return right_part - left_part

samples = []
labels = []
for eye_pos in range(-200, 200, 9):
    eye_pos /= 10
    sample = [calc_one_sample(eye_pos, slopes_thresh[i, 0], slopes_thresh[i, 1], slopes_thresh[i, 2]) 
              for i in range(slopes_thresh.shape[0])]
    samples.append(sample)
    labels.append(eye_pos)

samples = np.array(samples)
labels = np.array(labels)

model = LinearRegression(fit_intercept=False)
# model = Ridge(alpha=0.1, fit_intercept=False)
model.fit(samples, labels)
np.savetxt("traditional_etas_with_thresh.csv", model.coef_, delimiter=",")
# result = lsq_linear(samples, labels, bounds=(0.15, 5))

# constrained_coefs = result.x

# eye_position = -15
# one_sample = [calc_one_sample(eye_position, slopes_thresh[i, 0], slopes_thresh[i, 1])
#               for i in range(slopes_thresh.shape[0])]
# one_sample = np.array(one_sample)
# print("position: ", one_sample @ model.coef_)