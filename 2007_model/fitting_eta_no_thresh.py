import numpy as np
from utils import Utils
from sklearn.linear_model import LinearRegression, Ridge
from scipy.optimize import lsq_linear


# Load the CSV data into a numpy array named "slopes_thresh"
slopes_thresh = np.genfromtxt('./2007_model/slopes_thresholds.csv', delimiter=',')

lambda_0 = 100

# activation_function = Utils.ReLU
activation_function = lambda x: max(0, x / (lambda_0 + x))
# activation_function = lambda x: x / (20 + x)
# activation_function = Utils.ReLU


# note that we assume we have eta, vector of N weights, such that:
# sum_i=1^n eta_i * s_infty(rR_i) - sum_i=1^n eta_i * s_infty(rL_i) = eye_position
# and also we have rR_i(t) = k_i * eye_position + r0
# and rL_i(t) = -k_i * eye_position + r0
# so this function creates a sample for a given eye position

def calc_one_sample(eye_pos, k_i, r0_i):
    # activation_function = Utils.traditional
    right_part = activation_function((k_i * eye_pos) + r0_i)
    left_part = activation_function(((-k_i) * eye_pos) + r0_i)
    return right_part - left_part

samples = []
labels = []
for eye_pos in range(-200, 200, 9):
    eye_pos /= 10
    sample = [calc_one_sample(eye_pos, slopes_thresh[i, 0], slopes_thresh[i, 1]) 
              for i in range(slopes_thresh.shape[0])]
    samples.append(sample)
    labels.append(eye_pos)

samples = np.array(samples)
labels = np.array(labels)

model = LinearRegression(fit_intercept=False)
# model = Ridge(alpha=0.1, fit_intercept=False)
model.fit(samples, labels)
np.savetxt(f"./2007_model/traditional_etas_no_thresh_{lambda_0}.csv", model.coef_, delimiter=",")
# result = lsq_linear(samples, labels, bounds=(0.15, 5))

# constrained_coefs = result.x

# eye_position = -15
# one_sample = [calc_one_sample(eye_position, slopes_thresh[i, 0], slopes_thresh[i, 1])
#               for i in range(slopes_thresh.shape[0])]
# one_sample = np.array(one_sample)
# print("position: ", one_sample @ model.coef_)