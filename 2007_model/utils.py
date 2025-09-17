import numpy as np

tau_r = 1
tau_s = 0.01
lambda_0 = 100

class Utils:
    time_step = 0.002
    
    # etas_source_thresh = './2007_model/traditional_etas_with_thresh.csv'
    etas_source = f'./2007_model/traditional_etas_no_thresh_{lambda_0}.csv'
    # old_etas_source = './2007_model/traditional_etas.csv'
    eta = np.loadtxt(etas_source, delimiter=',')
    slopes_thresh = np.genfromtxt('./2007_model/slopes_thresholds.csv', delimiter=',')
    xi = slopes_thresh[:, 0]
    thresholds = slopes_thresh[:, 2]
    r0 = slopes_thresh[:, 1]
    
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
        return x / (lambda_0 + x)
    
    @staticmethod
    def traditional_by_spike(r, spike):
        return spike / (lambda_0 + r)
    
    @staticmethod
    def get_traditional_by_threshold(thresh):
        def res(x):
            if x <= thresh:
                return 0
            return Utils.traditional(x - thresh)
        return res
    
    @staticmethod
    def traditional_spikes(r, spike):
        return spike / (100 + r)
    
    @staticmethod
    def get_traditional_by_threshold_spikes(thresh):
        def res(r, spike):
            r = max(r - thresh, 0)
            return Utils.traditional_spikes(r, spike)
        return res
    
    def random_noise() -> float:
        if np.random.random() < 0.0001:
            return np.random.uniform(-0.005, 0.005)
        return 0.0
