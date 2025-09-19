import numpy as np
import pickle

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
    
    @staticmethod
    def get_xi_statistics():
        """
        Calculate mean and standard deviation for xi parameter.
        
        Returns:
            tuple: (mean, std) of xi values
        """
        mean_xi = np.mean(Utils.xi)
        std_xi = np.std(Utils.xi)
        return mean_xi, std_xi
    
    @staticmethod
    def save_params_to_pickle(filename='./2007_model/utils_params.pkl'):
        """
        Save xi, eta, and r0 parameters to a pickle file.
        This creates the file that 2007_for_cluster.py loads.
        
        Args:
            filename (str): Name of the pickle file to save. Defaults to 'utils_params.pkl'.
        """
        params = {
            'xi': Utils.xi,
            'eta': Utils.eta,
            'r0': Utils.r0
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(params, f)
        
        print(f"Parameters saved to {filename}")
        print(f"  - xi shape: {Utils.xi.shape}")
        print(f"  - eta shape: {Utils.eta.shape}")
        print(f"  - r0 shape: {Utils.r0.shape}")


# Generate the pickle file when this module is run directly
if __name__ == "__main__":
    Utils.save_params_to_pickle()
    # print(Utils.get_xi_statistics())
