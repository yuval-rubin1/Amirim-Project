import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys


# Load the msds array from file
if len(sys.argv) != 2:
    print("Usage: python analyze_msd.py <npy_filename>")
    sys.exit(1)

filename = sys.argv[1]
msds = np.load(filename)
print(f"Loaded msds array with shape: {msds.shape}")

# Compute average MSD for each lag
# msds has shape (num_simulations, time_steps)
# We want to average across all simulations for each time point
avg_msd = np.mean(msds, axis=0)

# Optional: create lag time array (assuming dt=0.001 as in the original code)
dt = 0.001
lag_time = np.arange(1, len(avg_msd) + 1) * dt

def msd_ou(tau, D, kappa):
    return (2*D/kappa)*(1.0 - np.exp(-kappa*tau))

def fit_ou_msd(lags, msd_vals):
    
    #initial guess

    # Take the first 70% of the data points
    n_points = int(0.5 * len(lags))
    time_lags = lags[:n_points]
    msd_vals = msd_vals[:n_points]
    k0 = max(3, int(0.05*len(time_lags)))
    s = np.polyfit(time_lags[:k0], msd_vals[:k0], 1)[0] if len(time_lags) >= k0 else (msd_vals[1]-msd_vals[0])/(time_lags[1]-time_lags[0])
    P = np.median(msd_vals[-k0:])  # plateau proxy
    D0 = max(1e-12, 0.5*s)
    kappa0 = max(1e-12, (2*D0)/max(P, 1e-12))
    p0 = (D0, kappa0)
    sigma = None
    popt, pcov = curve_fit(msd_ou, time_lags, msd_vals, p0=p0, sigma=sigma, absolute_sigma=False, maxfev=20000)
    D_hat, kappa_hat = popt
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else (np.nan, np.nan)

    return D_hat, kappa_hat, perr

def plot_ou_fit_comparison():
    """
    Fit the Ornstein-Uhlenbeck model to the average MSD data and create a comparison plot.
    """
    # Fit the OU model to the data
    D_fitted, kappa_fitted, param_errors = fit_ou_msd(lag_time, avg_msd)
    
    # Generate fitted curve using the fitted parameters
    fitted_curve = msd_ou(lag_time, D_fitted, kappa_fitted)
    
    # Create the plot with similar styling to connectomics.py
    plt.figure(figsize=(10, 6))
    
    # Plot the average MSD data
    plt.plot(lag_time, avg_msd, 'b-', label='Average MSD data', alpha=0.7, linewidth=1.5)
    
    # Plot the fitted OU model
    plt.plot(lag_time, fitted_curve, 'r--', linewidth=2, 
             label=f'OU model fit (D = {D_fitted:.6f}, κ = {kappa_fitted:.3f})')
    
    # Add labels and styling
    plt.xlabel('Lag (s)', fontsize=12)
    plt.ylabel('Mean Squared Displacement (deg²)', fontsize=12)
    plt.title(f'MSD vs Ornstein-Uhlenbeck Model Fit\n'
              f'D = {D_fitted:.6f} deg²/s, κ = {kappa_fitted:.3f} s⁻¹', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.savefig('ou_msd_fit_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print fitting results
    print(f"\nOU Model Fitting Results:")
    print(f"D (diffusion coefficient): {D_fitted:.6f} ± {param_errors[0]:.6f} deg²/s")
    print(f"κ (restoring force): {kappa_fitted:.3f} ± {param_errors[1]:.3f} s⁻¹")
    print(f"Characteristic time (1/κ): {1/kappa_fitted:.3f} s")
    print(f"Plateau value (2D/κ): {2*D_fitted/kappa_fitted:.6f} deg²")
    
    return D_fitted, kappa_fitted, param_errors

# Run the fitting and plotting
D_fitted, kappa_fitted, param_errors = plot_ou_fit_comparison()











