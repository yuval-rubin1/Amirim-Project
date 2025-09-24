import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import sys


def msd_ou(tau, D, kappa):
    return (2*D/kappa)*(1.0 - np.exp(-kappa*tau))


def fit_ou_msd(lags, msd_vals):
    
    #initial guess

    # Take the first 50% of the data points
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


def plot_ou_fit_comparison(lag_time, avg_msd):
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

def plot_loglog_ou_fit(lag_time, avg_msd, D_fitted, kappa_fitted):
    """
    Create a log-log plot of the average MSD data and the fitted OU model.
    """
    # Generate fitted curve using the fitted parameters
    fitted_curve = msd_ou(lag_time, D_fitted, kappa_fitted)
    
    # Create the log-log plot
    plt.figure(figsize=(10, 6))
    
    # Plot the average MSD data
    plt.loglog(lag_time, avg_msd, 'b-', label='Average MSD data', alpha=0.7, linewidth=1.5)
    
    # Plot the fitted OU model
    plt.loglog(lag_time, fitted_curve, 'r--', linewidth=2, 
               label=f'OU model fit (D = {D_fitted:.6f}, κ = {kappa_fitted:.3f})')
    
    # Add labels and styling
    plt.xlabel('Lag (s)', fontsize=12)
    plt.ylabel('Mean Squared Displacement (deg²)', fontsize=12)
    plt.title(f'Log-Log MSD vs Ornstein-Uhlenbeck Model Fit\n'
              f'D = {D_fitted:.6f} deg²/s, κ = {kappa_fitted:.3f} s⁻¹', fontsize=14)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    
    # Save the plot
    plt.savefig('loglog_ou_msd_fit_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def linear_analysis(lag_time, msd):

    dt = 0.0001
    lag_time = np.arange(1, len(msd)) * dt * 5  # Convert lag to seconds
    # Linear regression on the middle portion of the data
    start_idx = int(0.10 * len(lag_time))  # Skip first 10%
    end_idx = int(0.8 * len(lag_time))     # Skip last 20%

    # Extract the fitting region
    lag_fit = lag_time[start_idx:end_idx]
    msd_fit = msd[1:][start_idx:end_idx]  # msd[1:] to match lag_time length

    # Perform linear regression: MSD = 2*D*t + intercept
    # where D is the diffusion coefficient

    slope, intercept, r_value, p_value, std_err = stats.linregress(lag_fit, msd_fit)

    # Diffusion coefficient is slope/2 for 1D random walk
    diffusion_coefficient = slope / 2

    # Generate fitted line for plotting
    fitted_line = slope * lag_time + intercept

    plt.figure(figsize=(10, 6))
    plt.plot(lag_time, msd[1:], 'b-', label='MSD', alpha=0.7)
    plt.plot(lag_time, fitted_line, 'r--', linewidth=2, label=f'Linear fit (D = {diffusion_coefficient:.6f} deg²/s)')
    plt.axvline(lag_fit[0], color='gray', linestyle=':', alpha=0.5, label='Fit region')
    plt.axvline(lag_fit[-1], color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Lag (s)', fontsize=12)
    plt.ylabel('Mean Squared Displacement (deg²)', fontsize=12)
    plt.title(f'Mean Squared Displacement vs Lag\nDiffusion Coefficient: {diffusion_coefficient:.6f} deg²/s (R² = {r_value**2:.4f})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'2007_msd_linear.png', dpi=300, bbox_inches='tight')

    # Create log-log plot of MSD
    plt.figure(figsize=(10, 6))
    
    # Filter out zero and negative values for log plot
    valid_indices = (lag_time > 0) & (msd[1:] > 0)
    lag_time_valid = lag_time[valid_indices]
    msd_valid = msd[1:][valid_indices]
    
    plt.loglog(lag_time_valid, msd_valid, 'b-', label='MSD', alpha=0.7, linewidth=2)
    
    # Add power law reference lines for comparison
    # For normal diffusion: MSD ~ t^1
    # For subdiffusion: MSD ~ t^α where α < 1
    # For superdiffusion: MSD ~ t^α where α > 1
    
    if len(lag_time_valid) > 0:
        t_ref = lag_time_valid
        # Normal diffusion reference (slope = 1)
        msd_ref_normal = msd_valid[0] * (t_ref / t_ref[0])
        plt.loglog(t_ref, msd_ref_normal, 'k--', alpha=0.5, label='Normal diffusion (slope = 1)')
        
        # Fit power law to the data in log space
        log_lag = np.log10(lag_time_valid)
        log_msd = np.log10(msd_valid)
        
        # Use middle portion for fitting (same as linear fit)
        start_idx_log = int(0.10 * len(log_lag))
        end_idx_log = int(0.8 * len(log_lag))
        
        if end_idx_log > start_idx_log:
            log_slope, log_intercept, log_r_value, log_p_value, log_std_err = stats.linregress(
                log_lag[start_idx_log:end_idx_log], 
                log_msd[start_idx_log:end_idx_log]
            )
            
            # Generate fitted power law line
            msd_powerlaw_fit = (10**log_intercept) * (lag_time_valid**log_slope)
            plt.loglog(lag_time_valid, msd_powerlaw_fit, 'r--', linewidth=2, 
                      label=f'Power law fit: MSD ∝ t^{log_slope:.3f} (R² = {log_r_value**2:.3f})')
    
    plt.xlabel('Lag Time (s)', fontsize=12)
    plt.ylabel('Mean Squared Displacement (deg²)', fontsize=12)
    plt.title(f'Log-Log Plot: Mean Squared Displacement vs Lag Time', fontsize=14)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    plt.savefig(f'2007_msd_loglog.png', dpi=300, bbox_inches='tight')


def main():
    # Parse command line arguments
    if len(sys.argv) != 3:
        print("Usage: python analyze_msd.py <npy_filename> <analysis_type>")
        print("analysis_type: 'ou' for Ornstein-Uhlenbeck analysis, 'linear' for linear analysis")
        sys.exit(1)

    filename = sys.argv[1]
    analysis_type = sys.argv[2]

    if analysis_type not in ['ou', 'linear']:
        print("Error: analysis_type must be either 'ou' or 'linear'")
        sys.exit(1)
    
    # Load the msds array from file
    msds = np.load(filename)
    print(f"Loaded msds array with shape: {msds.shape}")

    # Compute average MSD for each lag
    # Check if msds is already 1D (single simulation) or 2D (multiple simulations)
    if msds.ndim == 1:
        # Already 1D, use as is
        avg_msd = msds
    else:
        # 2D array with shape (num_simulations, time_steps)
        # Average across all simulations for each time point
        avg_msd = np.mean(msds, axis=0)

    # Optional: create lag time array (assuming dt=0.001 as in the original code)
    out_dt = 0.001
    linear_dt = 0.0001
    dt = out_dt if analysis_type == 'ou' else linear_dt
    lag_time = np.arange(1, len(avg_msd) + 1) * dt

    # Run analysis based on the specified type
    if analysis_type == 'ou':
        # Run the OU fitting and plotting
        D_fitted, kappa_fitted, param_errors = plot_ou_fit_comparison(lag_time, avg_msd)
        plot_loglog_ou_fit(lag_time, avg_msd, D_fitted, kappa_fitted)
    elif analysis_type == 'linear':
        linear_analysis(lag_time, avg_msd)
    return


if __name__ == "__main__":
    main()











