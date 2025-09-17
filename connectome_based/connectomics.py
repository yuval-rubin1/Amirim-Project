import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import stats
from mpi4py import MPI

# Load from file
with open('data/lb_data.pkl', 'rb') as f:
    data = pickle.load(f)
    lb_Wnorm = data['lb_Wnorm']
    lb_cdf = data['lb_cdf']

def sorted_eigs(X):
    n = np.shape(X)[0]
    y,v = np.linalg.eig(X)
    vtmp = copy.deepcopy(y)
    for i in range(n):
        for j in range(i+1,n):
            if y[j] > y[i]:
                tmp = y[i]
                y[i] = y[j]
                y[j] = tmp
                vtmp[:] = v[:,i]
                v[:,i] = v[:,j]
                v[:,j] = vtmp[:]
    return y,v

def get_scaled_slopes(W,cdf,sf=-1):
    y,v = sorted_eigs(W) # call funciton
    slopes = np.real(v[:,0])
    if sf < 0:
        sf = 2.574 / np.mean(slopes[cdf.loc['integ']])
    return sf*slopes

def simulate(W_in,ynew=0.99,tau=0.1,customInput=False,v_in=None,x=0):
    W = copy.deepcopy(W_in)
    y,v = sorted_eigs(W)
    ymax = np.max(np.real(y))
    W = ynew*W/ymax
    N = np.shape(W)[0]
    if not customInput:
        v_in = 0.1*abs(np.random.randn(N))
        v_in += np.real(np.sum(v[:,0:1],axis=1)) + np.imag(np.sum(v[:,0:1],axis=1))
        v_in += 1*np.real(np.sum(v[:,1:3],axis=1)) + np.imag(np.sum(v[:,1:3],axis=1))
    v_in = v_in / np.linalg.norm(v_in)
    sim_steps = 110000
    input_filter = 0.001*np.exp(-np.linspace(0,10,101))
    I = 2 * np.ones(sim_steps)
    # I[995:1005] = 1e5
    # # I = np.convolve(I,input_filter)[0:sim_steps]
    r = np.zeros((sim_steps,N))
    r_spike = np.zeros((sim_steps,N))
    dt = 0.001
    for i in range(1,sim_steps):
        if i % 10 == 0:
            print(f"\r{i}/{sim_steps}", end="", flush=True)
        spike = np.where(r_spike[i-1,:] < 0, r_spike[i-1,:] * dt, (np.random.rand(N) < dt*r_spike[i-1,:]))
        r_spike[i,:] = r_spike[i-1,:] + (np.dot(W,spike) + dt*(- r_spike[i-1,:] + I[i-1]*v_in))/tau
        if i < 50000:
            r_spike[i,:] = r[i-1,:] + dt*(np.dot(W,r[i-1,:]) - r[i-1,:] + I[i-1]*v_in)/tau
        r[i,:] = r[i-1,:] + dt*(np.dot(W,r[i-1,:]) - r[i-1,:] + I[i-1]*v_in)/tau
        # r = r * (r > 0)
    starting_position = 50000
    resolution = 1
    return r[starting_position::resolution,:], r_spike[starting_position::resolution,:]

def extract_EP(rates, ks):
    mask = np.abs(ks) > 0.01
    filtered_rates = rates[mask]
    filtered_ks = ks[mask]
    return np.average(filtered_rates / filtered_ks)

def simulate_ep(W, cdf):
    chosen_leading_eigenvalue = 0.9
    intrinsic_tau = 1.0
    rates, rates_spikes = simulate(W,ynew=chosen_leading_eigenvalue,tau=intrinsic_tau)
    ks = get_scaled_slopes(W, cdf)
    ep = np.zeros(rates.shape[0])
    ep_spikes = np.zeros(rates.shape[0])
    for i in range(rates.shape[0]):
        ep[i] = extract_EP(rates[i,:], ks)
        ep_spikes[i] = extract_EP(rates_spikes[i,:], ks)
    return ep, ep_spikes

def compute_msd(x):
    N = len(x)
    msd = np.zeros(N)
    for dt in range(1, N):  # lag
        diffs = x[dt:] - x[:-dt]
        msd[dt] = np.mean(diffs**2)
    return msd

num_seeds = 1000
msds = []

def run_simulation(seed):
    np.random.seed(seed)
    ep, ep_spikes = simulate_ep(lb_Wnorm, lb_cdf)
    msd = compute_msd(ep_spikes)
    return msd

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Distribute seeds across MPI processes
seeds_per_process = num_seeds // size
remainder = num_seeds % size

# Calculate start and end seeds for this process
start_seed = rank * seeds_per_process + min(rank, remainder)
end_seed = start_seed + seeds_per_process + (1 if rank < remainder else 0)

# Run simulations on this process
local_msds = []
for seed in range(start_seed, end_seed):
    print(f"Process {rank}: Running simulation {seed}/{num_seeds}")
    local_msd = run_simulation(seed)
    local_msds.append(local_msd)

# Gather all results to rank 0
all_msds = comm.gather(local_msds, root=0)

# Process results on rank 0
if rank == 0:
    # Flatten the list of lists
    msds = []
    for process_msds in all_msds:
        msds.extend(process_msds)
    
    # Convert list to numpy array and compute average
    msds = np.array(msds)
    
    # Save msds array to file
    np.save('msds_array.npy', msds)
    avg_msd = np.mean(msds, axis=0)
    msd = avg_msd
else:
    msd = None

# Broadcast the result to all processes (if needed for further computation)
msd = comm.bcast(msd, root=0)
dt = 0.001
lag_time = np.arange(1, len(msd)) * dt  # Convert lag to seconds

# Only run analysis and plotting on rank 0
if rank == 0:
    # Linear regression on the middle portion of the data
    start_idx = int(0.02 * len(lag_time))  # Skip first 2%
    end_idx = int(0.4 * len(lag_time))     # Skip last 60%

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
    plt.plot(lag_time, msd[1:], 'b-', label='MSD data', alpha=0.7)
    plt.plot(lag_time, fitted_line, 'r--', linewidth=2, label=f'Linear fit (D = {diffusion_coefficient:.6f} deg²/s)')
    plt.axvline(lag_fit[0], color='gray', linestyle=':', alpha=0.5, label='Fit region')
    plt.axvline(lag_fit[-1], color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Lag (s)', fontsize=12)
    plt.ylabel('Mean Squared Displacement (deg²)', fontsize=12)
    plt.title(f'Mean Squared Displacement vs Lag\nDiffusion Coefficient: {diffusion_coefficient:.6f} deg²/s (R² = {r_value**2:.4f})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('msd_plot.png', dpi=300, bbox_inches='tight')
    
    print(f"\nCompleted {num_seeds} simulations using {size} MPI processes")
    print(f"Diffusion coefficient: {diffusion_coefficient:.6f} deg²/s")
    print(f"R²: {r_value**2:.4f}")

    # plt.figure(1, (10, 6))
    # plt.plot(range(len(ep)), ep, 'o-', color='blue', markersize=4, label='Rate-based')
    # plt.plot(range(len(ep_spikes)), ep_spikes, 'o-', color='red', markersize=4, label='Spike-based')
    # plt.xlabel('Time', fontsize=12)
    # plt.ylabel('Eye Position (deg)', fontsize=12)
    # plt.title('Eye Position Over Time', fontsize=14)
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.savefig('eye_position_plot.png', dpi=300, bbox_inches='tight')