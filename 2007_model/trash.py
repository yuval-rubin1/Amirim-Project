# # ---------- ISI (Inter-Spike Interval) Analysis ----------

# def compute_isi_stats(spike_array, dt):
#     """
#     Compute mean and std of Inter-Spike Intervals for all neurons

#     Args:
#         spike_array: 2D array (neurons x time) where 1 indicates a spike
#         dt: time step size

#     Returns:
#         mean_isis: array of mean ISI for each neuron
#         std_isis: array of std ISI for each neuron
#         all_isis: list of all ISI values across all neurons
#     """
#     mean_isis = []
#     std_isis = []
#     all_isis = []

#     for neuron_idx in range(spike_array.shape[0]):
#         # Find spike times for this neuron
#         spike_times = np.where(spike_array[neuron_idx, :] == 1)[0] * dt
        
#         if len(spike_times) >= 2:
#             # Compute ISIs (differences between consecutive spike times)
#             isis = np.diff(spike_times)
#             mean_isis.append(np.mean(isis))
#             std_isis.append(np.std(isis))
#             all_isis.extend(isis)
#         else:
#             # Not enough spikes to compute ISI
#             mean_isis.append(np.nan)
#             std_isis.append(np.nan)

#     return np.array(mean_isis), np.array(std_isis), np.array(all_isis)

# # Compute ISI statistics for right and left neurons
# mean_isi_R, std_isi_R, all_isis_R = compute_isi_stats(spiking_R, dt)
# mean_isi_L, std_isi_L, all_isis_L = compute_isi_stats(spiking_L, dt)

# print("ISI Analysis Results:")
# print("=" * 50)
# print(f"Right neurons - Overall mean ISI: {np.nanmean(mean_isi_R):.6f} s")
# print(f"Right neurons - Overall std ISI: {np.nanmean(std_isi_R):.6f} s")
# print(f"Right neurons - Population ISI std: {np.std(all_isis_R):.6f} s")
# print(f"Left neurons - Overall mean ISI: {np.nanmean(mean_isi_L):.6f} s")
# print(f"Left neurons - Overall std ISI: {np.nanmean(std_isi_L):.6f} s")
# print(f"Left neurons - Population ISI std: {np.std(all_isis_L):.6f} s")
# print(f"Total spikes (R): {np.sum(spiking_R)}")
# print(f"Total spikes (L): {np.sum(spiking_L)}")

# # ---------- plot spiking information  ----------

# # Create a figure with subplots for spiking analysis
# fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# # Plot 1: Spike raster plot for first 10 neurons
# time = np.arange(0, num_iterations) * dt
# neurons_to_plot = min(10, len(Utils.xi))
# for neuron in range(neurons_to_plot):
#     spike_times = time[spiking_R[neuron, :] == 1]
#     axes[0, 0].scatter(spike_times, [neuron] * len(spike_times), s=1, alpha=0.7)
# axes[0, 0].set_xlabel('Time (s)')
# axes[0, 0].set_ylabel('Neuron Index')
# axes[0, 0].set_title('Spike Raster Plot (Right Side, First 10 Neurons)')
# axes[0, 0].grid(True)

# # Plot 2: Total spike count over time
# spike_count_R = np.sum(spiking_R, axis=0)
# axes[0, 1].plot(time, spike_count_R)
# axes[0, 1].set_xlabel('Time (s)')
# axes[0, 1].set_ylabel('Total Spike Count')
# axes[0, 1].set_title('Total Spike Count Over Time (Right Side)')
# axes[0, 1].grid(True)

# # Plot 3: Firing rate vs r value for a specific neuron
# neuron_idx = len(Utils.xi) // 2  # middle neuron
# window_size = int(0.1 / dt)  # 100ms window
# firing_rate_R = np.convolve(spiking_R[neuron_idx, :], np.ones(window_size)/window_size, mode='same') / dt
# axes[1, 0].scatter(rR[neuron_idx, :], firing_rate_R, alpha=0.5, s=1)
# axes[1, 0].set_xlabel('r Value')
# axes[1, 0].set_ylabel('Firing Rate (Hz)')
# axes[1, 0].set_title(f'Firing Rate vs r Value (Neuron {neuron_idx})')
# axes[1, 0].grid(True)

# # Plot 4: Compare r values and spike probability over time for one neuron
# axes[1, 1].plot(time, rR[neuron_idx, :], label='r value', alpha=0.7)
# axes[1, 1].plot(time, rR[neuron_idx, :] * dt, label='Spike probability', alpha=0.7)
# spike_times = time[spiking_R[neuron_idx, :] == 1]
# axes[1, 1].scatter(spike_times, [0.01] * len(spike_times), s=10, color='red', label='Actual spikes', alpha=0.7)
# axes[1, 1].set_xlabel('Time (s)')
# axes[1, 1].set_ylabel('Value')
# axes[1, 1].set_title(f'r Value, Spike Probability, and Actual Spikes (Neuron {neuron_idx})')
# axes[1, 1].legend()
# axes[1, 1].grid(True)

# plt.tight_layout()
# plt.show()

# # ---------- ISI Distribution Plot ----------

# plt.figure(figsize=(12, 4))

# # Plot ISI histogram for right neurons
# plt.subplot(1, 2, 1)
# if len(all_isis_R) > 0:
#     plt.hist(all_isis_R, bins=50, alpha=0.7, density=True)
#     plt.axvline(np.mean(all_isis_R), color='red', linestyle='--', label=f'Mean: {np.mean(all_isis_R):.6f}s')
#     plt.axvline(np.mean(all_isis_R) + np.std(all_isis_R), color='orange', linestyle='--', label=f'Mean+STD: {np.mean(all_isis_R) + np.std(all_isis_R):.6f}s')
#     plt.axvline(np.mean(all_isis_R) - np.std(all_isis_R), color='orange', linestyle='--', label=f'Mean-STD: {np.mean(all_isis_R) - np.std(all_isis_R):.6f}s')
# plt.xlabel('Inter-Spike Interval (s)')
# plt.ylabel('Density')
# plt.title('ISI Distribution (Right Neurons)')
# plt.legend()
# plt.grid(True)

# # Plot ISI histogram for left neurons
# plt.subplot(1, 2, 2)
# if len(all_isis_L) > 0:
#     plt.hist(all_isis_L, bins=50, alpha=0.7, density=True)
#     plt.axvline(np.mean(all_isis_L), color='red', linestyle='--', label=f'Mean: {np.mean(all_isis_L):.6f}s')
#     plt.axvline(np.mean(all_isis_L) + np.std(all_isis_L), color='orange', linestyle='--', label=f'Mean+STD: {np.mean(all_isis_L) + np.std(all_isis_L):.6f}s')
#     plt.axvline(np.mean(all_isis_L) - np.std(all_isis_L), color='orange', linestyle='--', label=f'Mean-STD: {np.mean(all_isis_L) - np.std(all_isis_L):.6f}s')
# plt.xlabel('Inter-Spike Interval (s)')
# plt.ylabel('Density')
# plt.title('ISI Distribution (Left Neurons)')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()
