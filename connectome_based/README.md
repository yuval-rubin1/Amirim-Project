# Overview

This directory contains implementation of connectome-based model related functions. It uses code created by the original authors of the 2024 paper, available [here](https://github.com/goldman-lab/Connectome-Model).

The functions used from the authors code are the following:
1. `simulate`: Simulates the firing rates of the neurons in the network over time, given the connectivity matrix and other parameters.
2. `sorted_eigs`: Computes and sorts the eigenvalues and eigenvectors of the connectivity matrix.
3. `get_scaled_slopes`: Computes the tuning curves of the neurons in the network from the connectivity matrix, enabling calculation of eye position from firing rates.
Note that some modifications were made to the original code to fit the needs of this project.

# Usage

The main file in this directory is `connectomics.py`, which contains the implementation of the connectome-based model functions and the simulation code. It is meant to run on the cluster using MPI for parallel processing.

To run the simulation, use the following command:

```bash
python connectomics.py
```

The script will save the MSD results for each seed in a .npy file named `msds_array.npy`, on which you can then run the analysis using the `analyze_msd.py` script in the main directory (preferably with the `ou` parameter).