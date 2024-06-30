import numpy as np
from generate_data import generate_synthetic_data
from parameter_and_state_estimation import estimate_parameters_and_states
from visualize import plot_all

def main():
    """# Step 1: Generate synthetic data
    tau = 100.
    dt = 1
    theta = 1.0
    M = 100
    n_stimuli = int(1e3)
    period_square = 10
    W = np.zeros((2,2))
    W[0,1] = 1e-1
    W[1,0] = -1e-1
    H = np.array([[1, 0.7], [0.5, 0.8]])
    generate_synthetic_data(n_stimuli, period_square, W, H, tau, dt, theta, M)"""

    # Step 2: Estimate parameters and states
    estimate_parameters_and_states()

    # Step 3: Plot results
    plot_all()

if __name__ == "__main__":
    main()
