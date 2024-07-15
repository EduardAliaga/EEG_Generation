import numpy as np
from generate_data import generate_synthetic_data
from parameter_and_state_estimation import estimate_parameters_and_states
from visualize import plot_all

def main():
    # Step 1: Generate synthetic data
    tau = 10.0
    dt = 1e-2
    total_time = 30
    theta = 1.0
    M = 1e-1
    n_stimuli = int(total_time / dt)
    period_square = 0.3
    W = np.zeros((6,6))
    W[0,1] = 1
    W[1,0] = -1
  
    H = np.array([[1, 0.7], [0.5, 0.8]])
    f = 'linear'
    generate_synthetic_data(n_stimuli, total_time, period_square, W, H, tau, dt, theta, M, f)

    # Step 2: Estimate parameters and states
    estimate_parameters_and_states(f)

    # Step 3: Plot results
    plot_all()

if __name__ == "__main__":
    main()
