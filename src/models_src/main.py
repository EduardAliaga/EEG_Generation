import numpy as np
from generate_data import generate_synthetic_data
from visualize import plot_all
from linear_model import LinearModel
from sigmoid_model import SigmoidModel
from utils import *
import jax.numpy as jnp
from jax import grad
from jax import jacobian
import matplotlib.pyplot as plt

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
    f  = 'linear'

    generate_synthetic_data(n_stimuli, total_time, period_square, W, H, tau, dt, theta, M, f)
    data_file='synthetic_data.npy'
    stimuli, membrane_potentials, measurements, measurements_noisy, real_params = load_synthetic_data(data_file, f)
    model = LinearModel()
    membrane_potentials_predicted, measurements_predicted = model.fit(stimuli, measurements_noisy)

    norm_squared_errors =  get_norm_squared_errors(membrane_potentials_predicted, measurements_predicted, membrane_potentials, measurements, model.params_dict, real_params, model.state_dim)

    save_results(model.params_dict, membrane_potentials_predicted, norm_squared_errors)

    #estimate_parameters_and_states(f)
    plot_all()

if __name__ == "__main__":
    main()
