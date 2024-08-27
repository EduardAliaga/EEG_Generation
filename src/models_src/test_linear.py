import sys
sys.path.insert(0, '../../')
import numpy as np
from visualize import plot_all
from linear_model import LinearModel
from sigmoid_model import SigmoidModel
from dcm_model import DCM
from utils import *
import jax.numpy as jnp
from jax import grad
from jax import jacobian
import matplotlib.pyplot as plt
import os

def main():
    print(os.getcwd())
    data_file='/Users/aliag/Desktop/EEG_Generation/data/synthetic_data/synthetic_data_linear.npy'
    data = np.load(data_file, allow_pickle=True).item()
    stimuli = data['stimuli']
    states = data['states']
    states = np.array(states)
    measurements = data['measurements']
    measurements_noisy = data['measurements_noisy']
    aug_state_dim_flattened = 2
    n_params = 6
    covariance_value = 1e-4
    sources = 1
    tau = 100.0
    dt = 1e-1
    theta = 0.5
    W = np.zeros((2,2))
    W[0,1] = 1e-1
    W[1,0] = -1e-1
    # M = np.zeros((2,2))
    M = 100.0
    Q_x = np.eye(2) * 1e-6
    R_y = np.eye(2) * 1e-6
    P_x_ = np.eye(2) * 1e-6
    P_x = np.eye(2) * 1e-6
    P_params_ = np.eye(6) * 1e-6
    P_params = np.eye(6) * 1e-6
    # P_params[5,5] = 5e-3
    # P_params[4,4] = 5e-3
    Q_params = np.eye(6) * 1e-6
    W_init = np.array([[0,0.2],[-0.5, 0]])
    M_init = 30.0
    initial_H = np.array([[1, 1], [0.5, 0.4]])
    tau_init = 30.0
    params_dict = {
                    'W': W_init,
                    'M': M_init,
                    'tau' : tau_init
    }
    aug_state_dim = 2
    state_dim = 2
    n_iterations = 3e3
    initial_x = np.zeros((2,1))
    model = LinearModel(state_dim, aug_state_dim, sources, dt, initial_x, initial_H, params_dict, Q_x, R_y, P_x_, P_x, P_params_, P_params, Q_params)
    states_predicted, measurements_predicted = model.fit(stimuli, measurements_noisy)
    plt.figure()
    plt.plot(measurements[:,0])
    plt.plot(measurements_predicted[:,0])
    plt.show()
    print('hello')


if __name__ == "__main__":
    main()