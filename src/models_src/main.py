import numpy as np
from generate_data import generate_synthetic_data
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
    data_file='/Users/aliag/Desktop/EEG_Generation/src/models_src/synthetic_data.npy'
    data = np.load(data_file, allow_pickle=True).item()
    stimuli = data['stimuli']
    states = data['states']
    states = np.array(states)
    measurements = data['measurements']
    measurements_noisy = data['measurements_noisy']

    aug_state_dim_flattened = 18
    n_params = 23
    covariance_value = 1e-4
    sources = 2
    dt = 1e-2

    params_dict = {
                    'theta': 0.5,
                    'H_e': 0.4,
                    'tau_e': 15.0,
                    'H_i': 32.0,
                    'tau_i': 16.0,
                    'gamma_1': 1.0,
                    'gamma_2': 4/5,
                    'gamma_3': 1/4,
                    'gamma_4': 1/4,
                    'C_f': np.eye(2),
                    'C_l': np.eye(2), 
                    'C_u': np.ones(2),
                    'C_b': np.array([[7.0, 7.0], [0.2, 0]])
                }

    Q_x = np.eye(aug_state_dim_flattened) * covariance_value
    R_y = np.eye(sources) * covariance_value
    P_x_ = np.eye(aug_state_dim_flattened) * covariance_value
    P_x = np.eye(aug_state_dim_flattened) * covariance_value
    P_params_ = np.eye(n_params) * covariance_value
    P_params = np.eye(n_params) * covariance_value
    # P_params[1:1] = 5
    # P_params[2:2] = 5
    # P_params[3:3] = 5
    # P_params[10:14] = 8
    # P_params[14:18] = 8
    # P_params[18:20] = 7
    P_params[19,19] = 40
    P_params[20,20] = 30
    P_params[21,21] = 40
    P_params[22,22] = 30
    Q_params = np.eye(n_params) * covariance_value
    Q_params[19,19] = 10
    Q_params[20,20] = 10
    Q_params[21,21] = 10
    Q_params[22,22] = 10
    aug_state_dim = 9
    initial_x = np.ones((aug_state_dim, sources))
    initial_H = np.array([[1, 0.7], [0.5, 0.8]])
    state_dim = 9
    n_iterations = 1e3
    model = DCM(state_dim, aug_state_dim, sources, dt, initial_x, initial_H, params_dict, Q_x, R_y, P_x_, P_x, P_params_, P_params, Q_params)
    states_predicted, measurements_predicted = model.fit(stimuli, measurements_noisy)

    # H = states_predicted[-1, 9:11, :]
    # y = []
    # for t in range(0,len(states_predicted)):
    #     x0 = states_predicted[t, 2, :] - states_predicted[t, 3, :]
    #     y.append(H @ x0)
    # nsqe_measurements = get_norm_squared_error(measurements, y)


    #norm_squared_errors =  get_norm_squared_errors(states_predicted, measurements_predicted, states, measurements, model.params_dict, real_params, model.state_dim)

   #save_results(model.params_dict, states_predicted, norm_squared_errors)

    np.save("states_predicted.npy", states_predicted)
    np.save("measurements_predicted.npy", measurements_predicted)
    np.save("params_vec.npy", model.params_vec)
    np.save("H.npy", model.H)

    data_file ='synthetic_data.npy'
    data_file_2 ='states_predicted.npy'
    data_file_3 = 'measurements_predicted.npy' 
    data_file_4 = 'params_vec.npy'
    data_file_5 = 'H.npy'
    #stimuli, states, measurements, measurements_noisy, real_params = load_synthetic_data(data_file, f)
    data = np.load(data_file, allow_pickle=True).item()
    data_2 = np.load(data_file_2, allow_pickle = True)
    data_3 = np.load(data_file_3, allow_pickle = True)
    params_vec = np.load(data_file_4, allow_pickle = True)
    H = np.load(data_file_5, allow_pickle = True)

    print(params_vec)
    print(H)
    stimuli = data['stimuli']
    states = data['states']
    states = np.array(states)
    measurements = data['measurements']
    measurements_noisy = data['measurements_noisy']
    print(data_2.shape)
  
    print(np.linalg.norm(data_3 - measurements_noisy))
    print(np.linalg.norm(data_2[:,0,0] - states[0,0,:]))
    plt.figure()
    plt.plot(data_2[:,10,0], label = "predicted states")
    plt.plot(states[10,0,:], label = "ground_truth")
    plt.legend()
    plt.figure()
    plt.plot(data_3, label = "predicted measurements")
    plt.plot(measurements, label = "ground_truth")
    plt.legend()
    plt.show()

    # for i in range(0,9):
    #     plt.figure()
    #     plt.plot(states[i, 0,:])
    #     plt.plot(states_predicted[:,i, 0])
    # plt.show()
    #plot_all()

if __name__ == "__main__":
    main()
