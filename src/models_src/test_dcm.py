import numpy as np
from dcm_model import DCM
from utils import *
import matplotlib.pyplot as plt
import os
import scipy.signal as signal
import jax
import random
def main():
    print(os.getcwd())
    seed = 42  # Choose any integer for the seed
    np.random.seed(seed)  # For NumPy
    random.seed(seed)     # For Python's random module
    jax.random.PRNGKey(seed)  # For JAX (if you use JAX's random functions)
    model = 'dcm'
    model_data = 'dcm'
    # data_file=f'/Users/aliag/Desktop/EEG_Generation/data/synthetic_data/synthetic_data_{model_data}.npy'
    # data_file='/Users/aliag/Desktop/EEG_Generation/data/synthetic_data/synthetic_data_linear.npy'
    data_file='/Users/aliag/Desktop/EEG_Generation/data/synthetic_data/synthetic_data_dcm.npy'
    # data_file='/Users/aliag/Desktop/EEG_Generation/data/synthetic_data/synthetic_data_sigmoid.npy'
    # data_file='/Users/aliag/Desktop/EEG_Generation/data/real_data/Fz_Cz.npy'
    data = np.load(data_file, allow_pickle=True).item()
    stimuli = data['stimuli']
    states = data['states']
    states = np.array(states)
    measurements = data['measurements']
    measurements_noisy = data['measurements_noisy']
    new_states = np.zeros((11,2,3000))
    fs = 300
    lowcut = 1.0  # Low cutoff frequency in Hz
    highcut = 50.0  # High cutoff frequency in Hz

    # Normalizing the frequencies by Nyquist frequency
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(N=4, Wn=[low, high], btype='band')
    
    filtered_signal_0 = signal.filtfilt(b, a, measurements[:,0])
    filtered_signal_1 = signal.filtfilt(b, a, measurements[:,1])
    filtered_signals = np.array([filtered_signal_0[0:5000], filtered_signal_1[0:5000]]).T
    
    # states = states.T
    # for i in range(0,11):
    #     new_states[i,0] = states[0]
    #     new_states[i,1] = states[1]
    # stimuli = np.array([stimuli, stimuli]).T
    aug_state_dim_flattened = 18
    n_params = 23
    covariance_value = 1e-6
    sources = 2
    dt = 1e-1

    params_dict = {
                   'theta': 0.11,
                   'H_e': 0.9,
                   'tau_e': 3.0,
                   'H_i': 25.0,
                   'tau_i': 20.0,
                   'gamma_1': 1.0,
                   'gamma_2': 4/7,
                   'gamma_3': 1/8,
                   'gamma_4': 1/2,
                   'C_f': np.eye(2)*0.5,
                   'C_l': np.eye(2)*0.5,
                   'C_u': np.ones(2)*0.5,
                   'C_b': np.eye(2) * 0.5
                }
    Q_x = np.eye(aug_state_dim_flattened) * 1e-6
    # Q_x[18,18] = 10
    # Q_x[19,19] = 10
    # Q_x[20,20] = 10
    # Q_x[21,21] = 10
    R_y = np.eye(sources) * 1e-4
    P_x_ = np.eye(aug_state_dim_flattened) * 1e-6
    P_x = np.eye(aug_state_dim_flattened) * 1e-6
    P_params_ = np.eye(n_params) * 1e-6
    P_params = np.eye(n_params) * 1e-6
    # P_params_ = np.eye(n_params) * 1e-1
    # P_params = np.eye(n_params) * 1e-1
    # P_params[1:1] = 5
    # P_params[2:2] = 5
    # P_params[3:3] = 5
    # P_params[10:14] = 8
    # P_params[14:18] = 8
    # P_params[18:20] = 7
    # P_params[19,19] = 40
    # P_params[20,20] = 30
    # P_params[21,21] = 40
    # P_params[22,22] = 30
    Q_params = np.eye(n_params)*1e-6
    # Q_params[0,0] = 1e-4
    # Q_params[1,1] = 1e-4
    # Q_params[2,2] = 1e-4
    # Q_params[4,4] = 1e-4
    # Q_params[13,13] = 1e-4
    # Q_params[16,16] = 1e-4
    # Q_params[19,19] = 1e-4
    # Q_params[20,20] = 1e-4
    # Q_params[21,21] = 1e-4
    # Q_params[22,22] = 1e-4
    aug_state_dim = 9
    initial_x = np.zeros((aug_state_dim, sources))
    # initial_H = np.eye(2)

    initial_H = np.array([[0.3, 0.7], [0.9, 0.1]])
    # initial_H = np.array([[0.9, 0.9], [0.4, 0.5]])
    # initial_x[9:11] = initial_H
    # initial_H = np.array([[1, 1], [0.5, 0.4]])
    state_dim = 9
    n_iterations = 1e3
    dcm_model = DCM(state_dim, aug_state_dim, sources, dt, initial_x, initial_H, params_dict, Q_x, R_y, P_x_, P_x, P_params_, P_params, Q_params)
    # new_states = new_states.reshape(3000,11,2)
    states_predicted, measurements_predicted = dcm_model.fit(stimuli, measurements_noisy, states)

    np.save(f"/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/unknown_H/{model}/{model_data}_data/states_predicted.npy", states_predicted)
    np.save(f"/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/unknown_H/{model}/{model_data}_data/measurements_predicted.npy", measurements_predicted)
    # np.save(f"/Users/aliag/Desktop/TFG/Figures/Results/real_data/dcm/states_predicted.npy", states_predicted)
    # np.save(f"/Users/aliag/Desktop/TFG/Figures/Results/real_data/dcm/measurements_predicted.npy", measurements_predicted)
    # H = states_predicted[-1, 9:11, :]
    # y = []
    # for t in range(0,len(states_predicted)):
    #     x0 = states_predicted[t, 2, :] - states_predicted[t, 3, :]
    #     y.append(H @ x0)
    # nsqe_measurements = get_norm_squared_error(measurements, y)

if __name__ == "__main__":
    main()
