import numpy as np
from dcm_model import DCM
from utils import *
import matplotlib.pyplot as plt
import os
import scipy.signal as signal
import jax
import random
import itertools
import sys
sys.path.insert(0, '../')
# import visualization.predictions as pd

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
    params = data['params']
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
  
    n_params = 6
    covariance_value = 1e-6
    sources = 3
    state_dim = 9
    aug_state_dim_flattened = sources * state_dim
    dt = 1e-1
    noise = 0.2
    params_dict = {
                #    'theta': 0.11,
                #    'H_e': 0.9,
                #    'tau_e': 3.0,
                #    'H_i': 25.0,
                #    'tau_i': 20.0,
                #    'gamma_1': 1.0,
                #    'gamma_2': 4/7,
                #    'gamma_3': 1/8,
                #    'gamma_4': 1/2,
                    # 'C_f' : np.zeros(2),
                    # 'C_l' : np.zeros(2),
                    # 'C_u' : 0,
                    # 'C_b' : 0
                    # 'C_f' : np.array([25+ 25*noise,30+ 25*noise]),
                    # 'C_l' : np.array([2+ 2*noise,10+ 10*noise]),
                    # 'C_u' : 100+100*noise,
                    # 'C_b' : 45 + 45*noise
                    'C_f' : np.zeros(2),
                    'C_l' : np.zeros(2),
                    'C_u' : np.zeros(1),
                    'C_b' : np.zeros(1)
                }
    Q_x = np.eye(aug_state_dim_flattened) * 1e-6

    R_y = np.eye(sources) * 1e-6
    P_x_ = np.eye(aug_state_dim_flattened) * 1e-6
    P_x = np.eye(aug_state_dim_flattened) * 1e-6
    P_params_ = np.eye(n_params) * 1e-6
    P_params = np.eye(n_params) * 1e-6

    Q_params = np.eye(n_params)*1e-6

    aug_state_dim = 9
    initial_x = np.ones((state_dim, sources))
    # initial_H = np.eye(2)

    # initial_H = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    # intitial_H = initial_H + initial_H*0.1
    # initial_H = np.array([[1.1, 0.7, 0.4], [0.6, 0.5, 0.8], [1.4, 0.3, 1.1]])
    # initial_H =np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    # initial_H = np.array([[0.9, 0.9], [0.4, 0.5]])
    # initial_x[9:11] = initial_H
    # initial_H = np.array([[1, 1], [0.5, 0.4]])
    n_iterations = 1e3
    covariance_values = [1e-6, 1e-4, 1e-2]
    # dcm_model = DCM(state_dim, aug_state_dim, sources, dt, initial_x, initial_H, params_dict, Q_x, R_y, P_x_, P_x, P_params_, P_params, Q_params)
    # new_states = new_states.reshape(3000,11,2)
    # states_predicted, measurements_predicted = dcm_model.fit(stimuli, measurements_noisy, states)
    experiment = 0
    deviation = '20%'
    H_state = 'unknown_H'
    for Q_x_value, R_y_value, Q_params_value in itertools.product(covariance_values, repeat=3):
        # initial_H = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        initial_H = np.random.randn(3,3)
        # intitial_H = initial_H + initial_H*0.2
        directory = f'/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/{H_state}/0_start/experiment_{experiment}' 
        # directory = f'/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/{H_state}/{deviation}_deviation/experiment_{experiment}' 
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Initialize the covariance matrices with the current values
        Q_x = np.eye(aug_state_dim_flattened) * Q_x_value
        R_y = np.eye(sources) * R_y_value
        P_x_ = np.eye(aug_state_dim_flattened) * 1e-6
        P_x = np.eye(aug_state_dim_flattened) * 1e-6
        P_params_ = np.eye(n_params) * 1e-6
        P_params = np.eye(n_params) * 1e-6
        Q_params = np.eye(n_params) * Q_params_value
        if experiment == 5:
            print('hello')
        # Initialize your model with the current covariance matrices
        dcm_model = DCM(state_dim, aug_state_dim, sources, dt, initial_x, initial_H, params_dict, Q_x, R_y, P_x_, P_x, P_params_, P_params, Q_params)
        
        # Train the model
        states_predicted, measurements_predicted = dcm_model.fit(stimuli, measurements_noisy, states, experiment)
        
        # Save the results with unique filenames based on the covariance values
        # np.save(os.path.join(directory, f"states_predicted_Qx_{Q_x_value}_Ry_{R_y_value}_Qparams_{Q_params_value}.npy"), states_predicted)
        # np.save(os.path.join(directory, f"measurements_predicted_Qx_{Q_x_value}_Ry_{R_y_value}_Qparams_{Q_params_value}.npy"), measurements_predicted)

        # loss_file = f'/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/{H_state}/0_start/experiment_{experiment}/mse_csv.csv'
        # loss_file = f'/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/{H_state}/{deviation}_deviation/experiment_{experiment}/mse_csv.csv'
        # pd.plot_states_predicted_vs_real(states_predicted, states, directory, model)
        # pd.plot_measurements_predicted_vs_real(measurements_predicted, measurements_noisy, directory, model)
        # pd.plot_losses_from_csv(loss_file, directory, model)
        params_dict = {
                    'C_f' : np.zeros(2),
                    'C_l' : np.zeros(2),
                    'C_u' : np.zeros(1),
                    'C_b' : np.zeros(1)
                }
        initial_x = np.ones((state_dim, sources))
        experiment +=1
    # np.save(f"/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/known_H/states_predicted.npy", states_predicted)
    # np.save(f"/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/known_H//measurements_predicted.npy", measurements_predicted)
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
