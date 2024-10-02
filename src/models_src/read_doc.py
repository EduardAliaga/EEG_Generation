import numpy as np
import tqdm
model_data = 'dcm'

data_file='/Users/aliag/Desktop/EEG_Generation/data/synthetic_data/synthetic_data_dcm.npy'

data = np.load(data_file, allow_pickle=True).item()
stimuli = data['stimuli']
states = data['states']
states = np.array(states)
measurements = data['measurements']
measurements_noisy = data['measurements_noisy']

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


H = np.array([[1, 1], [0.5, 0.4]])
J = np.zeros((41,43))
J[0:2] = H
var = 1e-4
covariance_matrix = np.eye(43)*1e-6
z = np.zeros(41)
y_aug = np.zeros(43)
iterations = 100
measurements_predicted = np.zeros(2,3000)
for iteration in tqdm(range(iterations)):
    #expectation step
    y_aug[0:2] = measurements_noisy - H @ measurements_predicted
    covariance_matrix[0:2] = np.eye(2) * var
    conditioned_var = np.linalg.inv(J.T@np.linalg.inv(covariance_matrix)@J)
    z = z + conditioned_var @(J.T@np.linalg.inv(covariance_matrix)@y_aug)

    #maximization step
    P = np.linalg.inv(covariance_matrix) - np.linalg.inv(covariance_matrix) @ J @ conditioned_var @ J.T @ np.linalg.inv(covariance_matrix)
    F = -0.5 *  np.trace(P) + 0.5 * y_aug.T @ P.T @ P @ y_aug
    F2 = -0.5 * np.trace(P@P)
    var = var - np.linalg.inv(F2) @ F