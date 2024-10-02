import numpy as np
from utils import *
import jax
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import pandas as pd
import os

def compute_measurements_mse(measurements_predicted, measurements_real):
    training_mse = np.mean((measurements_predicted[0:2500] - measurements_real[0:2500]) ** 2)
    testing_mse = np.mean((measurements_predicted[2500:] - measurements_real[2500:]) ** 2)
    return training_mse, testing_mse

def compute_states_mse(states_predicted, states_real):
        training_mse_states = np.mean((states_predicted[0:2500,:,:] - states_real[0:9,:,:2500].reshape(2500,9,2)) ** 2)
        testing_mse_states = np.mean((states_predicted[2500:,:,:] - states_real[0:9,:,2500:].reshape(500,9,2)) ** 2)
        training_mse_state_x0 = np.mean((states_predicted[0:2500,0,:] - states_real[0,:,:2500].T) ** 2)
        testing_mse_state_x0 = np.mean((states_predicted[2500:,0,:] - states_real[0,:,2500:].T) ** 2)
        return training_mse_state_x0, testing_mse_state_x0, training_mse_states, testing_mse_states
    
def mse_to_csv(training_mse_state_x0, testing_mse_state_x0, training_mse_states, testing_mse_states,training_mse_measurements, testing_mse_measurements, save_path):
        # Create a dictionary for the data with iterations as the first column
    iterations = np.arange(1, len(training_mse_measurements) + 1)
    data = {
        'iterations': iterations,
        'training_mse_state_x0': training_mse_state_x0,
        'testing_mse_state_x0': testing_mse_state_x0,
        'training_mse_states': training_mse_states,
        'testing_mse_states': testing_mse_states,
        'training_mse_measurements': training_mse_measurements,
        'testing_mse_measurements': testing_mse_measurements
    }
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
     
    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(save_path, 'mse_csv.csv'), index=False)
    
    print(f"MSE results saved to {save_path}")



"""def measurement_function(x):
    return x[2:6].reshape(2,2) @ x[0:2]"""
class NeuralModel:
    def __init__(self, state_dim, aug_state_dim, sources, dt, initial_x, initial_H, params_dict, Q_x, R_y, P_x_, P_x, P_params_, P_params, Q_params):

        self.dt = dt
        self.state_dim = state_dim
        self.aug_state_dim = aug_state_dim
        self.sources = sources
        self.aug_state_dim_flattened = self.aug_state_dim * self.sources
        self.x = initial_x
        self.H = initial_H
        self.params_dict = params_dict
        initial_params_dict = params_dict
        self.params_vec = params_dict_to_vector(params_dict)
        self.initial_params_vec = params_dict_to_vector(params_dict)
        self.n_params = len(P_params)
        self.Q_x = Q_x 
        self.R_y = R_y
        self.P_x_ = P_x_
        self.P_x = P_x
        self.initial_P_x = P_x
        self.initial_P_x_ = P_x_
        self.initial_Q_x = Q_x
        self.initial_P_params = P_params
        self.initial_P_params_ = P_params_
        self.initial_Q_params = Q_params
        self.P_params_ = P_params_
        self.P_params = P_params
        self.Q_params = Q_params
        self.model = 'dcm'
        self.model_data = 'dcm'

    def fit(self, stimuli, measurements_noisy, states): 
        num_time_points = len(stimuli)
        total_measurements_predicted = []
        total_states_predicted = []
        measurements_test_mse = np.zeros(100)
        measurements_train_mse = np.zeros(100)
        states_train_mse = np.zeros(100)
        states_test_mse = np.zeros(100)
        state_0_train_mse = np.zeros(100)
        state_0_test_mse = np.zeros(100)
        params_dict_list = []
        H_list = []
        for i in tqdm(range(0,100)):
            print(i)
            states_predicted = np.zeros((num_time_points, self.aug_state_dim, self.sources))
            measurements_predicted = np.zeros((num_time_points, 2))
            # Set initial state
            states_predicted[0] = self.x
            for t in tqdm(range(num_time_points)):
                # self.params_vec[1] = 0
                # self.params_vec[2] = 0
                # self.params_vec[5] = 0
                # self.params_vec[6] = 0
                # self.params_vec[11] = 0
                # self.params_vec[12] = 0
                # if t ==930:
                #     print('hello')
                # self.params_vec[0] = 0
                # self.params_vec[3] = 0
                F_x = self.jacobian_f_o_x(stimuli[t-1])
                F_params = self.jacobian_f_o(stimuli[t-1])
                y = measurements_noisy[t-1]
                # Measurement prediction: h(x) = H @ x (your existing measurement function)
                y_hat = self.H @ self.x[0]
                # y_hat = self.H @ self.x
                if t>=int(num_time_points*0.8):
                        #y_hat = self.H @ self.x
                        # Update state prediction without measurements (i.e., prediction step only)
                        x_hat = self.f_o(stimuli[t-1]).reshape(9,2)
                        # x_hat = self.f_o(stimuli[t-1]).reshape(2,1)
                        self.x = x_hat
                else:
                    self.update_params(stimuli[t-1], t, F_x, F_params, self.H, y_hat, measurements_noisy)
                    self.params_dict = update_params_dic(self.params_dict, self.params_vec)
                    # Backpropagation and optimizer step for noise network
                if np.any(np.isnan(self.x)) or np.any(np.isnan(self.params_vec)):
                    break
                # Assign predicted values
                states_predicted[t] = self.x
                measurements_predicted[t] = y_hat
                
                # measurements_predicted[t] = y_hat
        # training_mse_states = []
        # testing_mse_states = []

        # # Loop over sources and states
        # for source in range(0, 2):
        #     for state in range(0, 9):
        #         # Compute the training and testing MSE for each (state, source)
        #         train_mse = np.mean((states_predicted[0:2500,0:9,:] - states[0:9,:,:2500].reshape(2500,9,2)) ** 2)
        #         test_mse = np.mean((states_predicted[2500:,0:9,:] - states[0:9,:,2500:].reshape(500,9,2)) ** 2)
        #         # Append to the lists
        #         training_mse_states.append([state, source, train_mse, test_mse])
            # self.params_vec  =self.initial_params_vec
            # self.Q_x = self.initial_Q_x
            # self.Q_params = self.initial_Q_params
            # self.R_y = self.R_y
            # self.Q_x = 0.9 * self.Q_x
            # self.R_y = 0.9 * self.R_y
            self.P_x = self.initial_P_x
            self.P_x_ = self.initial_P_x_
            self.P_params_ = self.initial_P_params_
            self.P_params = self.initial_P_params
            state_0_train_mse[i], state_0_test_mse[i], states_train_mse[i], states_test_mse[i] = compute_states_mse(states_predicted, states)
            measurements_train_mse[i], measurements_test_mse[i] = compute_measurements_mse(measurements_noisy, measurements_predicted)
            if np.any(np.isnan(self.x)) or np.any(np.isnan(self.params_vec)):
                self.x = np.zeros((self.aug_state_dim, self.sources))
                self.params_dict = {
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
                self.params_vec = self.initial_params_vec
                measurements_test_mse[i] += 1e6  
            
            self.params_dict = update_params_dic(self.params_dict, self.params_vec)
            params_dict_list.append(self.params_dict)
            # # measurements_predicted[t] = y_hat[:,0]
            print(f"state 0 train loss: {state_0_train_mse[i]}  state 0 test loss: {states_test_mse[i]}")
            print(f"states train loss: {states_train_mse[i]}  states test loss: {states_test_mse[i]}")
            print(f"measurement train loss: {measurements_train_mse[i]}  measurement test loss: {measurements_test_mse[i] }")
            # previous = testing_mse
            total_states_predicted.append(states_predicted)
            total_measurements_predicted.append(measurements_predicted)
            self.x = np.zeros((9,2))
            self.H = np.linalg.inv((states_predicted[0:t,0].T @ states_predicted[0:t,0]) + 1e-4 *np.eye(2)) @ states_predicted[0:t,0].T @ measurements_noisy[0:t]


            H_list.append(self.H)
        min_index = np.argmin(measurements_test_mse[:i])
        save_path = f'/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/unknown_H/{self.model}/{self.model_data}_data/'
        mse_to_csv(state_0_train_mse, state_0_test_mse, states_train_mse, states_test_mse,measurements_train_mse, measurements_test_mse, save_path)
        np.save(f'/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/unknown_H/{self.model}/{self.model_data}_data/params_predicted.npy', params_dict_list[min_index])
        np.save(f'/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/unknown_H/{self.model}/{self.model_data}_data/H_predicted.npy', H_list[min_index])
        print(min_index)
        # return states_predicted, measurements_predicted
        return total_states_predicted[min_index], total_measurements_predicted[min_index]
    
    def update_params(self, u, t, F_x, F_params, H, y_hat, y):
        dH = np.zeros((2,18))
        dH[0,0] = H[0,0]
        dH[0,1] = H[0,1]
        dH[1,0] = H[1,0]
        dH[1,1] = H[1,1]
        S = dH @ self.P_x_ @ dH.T + self.R_y 

        S_inv = np.linalg.inv(S)
        x_hat = self.f_o(u)
        # self.x = x_hat - self.P_x_ @ dH.T @ S_inv @ (y_hat[:,0] - y[t])
        self.x = x_hat - self.P_x_ @ dH.T @ S_inv @ (y_hat - y[t])
        I = np.eye(self.aug_state_dim_flattened)
        self.P_x_ = F_x @ self.P_x @ F_x.T + self.Q_x


        self.P_x = self.P_x_ @ (I + dH.T @ (self.R_y - dH @ self.P_x_ @ dH.T) @ dH @ self.P_x_)
        self.params_vec = self.params_vec - self.P_params_ @ F_params.T @ (x_hat - self.x)
        self.P_params_ = self.P_params - self.P_params @ F_params.T @ (self.Q_x + F_params @ self.P_params @ F_params.T) @ F_params @ self.P_params
        self.P_params = self.P_params_ + self.Q_params
        # self.Q_params = 0.9 *self.Q_params
        # self.R_y = 0.9*self.R_y
        self.x = self.x.reshape(self.aug_state_dim, self.sources)

    def jacobian_h(self, x):
        raise NotImplementedError("Derived classes should implement this method.")
    
    # The following methods need to be implemented in derived classes
    def f_o(self, x):
        raise NotImplementedError("Derived classes should implement this method.")
    
    def jacobian_f_o(self, x, u):
        raise NotImplementedError("Derived classes should implement this method.")
    
    def jacobian_f_o_x(self, x, u):
        raise NotImplementedError("Derived classes should implement this method.")
