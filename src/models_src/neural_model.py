import numpy as np
from utils import *
import jax
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import pandas as pd
import os

def compute_measurements_mse(measurements_predicted, measurements_real):
    n_time_points = len(measurements_real[:,0])
    train_index = int(0.8*n_time_points)
    training_mse_measurements = np.mean((measurements_predicted[0:train_index] - measurements_real[0:train_index]) ** 2)
    testing_mse_measurements = np.mean((measurements_predicted[train_index:] - measurements_real[train_index:]) ** 2)
    return training_mse_measurements, testing_mse_measurements

def compute_states_mse(states_predicted, states_real):
        n_time_points = len(states_real[0,0,:])
        train_index = int(0.8*n_time_points)
        training_mse_states = np.mean((states_predicted[0:train_index,:,:] - states_real[0:9,:,:train_index].reshape(train_index,9,3)) ** 2)
        testing_mse_states = np.mean((states_predicted[train_index:,:,:] - states_real[0:9,:,train_index:].reshape(n_time_points-train_index,9,3)) ** 2)
        # training_mse_state_x0 = np.mean((states_predicted[0:train_index,0,:] - states_real[0,:,:train_index].T) ** 2)
        # testing_mse_state_x0 = np.mean((states_predicted[train_index:,0,:] - states_real[0,:,train_index:].T) ** 2)
        return training_mse_states, testing_mse_states
    
def mse_to_csv(training_mse_states, testing_mse_states,training_mse_measurements, testing_mse_measurements, save_path):
        # Create a dictionary for the data with iterations as the first column
    iterations = np.arange(1, 31)
    
    data = {
        'iterations': iterations,
        "training_mse_states": training_mse_states,
        "testing_mse_states": testing_mse_states,
        "training_mse_measurements": training_mse_measurements,
        "testing_mse_measurements":testing_mse_measurements,
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

        self.theta = .56
        self.H_e = 2.08
        self.tau_e = 1.39
        self.H_i = 16.0
        self.tau_i = 32.0
        self.gamma_1 = 1.0
        self.gamma_2= 4/5
        self.gamma_3 = 1/4
        self.gamma_4 = 1/4  # gamma_3 value

    def fit(self, stimuli, measurements_noisy, states, experiment): 
        num_time_points = len(stimuli)
        total_measurements_predicted = []
        total_states_predicted = []
        iterations = 30
        measurements_test_mse = np.zeros(iterations)
        measurements_train_mse = np.zeros(iterations)
        states_train_mse = np.zeros(iterations)
        states_test_mse = np.zeros(iterations)
        params_dict_list = []
        H_list = []
        for i in tqdm(range(0,30)):
            print(i)
            states_predicted = np.zeros((num_time_points, self.aug_state_dim, self.sources))
            measurements_predicted = np.zeros((num_time_points, self.sources))
            # Set initial state
            states_predicted[0] = self.x
            for t in tqdm(range(num_time_points)):
                F_x = self.jacobian_f_o_x(stimuli[t-1])
                F_params = self.jacobian_f_o(stimuli[t-1])
                y = measurements_noisy[t-1]
                # Measurement prediction: h(x) = H @ x (your existing measurement function)
                y_hat = self.H @ self.x[0]
                # y_hat = self.H @ self.x
                if t>=int(num_time_points*0.8):
                        #y_hat = self.H @ self.x
                        # Update state prediction without measurements (i.e., prediction step only)
                        x_hat = self.f_o(stimuli[t-1]).reshape(9,self.sources)
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

            self.P_x = self.initial_P_x
            self.P_x_ = self.initial_P_x_
            self.P_params_ = self.initial_P_params_
            self.P_params = self.initial_P_params
            states_train_mse[i], states_test_mse[i] = compute_states_mse(states_predicted, states)
            measurements_train_mse[i], measurements_test_mse[i] = compute_measurements_mse(measurements_noisy, measurements_predicted)

            if np.any(np.isnan(self.x)) or np.any(np.isnan(self.params_vec)):
                self.x = np.ones((self.aug_state_dim, self.sources))
                noise = 0.1
                self.params_dict = {
                    'C_f' : np.zeros(2),
                    'C_l' : np.zeros(2),
                    'C_u' : np.zeros(1),
                    'C_b' : np.zeros(1)
                            }
                self.params_vec = self.initial_params_vec
                measurements_test_mse[i] += 1e6  
                self.P_x = self.initial_P_x
                self.P_x_ = self.initial_P_x_
                self.P_params_ = self.initial_P_params_
                self.P_params = self.initial_P_params
                self.H = np.random.randn(3,3)

            self.params_dict = update_params_dic(self.params_dict, self.params_vec)
            params_dict_list.append(self.params_dict)
            # # measurements_predicted[t] = y_hat[:,0]

            print(f"states train loss: {states_train_mse[i]}  states test loss: {states_test_mse[i]}")
            print(f"measurement train loss MSE: {measurements_train_mse[i]}  measurement test loss MSE: {measurements_test_mse[i] }")
            # previous = testing_mse
            total_states_predicted.append(states_predicted)
            total_measurements_predicted.append(measurements_predicted)
            states_predicted_new_params = []
            # for t in range(0,3000):
            #     self.x = self.x.reshape(9,3)
            #     states_predicted_new_params.append(self.f_o(stimuli[t-1]).reshape(9,self.sources))
            # states_predicted_new_params = np.array(states_predicted_new_params)
            self.x = np.ones((9,self.sources))

            self.H = np.linalg.inv((states_predicted[0:t,0].T @ states_predicted[0:t,0]) + 1e-4 *np.eye(self.sources)) @ states_predicted[0:t,0].T @ measurements_noisy[0:t]
            # self.H = np.exp(self.H)


            
            H_list.append(self.H)

        min_index = np.argmin(measurements_test_mse[:i])
        if measurements_test_mse[min_index]<100:
            print('hello')
        save_path = f'/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/unknown_H/0_start/experiment_{experiment}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        mse_to_csv(states_train_mse, states_test_mse,measurements_train_mse, measurements_test_mse, save_path)
        with open(os.path.join(save_path, 'params_predicted.txt'), 'w') as f: 
            for key, value in params_dict_list[min_index].items():
                f.write(f"{key}: {value}\n")
            if min_index < len(params_dict_list):
                for key, value in params_dict_list[min_index+1].items():
                    f.write(f"{key}: {value}\n")
        np.savetxt(os.path.join(save_path, 'H_predicted.txt'), H_list[min_index], fmt='%.6f')
        print(min_index)
        # return states_predicted, measurements_predicted
        return total_states_predicted[min_index], total_measurements_predicted[min_index]
    
    def update_params(self, u, t, F_x, F_params, H, y_hat, y):
        dH = np.zeros((self.sources,self.sources*self.state_dim))
        dH[0,0] = H[0,0]
        dH[0,1] = H[0,1]
        dH[0,2] = H[0,2]
        dH[1,0] = H[1,0]
        dH[1,1] = H[1,1]
        dH[1,2] = H[1,2]
        dH[2,0] = H[2,0]
        dH[2,1] = H[2,1]
        dH[2,2] = H[2,2]
        S = dH @ self.P_x_ @ dH.T + self.R_y 

        S_inv = np.linalg.inv(S+np.eye(3)*1e-6)
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
