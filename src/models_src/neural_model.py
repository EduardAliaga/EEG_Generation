import numpy as np
from utils import *
import jax

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
        self.initial_params_dict = params_dict.copy()
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


    def fit(self, stimuli, measurements_noisy):
        # Initialize predictions arrays with appropriate dimensions
        num_time_points = len(stimuli)
        # for i in range(0,3):
        #     print(i)
        states_predicted = np.zeros((num_time_points, self.aug_state_dim, self.sources))
        measurements_predicted = np.zeros((num_time_points, self.sources))
        # Set initial state
        states_predicted[0] = self.x
        for t in range(1, num_time_points):
            F_x = self.jacobian_f_o_x(stimuli[t-1])
            F_params = self.jacobian_f_o(stimuli[t-1])
            print(t)
            y = measurements_noisy[t-1]
            y_hat = self.H @ self.x[0]
            self.update_params(stimuli[t-1], t, F_x, F_params, self.H, y_hat, measurements_noisy)
            if np.any(np.isnan(self.x)):
                self.x = np.ones((self.aug_state_dim, self.sources))
                self.params_dict = self.initial_params_dict
                self.params_vec = self.initial_params_vec
                self.P_x = self.initial_P_x
                self.P_x_ = self.initial_P_x_
                self.P_params = self.initial_P_params
                self.P_params_ = self.initial_P_params_
                self.Q_x = self.initial_Q_x
                self.Q_params = self.initial_Q_params
                break
            # Assign predicted values
            states_predicted[t] = self.x
            measurements_predicted[t] = y_hat
            self.params_dict = update_params_dic(self.params_dict, self.params_vec)

            # self.H = np.linalg.inv((states_predicted[0:t,0].T @ states_predicted[0:t,0]) + 1e-4 *np.eye(2)) @ states_predicted[0:t,0].T @ measurements_noisy[0:t]
        np.save('params_dict', self.params_dict)
            
        return states_predicted, measurements_predicted

    def test(self, stimuli):
        num_time_points = len(stimuli)
        states_predicted = np.zeros((num_time_points, self.aug_state_dim, self.sources))
        measurements_predicted = np.zeros((num_time_points, self.sources))
        states_predicted[0] = self.x.reshape(11,2)
        for t in range(1, num_time_points):
            self.x = self.x.reshape(11,2)
            #self.H = self.x[self.state_dim:self.state_dim + self.sources]
            y_hat = self.H @ self.x[0]
            # Update state prediction without measurements (i.e., prediction step only)
            x_hat = self.f_o(stimuli[t-1]).reshape(11,2)
            self.x = x_hat
            # Assign predicted values
            states_predicted[t] = self.x
            measurements_predicted[t] = y_hat
        return states_predicted, measurements_predicted
    
    def update_params(self, u, t, F_x, F_params, H, y_hat, y):
        dH = np.zeros((2,18))
        dH[0,0] = H[0,0]
        dH[0,1] = H[0,1]
        dH[1,0] = H[1,0]
        dH[1,1] = H[1,1]
        S = dH @ self.P_x_ @ dH.T + self.R_y 
        S_inv = np.linalg.inv(S)
        x_hat = self.f_o(u)
        self.x = x_hat - self.P_x_ @ dH.T @ S_inv @ (y_hat - y[t])
        I = np.eye(self.aug_state_dim_flattened)
        self.P_x_ = F_x @ self.P_x @ F_x.T + self.Q_x
        self.P_x = self.P_x_ @ (I + dH.T @ (self.R_y - dH @ self.P_x_ @ dH.T) @ dH @ self.P_x_)
        self.params_vec = self.params_vec - self.P_params_ @ F_params.T @ (x_hat - self.x)
        self.P_params_ = self.P_params - self.P_params @ F_params.T @ (self.Q_x + F_params @ self.P_params @ F_params.T) @ F_params @ self.P_params
        self.P_params = self.P_params_ + self.Q_params

        self.Q_x = self.dt * self.Q_x
        self.Q_params = self.dt * self.Q_params
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
