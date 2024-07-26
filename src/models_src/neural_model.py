import numpy as np
from utils import *
import jax

"""def measurement_function(x):
    return x[2:6].reshape(2,2) @ x[0:2]"""


#TODO: change the name of the membrane potentials for state tmore generic
class NeuralModel:
    def __init__(self, state_dim = 9, aug_state_dim = 11, n_iterations = int(1e3), sources = 2, dt = 1e-2):
        #TODO: should I put the constant variables out of the class?

        self.dt = dt
        self.state_dim = state_dim
        self.aug_state_dim = aug_state_dim
        self.n_iterations = n_iterations
        self.sources = sources
        self.aug_state_dim_flattened = self.aug_state_dim * self.sources
        self.x = np.zeros((self.aug_state_dim, self.sources))
        self.H = np.eye(self.sources)
        self.x[self.state_dim:self.state_dim + self.sources] = self.H
        self.W = np.zeros((self.aug_state_dim, self.aug_state_dim))
        self.W[0,0] = 1
        self.W[1,1] = 1 
        self.M = 0.5
        self.tau = 80.0
        self.theta = 0.5
        self.params_dict = {
            'W': self.W,
            'M': self.M,
            'tau': self.tau,
        }
        self.params_vec = np.hstack((self.W.flatten(), self.M, self.tau))
        self.n_params = 23
        #TODO: put the covariance matrices as it is done with dt
        self.Q_x = np.eye(self.aug_state_dim_flattened) * 1e-6
        self.R_y = np.eye(self.sources) * 1e-6
        self.P_x_ = np.eye(self.aug_state_dim_flattened) * 1e-6
        self.P_x = np.eye(self.aug_state_dim_flattened) * 1e-6
        self.P_params_ = np.eye(self.n_params) * 1e-6
        self.P_params = np.eye(self.n_params) * 1e-6
        self.Q_params = np.eye(self.n_params) * 1e-6


    def fit(self, stimuli, measurements_noisy):
        # Initialize predictions arrays with appropriate dimensions
        num_time_points = len(stimuli)
        #states_predicted = np.zeros((num_time_points, self.aug_state_dim))
        states_predicted = np.zeros((num_time_points, self.aug_state_dim, self.sources))
        #measurements_predicted = np.zeros((num_time_points, self.state_dim))
        measurements_predicted = np.zeros((num_time_points, self.sources))
        # Set initial state
        states_predicted[0] = self.x

        for t in range(1, num_time_points):
            # TODO: define the jacobians as self. also 
            F_x = self.jacobian_f_o_x(stimuli[t-1])
            F_x = self.jacobian_f_o_x(stimuli[t-1])
            F_params = self.jacobian_f_o(stimuli[t-1])

            #H = self.x[self.state_dim:self.aug_state_dim].reshape((self.state_dim, self.state_dim))
            self.H = self.x[self.state_dim:self.state_dim + self.sources]
            y_hat = self.H @ self.x[0]
            dH = self.jacobian_h(self.x)

            self.update_params(stimuli[t-1], t, F_x, F_params, dH, y_hat, measurements_noisy)

            # Assign predicted values
            states_predicted[t] = self.x
            measurements_predicted[t] = y_hat
            #self.params_dict = vector_to_params(self.aug_state_dim, self.params_vec)

        return states_predicted, measurements_predicted

    def test(self, X):
        # Test the model using the input data X and return predictions.
        pass
    
    def update_params(self, u, t, F_x, F_params, dH, y_hat, y):
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
        #dH = np.array(self.jac_measurement_f_dH(x)).reshape(self.state_dim,self.aug_state_dim)
        raise NotImplementedError("Derived classes should implement this method.")
    
    # The following methods need to be implemented in derived classes
    def f_o(self, x):
        raise NotImplementedError("Derived classes should implement this method.")
    
    def jacobian_f_o(self, x, u):
        raise NotImplementedError("Derived classes should implement this method.")
    
    def jacobian_f_o_x(self, x, u):
        raise NotImplementedError("Derived classes should implement this method.")
