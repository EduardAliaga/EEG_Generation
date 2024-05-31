import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from autograd import jacobian
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define necessary functions and variables
def sigmoid(x, theta):
    return 1 / (1 + np.exp(-x * theta))

def f_o(x, u, theta, W, tau, dt):
    return x + dt * (-x / tau + W @ sigmoid(x, theta) + u)

def g_o(param):
    return param

class EEGModel(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(EEGModel, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Jacobians (These are approximations, adjust according to your model's specifics)
def jacobian_f_o_x(x, theta, W, tau, dt):
    sigmoid_x = sigmoid(x, theta)
    diag_sigmoid = np.diag(sigmoid_x * (1 - sigmoid_x))
    return np.eye(len(x)) + dt * (-1 / tau * np.eye(len(x)) + W @ diag_sigmoid)

def jacobian_f_o_theta(x, theta, W, tau, dt):
    sigmoid_x = sigmoid(x, theta)
    return dt * W @ (x * sigmoid_x * (1 - sigmoid_x)).reshape(-1, 1)

def jacobian_f_o_W(x, theta, tau, dt):
    sigmoid_x = sigmoid(x, theta)
    return dt * np.outer(sigmoid_x, x)

def jacobian_f_o_tau(x, theta, W, tau, dt):
    return -dt / (tau ** 2) * x

def recursive_update(x_init, eeg_model, theta_init, W_init, tau_init, u, y, H, dt, Q_x, Q_theta, Q_W, Q_tau, R_y, num_iterations):
    x = x_init
    theta = theta_init
    W = W_init
    tau = tau_init
    
    for t in range(1, num_iterations):
        # Predicted state and parameter updates
        x_pred = f_o(x, u[t-1], theta, W, tau, dt)
        theta_pred = g_o(theta)
        W_pred = g_o(W)
        tau_pred = g_o(tau)

        # Compute Jacobians
        F_x = jacobian_f_o_x(x, theta, W, tau, dt)
        F_theta = jacobian_f_o_theta(x, theta, W, tau, dt)
        F_W = jacobian_f_o_W(x, theta, tau, dt)
        F_tau = jacobian_f_o_tau(x, theta, W, tau, dt)

        # Update state x
        P_x = Q_x + F_x @ Q_x @ F_x.T
        K_x = P_x @ H.T @ np.linalg.inv(H @ P_x @ H.T + R_y)
        x = x_pred + K_x @ (y[t] - eeg_model(x_pred, H))
        
        # Update parameter theta
        P_theta = Q_theta + F_theta @ Q_theta @ F_theta.T
        K_theta = P_theta @ F_theta.T @ np.linalg.inv(Q_x + F_theta @ P_theta @ F_theta.T)
        theta = theta_pred - K_theta @ (x_pred - x)
        
        # Update parameter W
        P_W = Q_W + F_W @ Q_W @ F_W.T
        K_W = P_W @ F_W.T @ np.linalg.inv(Q_x + F_W @ P_W @ F_W.T)
        W = W_pred - K_W @ (x_pred - x)
        
        # Update parameter tau
        P_tau = Q_tau + F_tau @ Q_tau @ F_tau.T
        K_tau = P_tau @ F_tau.T @ np.linalg.inv(Q_x + F_tau @ P_tau @ F_tau.T)
        tau = tau_pred - K_tau @ (x_pred - x)
    
    return x, theta, W, tau

# Example usage
# Define initial values and parameters
x_init = np.random.randn(10)
theta_init = np.random.randn()
W_init = np.random.randn(10, 10)
tau_init = 1.0
u = np.random.randn(100, 10)
y = np.random.randn(100, 10)
H = np.eye(10)
dt = 0.01
Q_x = np.eye(10)
Q_theta = np.eye(1)
Q_W = np.eye(10)
Q_tau = np.eye(1)
R_y = np.eye(10)
num_iterations = 100

x, theta, W, tau = recursive_update(x_init, theta_init, W_init, tau_init, u, y, H, dt, Q_x, Q_theta, Q_W, Q_tau, R_y, num_iterations)
print("Final x:", x)
print("Final theta:", theta)
print("Final W:", W)
print("Final tau:", tau)





