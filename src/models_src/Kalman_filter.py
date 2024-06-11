import sys
sys.path.insert(0, '../')
import data.get_raw_data as grd
from models_src.models import Measurement_Model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x, theta):
    # Clipping input values to prevent overflow in the exponential function
    x_clipped = np.clip(x * theta, -100, 100)
    return 1 / (1 + np.exp(-x_clipped))

def f_o(x, u, theta, W, M, tau, dt):

    return x + dt * (-x / tau + W @ sigmoid(x, theta) + M @ u)

def g_o(param):
    return param

class EEGModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EEGModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
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

def jacobian_f_o_M(x, theta, u, dt):
    return np.eye(3) * dt * u

def check_for_nan(matrix, name):
    if np.isnan(matrix).any() or np.isinf(matrix).any():
        print(f"{name} contains NaNs or Infs:")
        print(matrix)
        return True
    
    return False
def recursive_update(x, eeg_model, theta, W, M, tau, u, y, dt, Q_x, Q_theta, Q_w, Q_M, Q_tau, R_y, num_iterations):
    P_x_ = np.eye(3) * 1e-4
    P_x = np.eye(3) * 1e-4
    P_w_ = np.eye(3) * 1e-4
    P_w = np.eye(3) * 1e-4
    P_M_ = np.random.randn(3)
    P_M = np.random.randn(3)
    x_entire = []

    criterion = nn.MSELoss()
    optimizer = optim.Adam(eeg_model.parameters(), lr=0.001)
    
    regularization_term = 1e-8  # Small regularization term to prevent overflow
    
    for t in range(1, num_iterations):
        # Predicted state and parameter updates
        x_pred = f_o(x, u[t-1], theta, W, M, tau, dt)
        theta_pred = g_o(theta)
        W_pred = g_o(W)
        tau_pred = g_o(tau)
        M_pred = g_o(M)

        # Compute Jacobians
        F_x = jacobian_f_o_x(x, theta, W, tau, dt)
        F_theta = jacobian_f_o_theta(x, theta, W, tau, dt)
        F_W = jacobian_f_o_W(x, theta, tau, dt)
        F_M = jacobian_f_o_M(x, theta, u[t-1], dt)
        F_tau = jacobian_f_o_tau(x, theta, W, tau, dt)

        # Measurement prediction
        y_hat = eeg_model(torch.tensor(x_pred, dtype=torch.float32))
        state_dict = eeg_model.state_dict()
        fc1_weights = state_dict['fc1.weight']
        fc2_weights = state_dict['fc2.weight']
        fc3_weights = state_dict['fc3.weight']
        H = torch.mm(torch.mm(fc3_weights, fc2_weights), fc1_weights).numpy()
        I = np.eye(3)

        if check_for_nan(H, "H"):
            break

        # Add regularization term and check for small eigenvalues
        S = H @ P_x_ @ H.T + R_y + np.eye(H.shape[0]) * regularization_term
        if np.linalg.cond(S) > 1e10:
            print("Condition number is too high for matrix inversion")
            break

        # Numerically stable inversion
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Matrix inversion error in S")
            break

        # Clamp residuals to avoid extreme values
        residual = y[t] - y_hat.detach().numpy()
        residual_clamped = np.clip(residual, -10, 10)

        # Update state estimate
        x_update = x_pred - P_x_ @ H.T @ S_inv @ residual_clamped
        if check_for_nan(x_update, "x_update"):
            break
        x = x_update

        loss = criterion(y_hat, torch.tensor(y[t], dtype=torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update other parameters
        W = W_pred - P_w_ @ F_W.T @ np.linalg.inv(Q_x + F_W @ P_w @ F_W.T + np.eye(F_W.shape[0]) * regularization_term) @ (x_pred - x)
        M = M_pred - P_M_ @ F_M.T @ np.linalg.inv(Q_x + F_M @ P_M @ F_M.T + np.eye(F_M.shape[0]) * regularization_term) @ (x_pred - x)

        P_x_ = F_x @ P_x @ F_x.T + Q_x
        P_w_ = P_w_ - P_w @ F_W.T @ np.linalg.inv(Q_x + F_W @ P_w @ F_W.T + np.eye(F_W.shape[0]) * regularization_term) @ F_W @ P_w
        P_M_ = P_M_ - P_M @ F_M.T @ np.linalg.inv(Q_x + F_M @ P_M @ F_M.T + np.eye(F_M.shape[0]) * regularization_term) @ F_M @ P_M

        P_x = P_x_ @ (I + H @ (R_y - H @ P_x_ @ H.T) @ H @ P_x_)
        P_w = Q_w + P_w_
        P_M = Q_M + P_M_

        x_entire.append(x)
    
    return x, np.array(x_entire), theta, W, M, tau, eeg_model

# Example usage
# Define initial values and parameters
x_init = np.random.randn(3)
theta_init = 1
W_init = np.random.randn(3, 3)
M_init = np.random.randn(3, 3)
tau_init = 1000.0
raw_inquiries = grd.process_eeg_and_triggers(
    eeg_file_path='/Users/aliag/Desktop/Data/S002/S002_Matrix_Calibration_Thu_18_May_2023_12hr43min40sec_-0400/raw_data_2.csv',
    triggers_file_path='/Users/aliag/Desktop/Data/S002/S002_Matrix_Calibration_Thu_18_May_2023_12hr43min40sec_-0400/triggers_2.txt'
)

inquiries_per_channels = []
stimulis = []
for inquiry in raw_inquiries:
    inquiries_per_channels.append(np.array(inquiry[['Cz', 'O1', 'O2']]))
    stm = np.array(inquiry['stimuli_signal']) + 10
    stimulis.append([stm, stm, stm])

stimulis = stimulis[10:90]
inquiries_per_channels = inquiries_per_channels[10:90]

dt = 1/300
Q_x = np.eye(3) * 1e-4
Q_theta = np.eye(1)
Q_W = np.eye(3) * 1e-2
Q_M = np.eye(3) * 1e-2
Q_tau = np.eye(1)
R_y = np.eye(3) * 1e-3
num_iterations = 3030
input_dim = 3
hidden_dim = 256
eeg_model = EEGModel(input_dim, hidden_dim, input_dim)

# Initialize state and parameters
x = x_init
theta = theta_init
W = np.array([[ 0.32580474, -1.42557181, -0.56313676],
 [-1.9174992,  -1.86292142,  0.8255883 ],
 [-0.55589746, -0.02377605,  0.32360835]])
M = M_init
tau = tau_init
membrane_potentials = []
# Train on all data
for i in range(len(stimulis)):
    print(i)
    u = np.transpose(stimulis[i])
    y = inquiries_per_channels[i]
    x, x_entire, theta, W, M, tau = recursive_update(x, eeg_model, theta, W, M, tau, u, y, dt, Q_x, Q_theta, Q_W, Q_M, Q_tau, R_y, num_iterations)
    membrane_potentials.append(x_entire)

print("Final theta:", theta)
print("Final W:", W)
print("Final M:", M)
print("Final tau:", tau)

# Plot results
y_final = []
for t in membrane_potentials[len(membrane_potentials)-1]:
    y_final.append(eeg_model(torch.tensor(t, dtype=torch.float32)).detach().numpy())
y_final = np.array(y_final)
print(len(np.transpose(y_final)[0]))
plt.plot(np.transpose(y_final)[0])
#plt.plot(np.array(membrane_potentials[len(membrane_potentials)-1]))
plt.show()






