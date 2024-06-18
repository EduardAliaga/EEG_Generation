import sys
sys.path.insert(0, '../')
import data.get_raw_data as grd
from models_src.models import EEGModel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(x, theta):
    # Clipping input values to prevent overflow in the exponential function
    x_clipped = np.clip(x * theta, -100, 100)
    return 1 / (1 + np.exp(-x*theta))

def tanh(x):
    # Clipping input values to prevent overflow in the exponential function
    x_clipped = np.clip(x, -100, 100)
    return np.tanh(x_clipped)

def f_o(x, u, theta, W, M, tau, dt, function):
    if function == 'sigmoid':
        return x + dt * (-x / tau + W @ sigmoid(x, theta) + M @ u)
    if function == 'tanh':
        return x + dt * (-x / tau + W @ tanh(x) + M @ u)

def g_o(param):
    return param

def jacobian_f_o_x(x, theta, W, tau, dt, function):
    if function == 'sigmoid':
        fx = sigmoid(x, theta)
        diag_matrix = np.diag(fx * (1 - fx) * theta)
    if function == 'tanh':
        fx = tanh(x)
        diag_matrix = np.diag(1 - fx ** 2)
    return np.eye(len(x)) + dt * (-1 / tau * np.eye(len(x)) + W @ diag_matrix)

def jacobian_f_o_theta(x, theta, W, tau, dt, function):
    if function == 'sigmoid':
        fx = sigmoid(x, theta)
        return dt * W @ (x * fx * (1 - fx))
    if function == 'tanh':
        fx = tanh(x)
        return dt * W @ x * (1 - fx ** 2)
    return 'non valid function'

def jacobian_f_o_W(x, theta, tau, dt, function):
    if function == 'sigmoid':
        fx = sigmoid(x, theta)
    if function == 'tanh':
        fx = tanh(x)
    return dt * np.tile(fx, (len(x), 1))

def jacobian_f_o_tau(x, theta, W, tau, dt):
    return -dt / (tau ** 2) * x

def jacobian_f_o_M(x, theta, u, dt):
    return np.eye(3) * dt * u

def recursive_update(x, eeg_model, theta, W, M, tau, u, y, dt, Q_x, Q_theta, Q_w, Q_M, Q_tau, R_y, P_x_, P_x, P_w_, P_w, P_M_, P_M, P_theta_, P_theta,num_iterations):
    x_entire = []

    criterion = nn.MSELoss()
    optimizer = optim.Adam(eeg_model.parameters(), lr=0.001)
    regularization_term = 1e-8 # Small regularization term to prevent overflow
    f = 'sigmoid'
    for t in range(1, num_iterations):
        # Predicted state and parameter updates
        x_pred = f_o(x, u[t-1], theta, W, M, tau, dt, f)
        print(u[t-1])
        theta_pred = g_o(theta)
        W_pred = g_o(W)
        tau_pred = g_o(tau)
        M_pred = g_o(M)

        # Compute Jacobians
        F_x = jacobian_f_o_x(x, theta, W, tau, dt, f)
        F_theta = jacobian_f_o_theta(x, theta, W, tau, dt, f)
        F_W = jacobian_f_o_W(x, theta, tau, dt, f)
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

        # Add regularization term and check for small eigenvalues
        S = H @ P_x_ @ H.T + R_y + np.eye(H.shape[0]) * regularization_term

        S_inv = np.linalg.inv(S)
        # Clamp residuals to avoid extreme values
        residual = y[t] - y_hat.detach().numpy()

        # Update state estimate
        x = x_pred - P_x_ @ H.T @ S_inv @ residual

        loss = criterion(y_hat, torch.tensor(y[t], dtype=torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update other parameters
        W = W_pred - P_w_ @ F_W.T  @ (x_pred - x)
        M = M_pred - P_M_ @ F_M.T  @ (x_pred - x)
        theta = theta_pred - P_theta_ @ F_theta.T @ (x_pred - x)

        P_x_ = F_x @ P_x @ F_x.T + Q_x
        P_w_ = P_w_ - P_w @ F_W.T @ (Q_x + F_W @ P_w @ F_W.T) @ F_W @ P_w
        P_M_ = P_M_ - P_M @ F_M.T @ (Q_x + F_M @ P_M @ F_M.T) @ F_M @ P_M
        print(F_theta)
        P_theta_ = P_theta_ - (P_theta @ F_theta).T @ (Q_x + F_theta.T @ P_theta @ F_theta) @ (F_theta.T @ P_theta).T

        #P_w_ = P_w_ - P_w @ F_W.T @ stable_pseudo_inverse(Q_x + F_W @ P_w @ F_W.T + np.eye(F_W.shape[0]) * regularization_term) @ F_W @ P_w
        #P_M_ = P_M_ - P_M @ F_M.T @ stable_pseudo_inverse(Q_x + F_M @ P_M @ F_M.T + np.eye(F_M.shape[0]) * regularization_term) @ F_M @ P_M
        P_x = P_x_ @ (I + H @ (R_y - H @ P_x_ @ H.T) @ H @ P_x_)
        P_w = Q_w + P_w_
        P_M = Q_M + P_M_
        P_theta = Q_theta + P_theta_

        x_entire.append(x)
        """print("p_x_")
        print(P_x_)
        print("p_x")
        print(P_x)
        print("F_x")
        print(F_x)
        print('x')
        print(x)
        print('x_pred')
        print(x_pred)
        print("W")
        print(W)
    """
    return x, np.array(x_entire), theta, W, M, tau, P_x_, P_x, P_w_, P_w, P_M_, P_M, P_theta_, P_theta


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
Q_x = np.eye(3) * 1e-6
Q_theta = np.eye(3) * 1e-6
Q_W = np.eye(3) * 1e-4
Q_M = np.eye(3) * 1e-4
Q_tau = np.eye(1)
R_y = np.eye(3) * 1e-3

P_x_ = np.eye(3) * 1e-4
P_x = np.eye(3) * 1e-4
P_w_ = np.eye(3) * 1e-6
P_w = np.eye(3) * 1e-4
P_M_ = np.eye(3) * 1e-4
P_M = np.eye(3) * 1e-4 
P_theta_ = np.eye(3) * 1e-4
P_theta = np.eye(3) * 1e-4

num_iterations = 3030
input_dim = 3
hidden_dim = 256
eeg_model = EEGModel(input_dim, hidden_dim, input_dim)

x = np.random.randn(3)
theta = 1
W = np.random.randn(3, 3)
M = np.random.randn(3, 3)
tau = 100.0
membrane_potentials = []

epochs = 5
#for epoch in range(epochs):
for i in range(len(stimulis)):
    u = np.transpose(stimulis[i])
    y = inquiries_per_channels[i]
    x, x_entire, theta, W, M, tau, P_x_, P_x, P_w_, P_w, P_M_, P_M, P_theta_, P_theta = recursive_update(x, eeg_model, theta, W, M, tau, u, y, dt, Q_x, Q_theta, Q_W, Q_M, Q_tau, R_y, P_x_, P_x, P_w_, P_w, P_M_, P_M, P_theta_, P_theta, num_iterations)
    membrane_potentials.append(x_entire)

print("Final theta:", theta)
print("Final W:", W)
print("Final M:", M)
print("Final tau:", tau)

# Plot results
y_final = []
for t in membrane_potentials[-1]:
    y_final.append(eeg_model(torch.tensor(t, dtype=torch.float32)).detach().numpy())
y_final = np.array(y_final)
print(len(np.transpose(y_final)[0]))
print(len(np.array(membrane_potentials[-1])))

#plot prediction and the actual EEG
time_array = np.arange(len(np.transpose(y_final)[0]))* 0.003
time_array_2 = np.arange(len(np.transpose(inquiries_per_channels[-1])[0]))* 0.003
plt.figure(figsize=(10, 6))
plt.plot(time_array, np.transpose(y_final)[0], c = 'red', label='Predicted EEG Signal')
plt.plot(time_array_2, np.transpose(inquiries_per_channels[-1])[0], label='Actual EEG Signal')
plt.title('Comparison of Predicted and Actual EEG Signals')
plt.xlabel('Time(s)')
plt.ylabel('mV')
plt.legend()
plt.grid(True)
plt.show()

#plot membrane potentials

plt.figure(figsize=(10, 6))
for i, membrane_potential in enumerate(np.transpose(np.array(membrane_potentials[-1]))):
    time_array_3 = np.arange(len(membrane_potential))* 0.003
    plt.plot(time_array_3, membrane_potential, label=f'Membrane Potential {i+1}')
#plt.plot(time_array_2, stimulis[-1][0], c = 'orange', linestyle = '--', label='stimuli signal')
plt.title('Membrane Potentials Over Time')
plt.xlabel('Time')
plt.ylabel('mV')
plt.legend()
plt.grid(True)
plt.show()






