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
import data.get_raw_data as grd


# Define the EEGModel
class EEGModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EEGModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the functions
def sigmoid(x, theta):
    x_clipped = np.clip(x * theta, -100, 100)
    return 1 / (1 + np.exp(-x*theta))

def sigmoid_derivative(x, theta):
    sig = sigmoid(x, theta)
    return theta * sig * (1 - sig)

def f_o(x, u, theta, W, M, tau, dt, function):
    if function == 'sigmoid':
        return x + dt * (-x / tau + W @ sigmoid(x, theta) + M @ u)
    if function == 'tanh':
        return x + dt * (-x / tau + W @ np.tanh(x) + M @ u)

def g_o(param):
    return param

def jacobian_f_o_x(x, theta, W, tau, dt, function):
    if function == 'sigmoid':
        fx = sigmoid(x, theta)
        diag_matrix = np.diag(sigmoid_derivative(x, theta))
    elif function == 'tanh':
        fx = np.tanh(x)
        diag_matrix = np.diag(1 - fx ** 2)
    else:
        raise ValueError("Invalid function. Choose 'sigmoid' or 'tanh'.")
    return np.eye(len(x)) + dt * (-1 / tau * np.eye(len(x)) + W @ diag_matrix)

def jacobian_f_o_theta(x, theta, W, tau, dt, function):
    if function == 'sigmoid':
        fx_derivative_theta = sigmoid_derivative(x, theta)
        return dt * (W @ (x * fx_derivative_theta)).reshape(-1, 1)
    elif function == 'tanh':
        return np.zeros((len(x), 1))
    else:
        raise ValueError("Invalid function. Choose 'sigmoid' or 'tanh'.")

def jacobian_f_o_W(x, theta, tau, dt, function):
    if function == 'sigmoid':
        fx = sigmoid(x, theta)
    elif function == 'tanh':
        fx = np.tanh(x)
    else:
        raise ValueError("Invalid function. Choose 'sigmoid' or 'tanh'.")
    return dt * np.kron(np.eye(len(x)), fx)

def jacobian_f_o_M(x, theta, u, dt):
    return dt * np.kron(np.eye(len(x)), u)

def jacobian_f_o_tau(x, theta, W, tau, dt):
    return (dt * x / tau**2).reshape(-1, 1)

def jacobian_f_o(x, u, theta, W, M, tau, dt, function):
    F_W = jacobian_f_o_W(x, theta, tau, dt, function)
    F_M = jacobian_f_o_M(x, theta, u, dt)
    F_theta = jacobian_f_o_theta(x, theta, W, tau, dt, function)
    F_tau = jacobian_f_o_tau(x, theta, W, tau, dt)
    J_combined = np.hstack((F_W, F_M, F_theta, F_tau))
    return J_combined

def recursive_update (x, eeg_model, theta, W, M, tau, u, y, dt, P_x_, P_x, Q_x, P_params_, P_params, Q_params, R_y, num_iterations):
    x_entire = []

    criterion = nn.MSELoss()
    optimizer = optim.Adam(eeg_model.parameters(), lr=0.001)
    regularization_term = 1e-8  # Small regularization term to prevent overflow
    f = 'tanh'
    params = np.hstack((W.flatten(), M.flatten(), theta, tau))
    
    for t in range(1, num_iterations):
        # Recover W
        W = params[:9].reshape((3,3))
        
        # Recover M
        M = params[9:18].reshape((3,3))
        # Recover theta and tau
        theta = params[18]
        tau = params[19]

        x_pred = f_o(x, u[t-1], theta, W, M, tau, dt, f)
        params_pred = params

        F_x = jacobian_f_o_x(x, theta, W, tau, dt, f)
        F_params = jacobian_f_o(x, u[t-1], theta, W, M, tau, dt, f)

        y_hat = eeg_model(torch.tensor(x_pred, dtype=torch.float32))
        state_dict = eeg_model.state_dict()
        fc1_weights = state_dict['fc1.weight']
        fc2_weights = state_dict['fc2.weight']
        fc3_weights = state_dict['fc3.weight']
        H = torch.mm(torch.mm(fc3_weights, fc2_weights), fc1_weights).numpy()
        I = np.eye(3)

        S = H @ P_x_ @ H.T + R_y + np.eye(H.shape[0]) * regularization_term
        S_inv = np.linalg.inv(S)
        residual = y[t] - y_hat.detach().numpy()

        x = x_pred - P_x_ @ H.T @ S_inv @ residual
        residual_x = x_pred - x

        loss = criterion(y_hat, torch.tensor(y[t], dtype=torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        P_x_ = F_x @ P_x @ F_x.T + Q_x
        P_x = P_x_ @ (I + H @ (R_y - H @ P_x_ @ H.T) @ H @ P_x_)
        params = params_pred - P_params_ @ F_params.T @ (residual_x) 

        P_params_ =  P_params_ - P_params @ F_params.T @ (Q_x + F_params @ P_params @ F_params.T) @ F_params @ P_params

        P_params = P_params + Q_params
        x_entire.append(x)

    return x, np.array(x_entire), theta, W, M, tau, P_x_, P_x, P_params_, P_params 


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
R_y = np.eye(3) * 1e-4
P_x_ = np.eye(3) * 1e-4
P_x = np.eye(3) * 1e-4
P_params_ = np.eye(20)*1e-4
P_params = np.eye(20)*1e-4
Q_params = np.eye(20)*1e-4
num_iterations = 3030
input_dim = 3
hidden_dim = 256
eeg_model = EEGModel(input_dim, hidden_dim, input_dim)

#Initial conditions
x = np.random.randn(3)
theta = 1
W = np.random.randn(3, 3)
M = np.random.randn(3, 3)
tau = 10
membrane_potentials = []

epochs = 5
#for epoch in range (epochs):
for i in range(len(stimulis)):
    print(f"Processing sample {i+1}/{len(stimulis)}")
    u = np.transpose(stimulis[i])
    y = inquiries_per_channels[i]
    x, x_entire, theta, W, M, tau, P_x_, P_x, P_params_, P_params = recursive_update(
        x, eeg_model, theta, W, M, tau, u, y, dt, P_x_, P_x, Q_x, P_params_, P_params, Q_params, R_y, num_iterations)
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
print("Length of predicted EEG signal:", len(np.transpose(y_final)[0]))
print("Length of membrane potentials:", len(np.array(membrane_potentials[-1])))

# Plot prediction and the actual EEG
time_array = np.arange(len(np.transpose(y_final)[0])) * 0.003
time_array_2 = np.arange(len(np.transpose(inquiries_per_channels[-1])[0])) * 0.003
plt.figure(figsize=(10, 6))
plt.plot(time_array, np.transpose(y_final)[0], c='red', label='Predicted EEG Signal')
plt.plot(time_array_2, np.transpose(inquiries_per_channels[-1])[0], label='Actual EEG Signal')
plt.title('Comparison of Predicted and Actual EEG Signals')
plt.xlabel('Time (s)')
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

