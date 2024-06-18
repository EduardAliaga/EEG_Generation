import sys
sys.path.insert(0, '../')
import numpy.random as rnd
import data.get_raw_data as grd
from models_src.models import Measurement_Model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define sigmoid
def sigmoid(x, theta):
     x_clipped = np.clip(x * theta, -100, 100)
     return 1 / (1 + np.exp(-x*theta))

# # Load raw inquiries
# raw_inquiries = grd.process_eeg_and_triggers(
#     eeg_file_path='/Users/aliag/Desktop/Data/S002/S002_Matrix_Calibration_Thu_18_May_2023_12hr43min40sec_-0400/raw_data_2.csv',
#     triggers_file_path='/Users/aliag/Desktop/Data/S002/S002_Matrix_Calibration_Thu_18_May_2023_12hr43min40sec_-0400/triggers_2.txt'
# )

# inquiries_per_channels = []
# stimulis = []
# for inquiry in raw_inquiries:
#     inquiries_per_channels.append(np.array(inquiry[['Cz', 'O1', 'O2']]))
#     stm = np.array(inquiry['stimuli_signal'])
#     stimulis.append([stm, stm, stm])

# stimulis = stimulis[10:90]
# inquiries_per_channels = inquiries_per_channels[10:90]
# Check shapes of membrane potentials
# Define the EEGModel

# Define the functions
def get_norm_squared_error(x, x_hat, regularization_term=1e-6):
    return get_squared_error(x,x_hat) / (x + regularization_term)**2

def get_squared_error(x, x_hat):
    return ((x-x_hat)**2)
def f_o(x, u, theta, W, M, tau, dt, function):

    return x + dt * (-x / tau + W @ x + M * u)

def g_o(param):
    return param

def jacobian_f_o_x(x, theta, W, tau, dt, function):
    return (1 - dt / tau) * np.eye(2) + dt * W

def jacobian_f_o_W(x, theta, tau, dt, function):
    F_W = np.array([[dt*x[0],dt*x[1],0,0],[0,0,dt*x[0], dt*x[1]]])
    return F_W

def jacobian_f_o_M(x, theta, u, dt):
    return dt * u

def jacobian_f_o_tau(x, theta, W, tau, dt):
    return (dt * x / tau**2)

def jacobian_f_o(x, u, theta, W, M, tau, dt, function):
    F_W = jacobian_f_o_W(x, theta, tau, dt, function)
    F_M = jacobian_f_o_M(x, theta, u, dt)
    F_M_array = np.array([F_M,F_M]).reshape(2, 1)
    F_tau = jacobian_f_o_tau(x, theta, W, tau, dt).reshape(2, 1)
    J_combined = np.hstack((F_W, F_M_array, F_tau))
    return J_combined

def recursive_update (x, theta, W, M, H, tau, u, y, dt, P_x_, P_x, Q_x, P_params_, P_params, Q_params,R_y, num_iterations):
    membrane_potentials_predicted = []
    f = 'tanh'
    dim_latent = len(x)
    params = np.hstack((W.flatten(), M, tau))
    membrane_potentials_predicted.append(x)
    for t in range(1, num_iterations):
 
        W = params[:dim_latent**2].reshape((dim_latent, dim_latent))
        M = params[dim_latent**2]
        tau = params[dim_latent**2 + 1]
        params_hat = g_o(params)

        F_x = jacobian_f_o_x(x, theta, W, tau, dt, f)
        F_params = jacobian_f_o(x, u[t-1], theta, W, M, tau, dt, f)

        x_hat = f_o(x, u[t-1], theta, W, M, tau, dt, f)    
        y_hat = H @ x_hat

        x_pred_array = np.array([x_hat])
        H = np.linalg.inv(x_pred_array.T @ x_pred_array) @ x_pred_array.T @ np.array([y_hat])
        #With a fixed and known H, the algorithm works fine (the parameters obtained are not the same but the result is)

        S = H @ P_x_ @ H.T + R_y 
        S_inv = np.linalg.inv(S)

        x = x_hat - P_x_ @ H.T @ S_inv @ (y_hat - y[t])

        I = np.eye(2)
        P_x_ = F_x @ P_x @ F_x.T + Q_x
        P_x = P_x_ @ (I + H @ (R_y - H @ P_x_ @ H.T) @ H @ P_x_)

        params = params_hat - P_params_ @ F_params.T @ (x_hat - x) 

        P_params_ =  P_params - P_params @ F_params.T @ (Q_x + F_params @ P_params @ F_params.T) @ F_params @ P_params
        P_params = P_params_ + Q_params

        membrane_potentials_predicted.append(x)
            
    return x, np.array(membrane_potentials_predicted), W, M, H, tau, P_x_, P_x, P_params_, P_params


n_stimuli = int(1e3)
# Create synthetic stimuli
period_square = 10
stimuli = np.zeros(n_stimuli)
for i_stimulus in range(0, n_stimuli, period_square):
    if (i_stimulus // period_square) % 2:
        stimuli[i_stimulus: i_stimulus + period_square] = np.ones(period_square)

tau = 1e2
dt = 1
theta = 1.0
W = np.zeros((2,2))
W[0,1] = 1e-1
W[1,0] = -1e-1
# M = np.zeros((2,2))
M = 100

membrane_potentials = np.zeros((n_stimuli, 2))
membrane_potentials[0] = np.array([-70, -70])
H = np.array([[-1, 1], [0.5, 0.5]])
measurements = np.zeros((n_stimuli, 2))
measurements[0] = H @ membrane_potentials[0]
# Generate membrane potentials
for t in range(1, n_stimuli):
    x = membrane_potentials[t-1]
    membrane_potentials[t] = x + dt * (-x / tau + W @ x + M * stimuli[t-1])
    measurements[t] = H @ membrane_potentials[t] 
seed_measurements = 2002
rng = rnd.default_rng(seed_measurements)
measurements_noisy = measurements + rng.multivariate_normal(mean=np.zeros(2), cov=np.eye(2), size=n_stimuli)


Q_x = np.eye(2) * 1e-8
R_y = np.eye(2) * 1e-8
P_x_ = np.eye(2) * 1e-8
P_x = np.eye(2) * 1e-8
P_params_ = np.eye(6) * 1e-8
P_params = np.eye(6) * 1e-8
Q_params = np.eye(6) * 1e-8
W_init = np.array([[0,0.2],[-0.5, 0]])
M_init = 30
H_init = np.array([[1.4, 1], [0.4, 0.6]])

membrane_potentials_predicted = []
x, membrane_potentials_n, W, M, H, tau, P_x_, P_x, P_params_, P_params = recursive_update(
    x, theta, W_init, M_init, H, tau, stimuli, measurements_noisy, dt, P_x_, P_x, Q_x, P_params_, P_params, Q_params, R_y, n_stimuli)
membrane_potentials_predicted.append(membrane_potentials_n)
membrane_potentials_predicted = np.array(membrane_potentials_predicted)

print(f"Membrane potentials shape: {membrane_potentials_predicted.shape}")

y = []
for t in range(0,1000):
    y.append(H @ membrane_potentials_predicted[0][t])
y = np.array(y)
print(measurements.shape)
norm_squared_error_x = get_norm_squared_error(membrane_potentials, membrane_potentials_predicted)
norm_squared_error_y = get_norm_squared_error(measurements, y)

#plot membrane potentials 
plt.figure("Membrane Potentials")
plt.plot(membrane_potentials[:,0])
plt.plot(membrane_potentials[:,1])

#plot membrane potentials predicted from the kalman filter
plt.figure("Membrane Potentials predicted")
plt.plot(np.transpose(membrane_potentials_predicted)[0])
plt.plot(np.transpose(membrane_potentials_predicted)[1])

#plot predictions from the membrane potentials predicted
plt.figure("predictions")
plt.plot(y)

plt.figure("Measurements")
plt.plot(measurements[:, 0])
plt.plot(measurements[:, 1])

plt.figure("Noisy Measurements")
plt.plot(measurements_noisy[:, 0])
plt.plot(measurements_noisy[:, 1])

plt.figure("norm squared error membrane potential")
plt.semilogy(norm_squared_error_x[0])

plt.figure("norm squared error measurements")
plt.semilogy(norm_squared_error_y.T[0], c = 'b')
plt.semilogy(norm_squared_error_y.T[1], c = 'r')

plt.figure("predictions vs measurements")
plt.plot(y.T[0], c='b', linestyle = '--')
plt.plot(y.T[1], c = 'r', linestyle = '--')
plt.plot(measurements[:, 0], c = 'b', alpha=0.3)
plt.plot(measurements[:, 1], c = 'r', alpha=0.3)

plt.show()