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
     
     return theta[0] / (1 + np.exp(-np.dot(theta[1: ], x)))


def sigmoid_derivative(x, theta, d):
    sig = sigmoid(x, theta)

    # TODO: I think the follwoing equations are not correct.
    # TODO: What about the derivative for the parameter that defines 
    # the amplitude of the sigmoid?
    if d == 'd_x':
        return theta * sig * (1 - sig / theta[0])
    elif d == 'd_theta':
        return x * sig * (1 - sig / theta[0])
    elif d == 'd_theta_0':
        return sig / theta[0]


# Define the functions
def get_norm_squared_error(x, x_hat, regularization_term=1e-6):
    return get_squared_error(x,x_hat) / (x + regularization_term)**2

def get_squared_error(x, x_hat):
    return ((x-x_hat)**2)

def f_o(x, u, theta, W, M, tau, dt, function):
    if function == 'sigmoid':
        return x + dt * (-x / tau + W @ sigmoid(x, theta) + M * u)
    if function == 'tanh':
        return x + dt * (-x / tau + W @ np.tanh(x) + M * u)
    if function == 'linear':
        return x + dt * (-x / tau + W @ x + M * u)

def g_o(param):
    return param

def jacobian_f_o_x(x, theta, W, tau, dt, function):
    if function == 'sigmoid':
        fx = sigmoid(x, theta)
        diag_matrix = np.diag(sigmoid_derivative(x, theta,'d_x'))
    elif function == 'tanh':
        fx = np.tanh(x)
        diag_matrix = np.diag(1 - fx ** 2)
    elif function == 'linear':
        return (1 - dt / tau) * np.eye(2) + dt * W
    return np.eye(len(x)) + dt * (-1 / tau * np.eye(len(x)) + W @ diag_matrix)

def jacobian_f_o_W(x, theta, tau, dt, function):
    if function == 'sigmoid':
        F_W = np.array([[dt*sigmoid(x[0], theta),dt*sigmoid(x[1], theta),0,0],[0,0,dt*sigmoid(x[0], theta), dt*sigmoid(x[1], theta)]])
    elif function == 'linear':
        F_W = np.array([[dt*x[0],dt*x[1],0,0],[0,0,dt*x[0], dt*x[1]]])
    return F_W

def jacobian_f_o_M(x, theta, u, dt):
    return dt * u

def jacobian_f_o_tau(x, theta, W, tau, dt):
    return (dt * x / tau**2)

def jacobian_f_o_theta(x, theta, W, tau, dt, function):
    derivative = np.array([sigmoid_derivative(x, theta, 'd_theta')])
    return dt*(W @ derivative.T)

def jacobian_f_o(x, u, theta, W, M, tau, dt, function):
    F_W = jacobian_f_o_W(x, theta, tau, dt, function)
    F_M = jacobian_f_o_M(x, theta, u, dt)
    F_M_array = np.array([F_M,F_M]).reshape(2, 1)
    F_tau = jacobian_f_o_tau(x, theta, W, tau, dt).reshape(2, 1)
    F_theta = jacobian_f_o_theta(x, theta, W, tau, dt, function)
    if function == 'sigmoid':
        J_combined = np.hstack((F_W, F_M_array, F_tau, F_theta))
    else:
        J_combined = np.hstack((F_W, F_M_array, F_tau))
    return J_combined

def recursive_update (x, theta, W, M, H, tau, u, y, dt, P_x_, P_x, Q_x, P_params_, P_params, Q_params,R_y, function, num_iterations):
    membrane_potentials_predicted = []
    dim_latent = len(x)
    if function == 'sigmoid':
        params = np.hstack((W.flatten(), M, tau, theta_init))
    else:
        params = np.hstack((W.flatten(), M, tau))
    membrane_potentials_predicted.append(x)
    for t in range(1, num_iterations):
 
        W = params[:dim_latent**2].reshape((dim_latent, dim_latent))
        M = params[dim_latent**2]
        tau = params[dim_latent**2 + 1]

        if function == 'sigmoid':
            theta = params[dim_latent**2 + 2]

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
#H = np.array([[1, 0.7], [0.5, 0.8]])
measurements = np.zeros((n_stimuli, 2))
measurements[0] = H @ membrane_potentials[0]
# Generate membrane potentials

f = 'sigmoid'
for t in range(1, n_stimuli):
    x = membrane_potentials[t-1]
    if f == 'sigmoid':
        membrane_potentials[t] = x + dt * (-x / tau + W @ sigmoid(x, theta) + M * stimuli[t-1])
    elif f == 'tanh':
        membrane_potentials[t] = x + dt * (-x / tau + W @ np.tanh(x) + M * stimuli[t-1])
    elif f == 'linear':
        membrane_potentials[t] = x + dt * (-x / tau + W @ x + M * stimuli[t-1])
    measurements[t] = H @ membrane_potentials[t] 
seed_measurements = 2002
rng = rnd.default_rng(seed_measurements)
measurements_noisy = measurements + rng.multivariate_normal(mean=np.zeros(2), cov=np.eye(2), size=n_stimuli)

if f == 'sigmoid':
    n_params = 7
else:
    n_params = 6
dim_timestep = len(x)
Q_x = np.eye(dim_timestep) * 1e-8
R_y = np.eye(dim_timestep) * 1e-8
P_x_ = np.eye(dim_timestep) * 1e-8
P_x = np.eye(dim_timestep) * 1e-8
P_params_ = np.eye(n_params) * 1e-8
P_params = np.eye(n_params) * 1e-8
Q_params = np.eye(n_params) * 1e-8
W_init = np.array([[0,0.2],[-0.5, 0]])
M_init = 30
H_init = np.array([[-1.4, -0.5], [-0.4, -0.6]])
theta_init = 0.5

membrane_potentials_predicted = []
x, membrane_potentials_n, W, M, H, tau, P_x_, P_x, P_params_, P_params = recursive_update(
    x, theta_init, W_init, M_init, H, tau, stimuli, measurements_noisy, dt, P_x_, P_x, Q_x, P_params_, P_params, Q_params, R_y, f, n_stimuli)
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


plt.figure("Stimuli")
plt.plot(stimuli, c = 'r', label = "Stimuli")
plt.title("Stimuli")
plt.xlabel("Time (ms)")
plt.legend()

#plot membrane potentials 
plt.figure("Synthetic Membrane Potentials")
plt.plot(membrane_potentials[:, 0], c = '#1f77b4', label="Membrane Potential 1")
plt.plot(membrane_potentials[:, 1], c = 'orange', label="Membrane Potential 2")
plt.title("Synthetic Membrane Potentials")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()

# Membrane Potentials predicted from the Kalman filter
plt.figure("Predicted Membrane Potentials")
plt.plot(np.transpose(membrane_potentials_predicted)[0], c = '#1f77b4', label = "Membrane Potential 1")
plt.plot(np.transpose(membrane_potentials_predicted)[1], c = 'orange', label = "Membrane Potential 2")
plt.title("Predicted Membrane Potentials")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.legend()

# Predictions from the membrane potentials predicted
plt.figure("Predicted Measurements")
plt.plot(y.T[0], c = '#1f77b4', label = "Predicted Measurement 1")
plt.plot(y.T[1], c = 'orange', label = "Predicted Measurement 2")
plt.title("Predicted Measurements")
plt.xlabel("Time (ms)")
plt.ylabel("EEG potential (mV)")
plt.legend()

# Measurements
plt.figure("Synthetic Measurements")
plt.plot(measurements[:, 0], c = '#1f77b4', label = "Measurement 1")
plt.plot(measurements[:, 1], c = 'orange', label = "Measurement 2")
plt.title("Synthetic Measurements")
plt.xlabel("Time (ms)")
plt.ylabel("EEG potential (mV)")
plt.legend()

# Noisy Measurements
plt.figure("Synthetic Noisy Measurements")
plt.plot(measurements_noisy[:, 0], c = '#1f77b4', label="Noisy Measurement 1")
plt.plot(measurements_noisy[:, 1], c = 'orange', label="Noisy Measurement 2")
plt.title("Synthetic Noisy Measurements")
plt.xlabel("Time (ms)")
plt.ylabel("EEG potential (mV)")
plt.legend()

# Norm squared error membrane potential
plt.figure("Norm Squared Error Membrane Potential")
plt.semilogy(norm_squared_error_x[0].T[0], c = 'b', label="Error in Membrane Potential 1")
plt.semilogy(norm_squared_error_x[0].T[1], c = 'r', label="Error in Membrane Potential 2")
plt.title("Norm Squared Error in Membrane Potential")
plt.xlabel("Time (ms)")
plt.ylabel("Norm Squared Error")
plt.legend()

# Norm squared error measurements
plt.figure("Norm Squared Error Measurements")
plt.semilogy(norm_squared_error_y.T[0], c = 'b', label="Error in Measurement 1")
plt.semilogy(norm_squared_error_y.T[1], c = 'r', label="Error in Measurement 2")
plt.title("Norm Squared Error in Measurements")
plt.xlabel("Time (ms)")
plt.ylabel("Norm Squared Error")
plt.legend()

# Predictions vs Measurements
plt.figure("Predictions vs Synthetic Measurements")
plt.plot(y.T[0], c = '#1f77b4', linestyle='--', label = "Predicted Measurement 1")
plt.plot(y.T[1], c = 'orange', linestyle='--', label = "Predicted Measurement 2")
plt.plot(measurements[:, 0], c = '#1f77b4', alpha=0.3, label = "Actual Measurement 1")
plt.plot(measurements[:, 1], c ='orange', alpha=0.3, label = "Actual Measurement 2")
plt.title("Predictions vs Measurements")
plt.xlabel("Time (ms)")
plt.ylabel("EEG potential (mV)")
plt.legend()

plt.figure("Predicted Membrane Potentials vs Synthetic Membrane Potentials")
plt.plot(np.transpose(membrane_potentials_predicted)[0], c = '#1f77b4', linestyle='--', label = "Predicted Membrane Potential 1")
plt.plot(np.transpose(membrane_potentials_predicted)[1], c = 'orange', linestyle='--', label = "Predicted Membrane Potential 2")
plt.plot(membrane_potentials[:, 0], c = '#1f77b4', alpha=0.3, label = "Membrane Potential 1")
plt.plot(membrane_potentials[:, 1], c = 'orange', alpha=0.3, label = "Membrane Potential 2")
plt.title("Predicted Membrane Potentials vs Synthetic Membrane Potentials")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()

# Show all plots
plt.show()