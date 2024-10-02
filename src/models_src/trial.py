import sys
sys.path.insert(0, '../')
import numpy.random as rnd
import jax
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize
# Define sigmoid

def objective(m_vec):
    M = m_vec.reshape((2, 2))
    return np.linalg.norm(M @ x - y)**2

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

def f_o(x, u, W, M, tau, dt):

    return x + dt * (-x / tau + W @ x + M * u)

def g_o(param):
    return param

def recursive_update (x, theta, W, M, H, tau, u, y, dt, P_x_, P_x, Q_x, P_params_, P_params, Q_params,R_y, num_iterations):
    W_init = W
    M_init = M
    tau_init = tau
    init_P_x = P_x
    init_P_x_ = P_x_
    init_P_params = P_params
    init_P_params_ = P_params_
    jacobian_F = jax.jit(jax.jacobian(f_o,argnums=(0)))
    F_W = jax.jit(jax.jacobian(f_o,argnums=(2)))
    F_M = jax.jit(jax.jacobian(f_o,argnums=(3)))
    F_tau = jax.jit(jax.jacobian(f_o,argnums=(4)))
    params = np.hstack((W.flatten(), M, tau))
    membrane_potentials_dif_it = np.zeros((10,3001,2))
    W_dif_it = np.zeros((10,2,2))
    
    c = 0
    for i in tqdm(range(20)):
        membrane_potentials_predicted = []

        f = 'linear'
        dim_latent = len(x)
        membrane_potentials_predicted.append(np.zeros(2))
        for t in tqdm(range(num_iterations)):
            W = params[:dim_latent**2].reshape((dim_latent, dim_latent))
            # W[0,0] = 0
            # W[1,1] = 0
            M = params[dim_latent**2]
            tau = params[dim_latent**2 + 1]
            # tau = np.clip(tau, 95, 105)
            # M = np.clip(M, 95, 105)

            F_x = jacobian_F(x, u[t-1], W, M, tau, dt).reshape(2,2)
            F_params = np.hstack([F_W(x, u[t-1], W, M, tau, dt).reshape(2,4), F_M(x, u[t-1], W, M, tau, dt).reshape(2,1), F_tau(x, u[t-1], W, M, tau, dt,).reshape(2,1)]).reshape(2,6)
            x_hat = f_o(x, u[t-1], W, M, tau, dt)    
            y_hat = H @ x_hat

            #H = np.linalg.inv(x_pred_array.T @ x_pred_array) @ x_pred_array.T @ np.array([y_hat])
            #With a fixed and known H, the algorithm works fine (the parameters obtained are not the same but the result is)
            
            S = H @ P_x_ @ H.T + R_y 
            S_inv = np.linalg.inv(S)

            x = x_hat - P_x_ @ H.T @ S_inv @ (y_hat - y[t])
            # x = np.clip(x, -1000, 1150)
            I = np.eye(2)
            P_x_ = F_x @ P_x @ F_x.T + Q_x
            P_x = P_x_ @ (I + H @ (R_y - H @ P_x_ @ H.T) @ H @ P_x_)

            params = params - P_params_ @ F_params.T @ (x_hat - x) 
            
            P_params_ =  P_params - P_params @ F_params.T @ (Q_x + F_params @ P_params @ F_params.T) @ F_params @ P_params
            P_params = P_params_ + Q_params

            # if np.any(np.isnan(x)):
            if np.any(np.isnan(x)):
                x = np.zeros(2)
                # params = np.hstack((W_init.flatten(), M_init, tau_init)) 
                params = np.random.rand(6)
                P_x = init_P_x
                P_x_ = init_P_x_
                P_params = init_P_params
                P_params_ = init_P_params_
                break
            membrane_potentials_predicted.append(x)

                # membrane_potentials_dif_it.append(np.array(membrane_potentials_predicted))
            
        def objective(H_vec):
            H_matrix = H_vec.reshape((2, 2))
            return np.linalg.norm(membrane_potentials_predicted[:t] @ H_matrix.T - np.array(y[:t]))**2

        # Define the constraint that H must be non-negative
        def non_negative_constraint(H_vec):
            return np.abs(H_vec)

        def constrained_function(H_mat):
            # n_dims = len(H_mat)
            # for i0 in range(n_dims):
            #     for i1 in range(n_dims):
            #         if i0 != i1:
            #             H_mat[i0, i1] = 0
            H_mat[1] = 0
            H_mat[2] = 0

            return H_mat

        # Initial guess for H
        H_initial = H.flatten()

        # Optimization with non-negativity constraint
        constraints = {'type': 'ineq', 'fun': non_negative_constraint}
        result = minimize(objective, H_initial, constraints=constraints)

        # Update H with the optimized result
        H = result.x.reshape((2, 2))

        if i % 10 == 0 and len(membrane_potentials_predicted) == 3001:
            membrane_potentials_dif_it[c] = np.array(membrane_potentials_predicted)
            W_dif_it[c] = W
            c+=1
            x = np.zeros(2)
            #params = np.random.rand(6)
            P_x = init_P_x
            P_x_ = init_P_x_
            P_params = init_P_params
            P_params_ = init_P_params_
            F_x = np.zeros((2,2))
            F_params = np.zeros((2,6))
            H = H_init +np.random.rand(2,2)
    membrane_potentials_predicted = np.array(membrane_potentials_predicted)
        # H = np.linalg.inv(membrane_potentials_predicted[:t].T @ membrane_potentials_predicted[:t]) @ membrane_potentials_predicted[:t].T @ np.array(y[:t])
        # H = np.clip(H, 1e-2, 2)
        # H = H.reshape(2,2)
    return x, membrane_potentials_dif_it, W_dif_it, M, H, tau, P_x_, P_x, P_params_, P_params


n_stimuli = 3000
# Create synthetic stimuli
period_square = 10
stimuli = np.zeros(n_stimuli)
j = 0
for i_stimulus in range(0, n_stimuli, period_square):
    if (i_stimulus // period_square) % 2:
            if j % 10 == 0:
                stimuli[i_stimulus: i_stimulus + period_square] = np.ones(period_square)
            else:
                stimuli[i_stimulus: i_stimulus + period_square] = np.ones(period_square) * 0.5
            j+=1

tau = 1e2
dt = 1e-1
theta = 1.0
W = np.zeros((2,2))
W[0,1] = 1e-1
W[1,0] = -1e-1
# M = np.zeros((2,2))
M = 100

membrane_potentials = np.zeros((n_stimuli, 2))
membrane_potentials[0] = np.array([0, 0])
H = np.array([[1, 0.8], [0.5, 0.4]])
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
Q_x = np.eye(2) * 1e-6
R_y = np.eye(2) * 1e-6
P_x_ = np.eye(2) * 1e-6
P_x = np.eye(2) * 1e-6
P_params_ = np.eye(6) * 1e-6
P_params = np.eye(6) * 1e-6
# P_params[5,5] = 5e-3
# P_params[4,4] = 5e-3
Q_params = np.eye(6) * 1e-6
# Q_params[5,5] = 1e-4
# Q_params[4,4] = 1e-4
W_init = np.array([[0,0.2],[-0.5, 0]])
M_init = 90
H_init = np.array([[.9, 0], [0, .36]])
tau_init = 90

membrane_potentials_predicted = []
x, membrane_potentials_n, W, M, H, tau, P_x_, P_x, P_params_, P_params = recursive_update(
    x, theta, W_init, M_init, H_init, tau_init, stimuli, measurements_noisy, dt, P_x_, P_x, Q_x, P_params_, P_params, Q_params, R_y, n_stimuli)

# membrane_potentials_predicted.append(membrane_potentials_n)
#  membrane_potentials_predicted = np.array(membrane_potentials_predicted)

# print(f"Membrane potentials shape: {membrane_potentials_predicted.shape}")

y = []
measurements_predicted_n = []
for prediction in membrane_potentials_n:
    for t in range(0,3000):
        y.append(H @ prediction[0][t])
    y = np.array(y)
    measurements_predicted_n.append(y)
    
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