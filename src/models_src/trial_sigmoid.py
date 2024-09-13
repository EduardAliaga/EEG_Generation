import sys
sys.path.insert(0, '../')
import numpy.random as rnd
import jax
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# Define sigmoid
def sigmoid(x, theta):
     
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

    return x + dt * (-x / tau + W @ jax.nn.sigmoid(x*0.56) + M * u)

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
    for i in tqdm(range(30)):
        membrane_potentials_predicted = []
        dim_latent = len(x)
        membrane_potentials_predicted.append(np.zeros(2))
        for t in tqdm(range(num_iterations)):
            W = params[:dim_latent**2].reshape((dim_latent, dim_latent))
            M = params[dim_latent**2]
            tau = params[dim_latent**2 + 1]
            tau = np.clip(tau, 80, 150)
            # M = np.clip(M, 80, 150)

            F_x = jacobian_F(x, u[t-1], W, M, tau, dt).reshape(2,2)
            F_params = np.hstack([F_W(x, u[t-1], W, M, tau, dt).reshape(2,4), F_M(x, u[t-1], W, M, tau, dt).reshape(2,1), F_tau(x, u[t-1], W, M, tau, dt,).reshape(2,1)]).reshape(2,6)
            x_hat = f_o(x, u[t-1], W, M, tau, dt)    
            y_hat = H @ x_hat

            #H = np.linalg.inv(x_pred_array.T @ x_pred_array) @ x_pred_array.T @ np.array([y_hat])
            #With a fixed and known H, the algorithm works fine (the parameters obtained are not the same but the result is)

            S = H @ P_x_ @ H.T + R_y 
            S_inv = np.linalg.inv(S)

            x = x_hat - P_x_ @ H.T @ S_inv @ (y_hat - y[t])
            I = np.eye(2)
            P_x_ = F_x @ P_x @ F_x.T + Q_x
            P_x = P_x_ @ (I + H @ (R_y - H @ P_x_ @ H.T) @ H @ P_x_)

            params = params - P_params_ @ F_params.T @ (x_hat - x) 
            
            P_params_ =  P_params - P_params @ F_params.T @ (Q_x + F_params @ P_params @ F_params.T) @ F_params @ P_params
            P_params = P_params_ + Q_params
            # Q_x = dt * Q_x
            # Q_params = dt * Q_params
            if np.any(np.isnan(x)):
                x = np.zeros(2)
                params = np.hstack((W_init.flatten(), M_init, tau_init))
                P_x = init_P_x
                P_x_ = init_P_x_
                P_params = init_P_params
                P_params_ = init_P_params_
                F_x = np.zeros((2,2))
                F_params = np.zeros((2,6))
                break
            membrane_potentials_predicted.append(x)
        membrane_potentials_predicted = np.array(membrane_potentials_predicted)
        # H = np.linalg.inv(membrane_potentials_predicted[:t].T @ membrane_potentials_predicted[:t]) @ membrane_potentials_predicted[:t].T @ np.array(y[:t])
        # H = np.clip(H, 1e-2, 2)
        # H = H.reshape(2,2)
    return x, np.array(membrane_potentials_predicted), W, M, H, tau, P_x_, P_x, P_params_, P_params


n_stimuli = 3000
# Create synthetic stimuli
period_square = 10
stimuli = np.zeros(n_stimuli)
for i_stimulus in range(0, n_stimuli, period_square):
    if (i_stimulus // period_square) % 2:
        stimuli[i_stimulus: i_stimulus + period_square] = np.ones(period_square)

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
H = np.array([[1, 1], [0.5, 0.4]])
measurements = np.zeros((n_stimuli, 2))
measurements[0] = H @ membrane_potentials[0]
# Generate membrane potentials
for t in range(1, n_stimuli):
    x = membrane_potentials[t-1]
    membrane_potentials[t] = x + dt * (-x / tau + W @ jax.nn.sigmoid(x) + M * stimuli[t-1])
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
P_params[5,5] = 5e-3
P_params[4,4] = 5e-3
Q_params = np.eye(6) * 1e-6
Q_params[4,4] = 1e-5
Q_params[5,5] = 1e-3
W_init = np.array([[0,0.2],[-0.5, 0]])
M_init = 90
H_init = np.array([[1.4, 1], [0.4, 0.6]])
tau_init = 90
x = np.zeros(2)
membrane_potentials_predicted = []
x, membrane_potentials_n, W, M, H, tau, P_x_, P_x, P_params_, P_params = recursive_update(
    x, theta, W_init, M_init, H, tau_init, stimuli, measurements_noisy, dt, P_x_, P_x, Q_x, P_params_, P_params, Q_params, R_y, n_stimuli)

membrane_potentials_predicted.append(membrane_potentials_n)
membrane_potentials_predicted = np.array(membrane_potentials_predicted)

print(f"Membrane potentials shape: {membrane_potentials_predicted.shape}")

y = []
for t in range(0,3000):
    y.append(H @ membrane_potentials_predicted[0][t])
y = np.array(y)
print(measurements.shape)