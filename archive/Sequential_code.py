import numpy as np
import jax
import matplotlib.pyplot as plt

def f_x0(x, dt):
     return x[0] + dt * (x[5] - x[6])

def f_x1(x, dt):
     return x[1] + dt * x[4]

def f_x2(x, dt):
     return x[2] + dt * x[5]

def f_x3(x, dt):
     return x[3] + dt * x[6]

def f_x4(x, u, dt, theta, H_e, tau_e, gamma_1, C_f, C_l, C_u):
     return x[4] + dt * (H_e/tau_e * ((C_f + C_l + gamma_1 * np.eye(2)) @ (jax.nn.sigmoid(x[0] * theta)-0.5) + C_u @ u) - 2*x[4]/tau_e - x[1]/tau_e**2)

def f_x5(x, dt, theta, H_e, tau_e, gamma_2, C_b, C_l):
     return x[5] + dt * (H_e/tau_e * ((C_b + C_l) @ (jax.nn.sigmoid(x[0] * theta) - 0.5) + gamma_2 * (jax.nn.sigmoid(x[3] * theta) - 0.5)) * 2 * x[5]/tau_e - x[2]/tau_e**2)

def f_x6(x, dt, theta, H_i, tau_i, gamma_4):
     return x[6] + dt * (H_i/tau_i * gamma_4 * (jax.nn.sigmoid(x[7] * theta) - 0.5) - 2 * x[6]/tau_i - x[3]/tau_i**2)

def f_x7(x, dt):
     return x[7] + dt * x[8]

def f_x8(x, dt, theta, H_e, tau_e, gamma_3, C_b, C_l):
    return x[8] + dt * (H_e/tau_e * ((C_b + C_l + gamma_3 * np.eye(len(x[0]))) @ (jax.nn.sigmoid(x[0] * theta) - 0.5)) - 2 * x[8]/tau_e - x[7]/tau_e**2)

def f_x_h_0(x):
     return x[9]

def f_x_h_1(x):
     return x[10]

def f_o(x, u, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b):
    return jax.numpy.array([
        f_x0(x, dt),
        f_x1(x, dt),
        f_x2(x, dt),
        f_x3(x, dt),
        f_x4(x, u, dt, theta, H_e, tau_e, gamma_1, C_f, C_l, C_u),
        f_x5(x, dt, theta, H_e, tau_e, gamma_2, C_b, C_l),
        f_x6(x, dt, theta, H_i, tau_i, gamma_4),
        f_x7(x, dt),
        f_x8(x, dt, theta, H_e, tau_e, gamma_3, C_b, C_l),
    ])

def jacobian_f_o(x, u, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b):
     #    F_theta = np.array(jax.jit(jax.jacobian(f_o, argnums=3))(x, u, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b)).reshape(18, 1)
     #    F_H_e = np.array(jax.jit(jax.jacobian(f_o, argnums=4))(x, u, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b)).reshape(18, 1)
     #    F_tau_e = np.array(jax.jit(jax.jacobian(f_o, argnums=5))(x, u, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b)).reshape(18, 1)
     #    F_H_i = np.array(jax.jit(jax.jacobian(f_o, argnums=6))(x, u, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b)).reshape(18, 1)
     #    F_tau_i = np.array(jax.jit(jax.jacobian(f_o, argnums=7))(x, u, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b)).reshape(18, 1)
     #    F_gamma_1 = np.array(jax.jit(jax.jacobian(f_o, argnums=8))(x, u, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b)).reshape(18, 1)
     #    F_gamma_2 = np.array(jax.jit(jax.jacobian(f_o, argnums=9))(x, u, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b)).reshape(18, 1)
     #    F_gamma_3 = np.array(jax.jit(jax.jacobian(f_o, argnums=10))(x, u, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b)).reshape(18, 1)
     #    F_gamma_4 = np.array(jax.jit(jax.jacobian(f_o, argnums=11))(x, u, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b)).reshape(18, 1)
     #    F_C_f = np.array(jax.jit(jax.jacobian(f_o, argnums=12))(x, u, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b)).reshape(18, 4)
     #    F_C_l = np.array(jax.jit(jax.jacobian(f_o, argnums=13))(x, u, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b)).reshape(18, 4)
        F_C_u = np.array(jax.jit(jax.jacobian(f_o, argnums=14))(x, u, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b)).reshape(18, 2)
     #    F_C_b = np.array(jax.jit(jax.jacobian(f_o, argnums=15))(x, u, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b)).reshape(18, 4)
     #    F_params = np.hstack([F_theta, F_H_e, F_tau_e, F_H_i, F_tau_i, F_gamma_1, F_gamma_2, F_gamma_3, F_gamma_4, F_C_f, F_C_l, F_C_u, F_C_b])
        return F_C_u

def measurement_function(x):
    return x[9:11] @ (x[2] - x[3])

data_file ='/Users/aliag/Desktop/EEG_Generation/synthetic_data.npy'
data_file_2 ='states_predicted.npy'
data_file_3 = 'measurements_predicted.npy' 
data_file_4 = 'params_vec.npy'
data_file_5 = 'H.npy'
#stimuli, states, measurements, measurements_noisy, real_params = load_synthetic_data(data_file, f)
data = np.load(data_file, allow_pickle=True).item()
# data_2 = np.load(data_file_2, allow_pickle = True)
# data_3 = np.load(data_file_3, allow_pickle = True)
# params_vec = np.load(data_file_4, allow_pickle = True)
# H = np.load(data_file_5, allow_pickle = True)

# print(params_vec)
# print(H)
stimuli = data['stimuli']
states = data['states']
states = np.array(states)
measurements = data['measurements']
measurements_noisy = data['measurements_noisy']
# print(np.linalg.norm(data_3 - measurements_noisy))
# print(np.linalg.norm(data_2[:,0,0] - states[0,0,:]))
# plt.figure()
# plt.plot(data_2[:,0,0], label = "prediction")
# plt.plot(states[0,0,:], label = "ground_truth")
# plt.legend()
# plt.figure()
# plt.plot(data_3, label = "prediction")
# plt.plot(measurements_noisy, label = "ground_truth")
# plt.legend()
# plt.show()

aug_state_dim = 9
aug_state_dim_flattened = 18
sources = 2
state_dim = 9
num_time_points = len(stimuli)
x = np.zeros((9,2))
dt = 1e-2
theta = 0.56
H_e = 0.1
tau_e = 10.0
H_i = 16.0
tau_i= 32.0
gamma_1= 1.0
gamma_2= 4/5
gamma_3= 1/4
gamma_4= 1/4  # gamma_3 value
sources= 2
C_f= np.eye(sources)
C_l= np.eye(sources)
C_u= np.array([1000.0,1000.0])
C_b = np.eye(sources)


n_params = 2
Q_x = np.eye(aug_state_dim_flattened) * 1e-4
R_y = np.eye(sources) * 1e-4
P_x_ = np.eye(aug_state_dim_flattened) * 1e-4
P_x = np.eye(aug_state_dim_flattened) * 1e-4
P_params_ = np.eye(n_params) * 1e-4
P_params = np.eye(n_params) * 1e-4
Q_params = np.eye(n_params) * 1e-4
params_vec = np.hstack((theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f.flatten(), C_l.flatten(), C_u.flatten(), C_b.flatten()))
params_vec_2 = C_u.reshape(2)
#states_predicted = np.zeros((num_time_points, aug_state_dim))
states_predicted = np.zeros((num_time_points, aug_state_dim, sources))
#measurements_predicted = np.zeros((num_time_points, state_dim))
measurements_predicted = np.zeros((num_time_points, sources))
# Set initial state
states_predicted[0] = x
H = np.array([[1, 0.7], [0.5, 0.8]])
dH = np.zeros((2,18))
dH[0,0] = H[0,0]
dH[0,1] = H[0,1]
dH[1,0] = H[1,0]
dH[1,1] = H[1,1]
for t in range(1, 3000):
    # TODO: define the jacobians as  also 
    #F_x = np.array(jax.jit(jax.jacobian(f_o, argnums = (0)))(x, stimuli[t-1], dt, params_vec[0], params_vec[1], params_vec[2], params_vec[3], params_vec[4], params_vec[5], params_vec[6], params_vec[7], params_vec[8], params_vec[9:13].reshape(2,2), params_vec[13:17].reshape(2,2), params_vec[17:19].reshape(2), params_vec[19:23].reshape(2,2))).reshape(18,18)
    F_x = np.array(jax.jit(jax.jacobian(f_o, argnums = (0)))(x, stimuli[t-1], dt, params_vec[0], params_vec[1], params_vec[2], params_vec[3], params_vec[4], params_vec[5], params_vec[6], params_vec[7], params_vec[8], params_vec[9:13].reshape(2,2), params_vec[13:17].reshape(2,2), params_vec_2, params_vec[19:23].reshape(2,2))).reshape(18,18)
    F_params = jacobian_f_o(x, stimuli[t-1], dt, params_vec[0], params_vec[1], params_vec[2], params_vec[3], params_vec[4], params_vec[5], params_vec[6], params_vec[7], params_vec[8], params_vec[9:13].reshape(2,2), params_vec[13:17].reshape(2,2), params_vec_2, params_vec[19:23].reshape(2,2))
    print(t)
    #H = .x[.state_dim:.aug_state_dim].reshape((.state_dim, .state_dim))
    y_hat = H @ x[0]
    S = dH @ P_x_ @ dH.T + R_y 
    S_inv = np.linalg.inv(S)
    x_hat = f_o(x, stimuli[t-1], dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b).flatten()
    x = x_hat - P_x_ @ dH.T @ S_inv @ (y_hat - measurements_noisy[t])

    I = np.eye(aug_state_dim_flattened)
    P_x_ = F_x @ P_x @ F_x.T + Q_x
    P_x = P_x_ @ (I + dH.T @ (R_y - dH @ P_x_ @ dH.T) @ dH @ P_x_)

    params_vec_2 = params_vec_2 - P_params_ @ F_params.T @ (x_hat - x)
    #print(params_vec)
    P_params_ = P_params - P_params @ F_params.T @ (Q_x + F_params @ P_params @ F_params.T) @ F_params @ P_params
    P_params = P_params_ + Q_params

    Q_x = dt * Q_x
    Q_params = dt * Q_params
    x = x.reshape(aug_state_dim, sources)

    # Assign predicted values
    states_predicted[t] = x
    measurements_predicted[t] = y_hat

print(states_predicted.shape)
np.save("states_predicted.npy", states_predicted)
np.save("measurements_predicted.npy", measurements_predicted)
np.save("params_vec.npy", params_vec)
np.save("H.npy", H)
