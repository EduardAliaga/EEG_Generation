from filterpy.common import Saver
from filterpy.kalman import CubatureKalmanFilter as CKF
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import numpy as np
from numpy.random import randn
from pytest import approx
from scipy.spatial.distance import mahalanobis as scipy_mahalanobisz
from tqdm import tqdm
import matplotlib.pyplot as plt

def f_o(x, u, dt, W, M, tau):
    return x + dt * (-x / tau + W @ x + M * u)

def fz(z, dt, u):
    state_dim = 2
    x = z[0:state_dim].reshape(2,1)
    W = z[state_dim:state_dim+4].reshape(2,2)
    M = z[state_dim+4]
    tau = z[state_dim+5]
    x_new =  f_o(x[:,0], u, dt, W, M, tau)
    params = np.hstack([W.flatten(), M, tau])
    #params = np.clip(params, 1e-2, 35)
    z_new = np.hstack([x_new.flatten(), params.flatten()])
    return z_new

def hz(z):
    H = np.array([[1, 1], [0.5, 0.4]])
    return H @ z[0:2]

def params_dict_to_vector(params_dict):
    params_vec = []
    for key, value in params_dict.items():
        if isinstance(value, np.ndarray):
            params_vec.extend(value.flatten())
        else:
            params_vec.append(value)
    return np.array(params_vec).reshape(-1,1)
tau = 50.0
M = 70.0
theta = 1.0
W = np.zeros((2,2))
W[0,1] = 5e-1
W[1,0] = -5e-1

init_params = {
    'W': W,
    'M': M,
    'tau' : tau,
}

params_vec = params_dict_to_vector(init_params)

state_dim = 2
state_params_dim = 8

ckf = CKF(dim_x=state_params_dim, dim_z=2, dt=1e-1, hx=hz, fx=fz)
x_init = np.hstack([np.zeros(state_dim),params_vec[:,0]]).reshape(8,1)
ckf.x = x_init
ckf.P = np.eye(state_params_dim) * 1e-4
#ckf.P[2,2] = 1e-1
ckf.P[3,3] = 1e-4
#ckf.P[4,4] = 1e-1
ckf.P[5,5] = 1e-4
ckf.P[6,6] = 5
ckf.P[7,7] = 5

ckf.R = np.eye(2) * 1e-4
ckf.Q = np.eye(state_params_dim) * 1e-4
# ckf.Q[2,2] = 1e-1
# ckf.Q[3,3] = 1e-2
# ckf.Q[4,4] = 1
# ckf.Q[5,5] = 1e-2
ckf.Q[6,6] = 5
ckf.Q[7,7] = 5
data_file='/Users/aliag/Desktop/EEG_Generation/data/synthetic_data/synthetic_data_linear.npy'
data = np.load(data_file, allow_pickle=True).item()
stimuli = data['stimuli']
states = data['states']
states = np.array(states)
measurements = data['measurements']
measurements_noisy = data['measurements_noisy']

states_predicted = []
t = 0
for i in tqdm(range(3000)):
    ckf.predict(fx_args = (stimuli[i-1]))
    ckf.update(measurements[i-1].reshape(-1,1))
    # test mahalanobis
    states_predicted.append(ckf.x)
states_predicted = np.array(states_predicted)
y = []
for i in range(len(states_predicted)):
    y.append(hz(states_predicted[i]))
y = np.array(y)

print('hello')
