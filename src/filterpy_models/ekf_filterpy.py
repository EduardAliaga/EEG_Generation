from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from tqdm import tqdm

import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from numpy import array, eye, asarray

from filterpy.common import Saver
from filterpy.examples import RadarSim
from pytest import approx
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis

DO_PLOT = False


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def f_x0(x, dt):
     return x[0] + dt * (x[5] - x[6])

def f_x1(x, dt):
     return x[1] + dt * x[4]

def f_x2(x, dt):
     return x[2] + dt * x[5]

def f_x3(x, dt):
     return x[3] + dt * x[6]

def f_x4(x, u, dt, theta, H_e, tau_e, gamma_1, C_f, C_l, C_u):
     return x[4] + dt * (H_e/tau_e * ((C_f + C_l + gamma_1 * np.eye(2)) @ (sigmoid(x[0] * theta)-0.5) + C_u @ u) - 2*x[4]/tau_e - x[1]/tau_e**2)

def f_x5(x, dt, theta, H_e, tau_e, gamma_2, C_b, C_l):
     return x[5] + dt * (H_e/tau_e * ((C_b + C_l) @ (sigmoid(x[0] * theta) - 0.5) + gamma_2 * (sigmoid(x[3] * theta) - 0.5)) * 2 * x[5]/tau_e - x[2]/tau_e**2)

def f_x6(x, dt, theta, H_i, tau_i, gamma_4):
     return x[6] + dt * (H_i/tau_i * gamma_4 * (sigmoid(x[7] * theta) - 0.5) - 2 * x[6]/tau_i - x[3]/tau_i**2)

def f_x7(x, dt):
     return x[7] + dt * x[8]

def f_x8(x, dt, theta, H_e, tau_e, gamma_3, C_b, C_l):
    return x[8] + dt * (H_e/tau_e * ((C_b + C_l + gamma_3 * np.eye(len(x[0]))) @ (sigmoid(x[0] * theta) - 0.5)) - 2 * x[8]/tau_e - x[7]/tau_e**2)

def fz(z, dt, u):
    state_dim = 18
    theta = z[state_dim]
    H_e = z[state_dim+1]
    tau_e = z[state_dim+2]
    H_i = z[state_dim+3]
    tau_i = z[state_dim+4]
    gamma_1 = z[state_dim+5]
    gamma_2 = z[state_dim+6]
    gamma_3 = z[state_dim+7]
    gamma_4 = z[state_dim+8]
    C_f = z[state_dim+9:state_dim+13].reshape(2,2)
    C_l = z[state_dim+13:state_dim+17].reshape(2,2)
    C_u = z[state_dim+17:state_dim+19]
    C_b = z[state_dim+19: state_dim+23].reshape(2,2)
    x = z[0:state_dim].reshape(9,2)
    x_new =  np.array([
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
    params = np.hstack([theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f.flatten(), C_l.flatten(), C_u, C_b.flatten()])
    #params = np.clip(params, 1e-2, 35)
    z_new = np.hstack([x_new.flatten(), params.flatten()])
    return z_new

def hz(z):
    t_H = np.array([[1, 0.7], [0.5, 0.8]])
    return t_H @ z[0:2]

def params_dict_to_vector(params_dict):
    params_vec = []
    for key, value in params_dict.items():
        if isinstance(value, np.ndarray):
            params_vec.extend(value.flatten())
        else:
            params_vec.append(value)
    return np.array(params_vec).reshape(-1,1)

real_params = {'theta': 0.56, 'H_e': 0.1, 'tau_e': 10.0, 'H_i': 16.0, 'tau_i': 32.0, 'gamma_1': 1.0, 'gamma_2': 0.8, 'gamma_3': 0.25, 'gamma_4': 0.25, 'C_f': np.array([[1., 0.],
       [0., 1.]]), 'C_l': np.array([[1., 0.],
       [0., 1.]]), 'C_u': np.array([1., 1.]), 'C_b': np.array([[1., 0.],
       [0., 1.]])}

params_vec = params_dict_to_vector(real_params)

def H_of(x):
    H = np.zeros((2,41))
    H[:,0:2] = np.array([[1, 0.7], [0.5, 0.8]])
    return H

dt = 0.05
proccess_error = 0.05

rk = ExtendedKalmanFilter(dim_x=41, dim_z=2)
state_params_dim = 41
x_init = np.hstack([np.zeros(18),params_vec[:,0]]).reshape(41,1)
rk.x = x_init + np.random.randn(len(x_init),1)
rk.x[20] = 7
rk.x[21] = 14
rk.x[22] = 30
rk.P = np.eye(state_params_dim) * 100
rk.P[20,20] = 30
rk.P[21,21] = 40
rk.P[22,22] = 30

rk.R = np.eye(2) * 10
rk.Q = np.eye(state_params_dim) * 100
rk.Q[20,20] = 30
rk.Q[21,21] = 40
rk.Q[22,22] = 30
xs = []
radar = RadarSim(dt)
ps = []
pos = []

data_file='/Users/aliag/Desktop/EEG_Generation/src/models_src/synthetic_data.npy'
data = np.load(data_file, allow_pickle=True).item()
stimuli = data['stimuli']
states = data['states']
states = np.array(states)
measurements = data['measurements']
measurements_noisy = data['measurements_noisy']
dt = 1e-2
states_predicted = []
t = 0

for i in tqdm(range(3000)):
    rk.update(measurements_noisy[i-1].reshape(-1,1), H_of, hz, R=rk.R)
    ps.append(rk.P)
    rk.predict()
    xs.append(rk.x)
    states_predicted.append(rk.x)
states_predicted = np.array(states_predicted)
print('hello')
    # test mahalanobis





