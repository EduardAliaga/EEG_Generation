import numpy as np
import jax

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
        f_x_h_0(x),
        f_x_h_1(x)
    ])
