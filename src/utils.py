import numpy as np

def sigma(x, theta):
    return 1 / (1 + np.exp(-theta * x))

def membrane_potential_equation(x, t, tau, w, theta, u):
    dxdt = np.zeros_like(x)
    N = len(x)
    time_index = min(int(t), len(u) - 1)  # Ensure index is within bounds
    for i in range(N):
        dxdt[i] = -x[i] / tau + np.sum(w[i, :] * sigma(x, theta)) + u[time_index]
    return dxdt

def eeg_linear_model_computation(x, matrix):
    return x @ matrix

