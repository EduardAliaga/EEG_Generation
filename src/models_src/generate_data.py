import numpy as np
import numpy.random as rnd
import state_functions_dcm as sf_dcm
import matplotlib.pyplot as plt
from utils import sigmoid


def generate_stimuli(period_square, total_time, n_time_points):
    """
    Generate a square wave stimuli.

    Parameters:
    n_stimuli (int): Number of stimuli.
    period_square (int): Period of the square wave stimuli.

    Returns:
    array: Generated stimuli.
    """
    stimulus_time = np.linspace(0, total_time, n_time_points)
    stimuli = np.zeros_like(stimulus_time)
    for i_stimulus, stimulus_ in enumerate(stimulus_time):
        stimuli[i_stimulus] = (stimulus_  // period_square) % 2

    return stimuli

def generate_states_for_linear_and_sigmoid_case(stimuli, model, state_dim, aug_state_dim, theta, tau, dt, M, W, H):

    n_time_points = len(stimuli)
    states = np.zeros((n_time_points, aug_state_dim))
    states[0] = np.array([-0.07, -0.07, 0, 0, 0, 0])
    states[0][state_dim:aug_state_dim] = H.flatten()
    
    for t in range(1, n_time_points):
        x = states[t-1]
        if model =='linear':
            states[t] = x + dt * (-x / tau + W @ x + M * stimuli[t-1])
        elif model == 'sigmoid':
            states[t] = x + dt * (-x / tau + W @ sigmoid(x, theta) + M * stimuli[t-1])
        states[t][state_dim:aug_state_dim] = H.flatten()
    return states

def generate_states_for_dcm_case(stimuli, state_dim, aug_state_dim, sources, H, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b):

    n_time_points = stimuli.shape[0]
    states = np.zeros((aug_state_dim, sources, n_time_points))
    for source in range(0, sources):
        states[state_dim - source, :, 0] = H[source]
    
    for t in range(1, n_time_points):
        x = states[:, :, t-1]
        states[:, :, t] = sf_dcm.f_o(x, stimuli[t-1], dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b)
        for source in range(0, sources):
            states[state_dim - source, :, t] = H[source]
    return states

def generate_measurements_dcm_case(states, H, state_dim, aug_state_dim, sources, noise = 1e-4, noise_seed=2002):
    n_stimuli = states.shape[2]
    measurements = np.zeros((n_stimuli, sources))
    x0 = states[1, :, 0] - states[2, :, 0]
    measurements[0] = H @ x0
    states[state_dim:aug_state_dim, :, 0] = H
    for t in range(1, n_stimuli):
        x0 = states[2, :, t] - states[3, :, t]
        measurements[t] = H @ x0

    rng = rnd.default_rng(noise_seed)
    measurements_noisy = measurements + rng.multivariate_normal(mean=np.zeros(2), cov=np.eye(2) * noise, size=n_stimuli)
    
    return measurements, measurements_noisy

def generate_measurements_linear_and_sigmoid_cases(states, H, state_dim, aug_state_dim, noise = 1e-4, noise_seed=2002):

    n_time_points = states.shape[1]
    measurements = np.zeros((n_time_points, 2))
    measurements[0] = H @ states[0][0:state_dim]
    states[0][state_dim:aug_state_dim] = H.flatten()
    for t in range(1, n_time_points):
        measurements[t] = H @ states[t][0:state_dim]

    rng = rnd.default_rng(noise_seed)
    measurements_noisy = measurements + rng.multivariate_normal(mean=np.zeros(2), cov=np.eye(2) * noise, size = n_time_points)
    
    return measurements, measurements_noisy

def generate_synthetic_data(period_square, total_time, n_time_points, params_x, params_y, model):

    stimuli = generate_stimuli(period_square, total_time, n_time_points)
    if model == 'linear' or model == 'sigmoid':
        states = generate_states_for_linear_and_sigmoid_case(stimuli, model, **params_x)
        generate_measurements_linear_and_sigmoid_cases(states, **params_y)

    elif model == 'dcm':
        stimuli = np.array([stimuli, stimuli]).T
        states = generate_states_for_dcm_case(stimuli, **params_x)
        measurements, measurements_noisy = generate_measurements_dcm_case(states, **params_y)
    
    np.save('synthetic_data2.npy', {
        'stimuli': stimuli,
        'states': states,
        'measurements': measurements,
        'measurements_noisy': measurements_noisy,
        'params': params_x,
    })

if __name__ == "__main__":
    total_time = 30
    dt = 1e-2
    n_time_points = int(total_time / dt)
    period_square = 0.3
    sources = 2
    model = 'dcm'
    #params for dcm model: state_dim, aug_state_dim, sources, H, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b
    params_x = {
        'state_dim': 9,
        'aug_state_dim': 11,
        'sources' : 2,
        'H': np.array([[1, 0.7], [0.5, 0.8]]),
        'dt': dt,
        'theta': 0.56,
        'H_e': 0.1,
        'tau_e': 10.0,
        'H_i': 16.0,
        'tau_i': 32.0,
        'gamma_1': 1.0,
        'gamma_2': 4/5,
        'gamma_3': 1/4,
        'gamma_4': 1/4,  # gamma_3 value
        'sources': 2,
        'C_f': np.random.rand(sources, sources),
        'C_l': np.random.rand(sources, sources), 
        'C_u': np.random.rand(sources),
        'C_b': np.random.rand(sources, sources), 
    }
    params_y = {
        'H': params_x['H'],
        'state_dim': params_x['state_dim'],
        'aug_state_dim': params_x['aug_state_dim'],
        'sources': params_x['sources']
    }
    generate_synthetic_data(period_square, total_time, n_time_points, params_x, params_y, model)
