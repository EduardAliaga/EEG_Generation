import sys
sys.path.insert(0, '../../')
import numpy as np
import numpy.random as rnd
import src.models_src.state_functions_dcm as sf_dcm
import matplotlib.pyplot as plt
from src.models_src.utils import sigmoid

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
        stimuli[i_stimulus] = ((stimulus_  // period_square) % 2)

    return stimuli

def generate_states_for_linear_and_sigmoid_case(stimuli, model, state_dim, aug_state_dim, sources, dt, theta, tau, M, W, H):

    n_time_points = len(stimuli)
    states = np.zeros((n_time_points, aug_state_dim))
    states[0] = np.zeros(2)
    for t in range(1, n_time_points):
        x = states[t-1]
        if model =='linear':
            states[t] = x + dt * (-x / tau + W @ x + M * stimuli[t-1])
        elif model == 'sigmoid':
            states[t] = x + dt * (-x / tau + W @ sigmoid(x, theta) + M * stimuli[t-1])
    return states

def generate_states_for_dcm_case(stimuli, state_dim, aug_state_dim, sources, H, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b):

    n_time_points = stimuli.shape[0]
    states = np.zeros((aug_state_dim, sources, n_time_points))
    for source in range(0, sources):
        states[state_dim - source, :, 0] = H[source]
    
    for t in range(1, n_time_points):
        x = states[:, :, t-1]
        states[:, :, t] = sf_dcm.f_o2(x, stimuli[t-1], dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b)
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

def generate_measurements_linear_and_sigmoid_cases(states, H, state_dim, aug_state_dim, sources, noise = 1e-4, noise_seed=2002):

    n_time_points = states.shape[0]
    measurements = np.zeros((n_time_points, 2))
    measurements[0] = H @ states[0][0:state_dim]
    for t in range(1, n_time_points):
        measurements[t] = H @ states[t][0:state_dim]

    rng = rnd.default_rng(noise_seed)
    measurements_noisy = measurements + rng.multivariate_normal(mean=np.zeros(2), cov=np.eye(2) * noise, size = n_time_points)
    
    return measurements, measurements_noisy

def generate_synthetic_data(period_square, total_time, n_time_points, params_x, params_y, model):

    stimuli = generate_stimuli(period_square, total_time, n_time_points)
    if model == 'linear' or model == 'sigmoid':
        states = generate_states_for_linear_and_sigmoid_case(stimuli, model, **params_x)
        measurements, measurements_noisy = generate_measurements_linear_and_sigmoid_cases(states, **params_y)

    elif model == 'dcm':
        stimuli = np.array([stimuli, stimuli]).T
        states = generate_states_for_dcm_case(stimuli, **params_x)
        measurements, measurements_noisy = generate_measurements_dcm_case(states, **params_y)
    
    np.save(f'/Users/aliag/Desktop/EEG_Generation/data/synthetic_data/synthetic_data_{model}.npy', {
        'stimuli': stimuli,
        'states': states,
        'measurements': measurements,
        'measurements_noisy': measurements_noisy,
        'params': params_x,
    })

