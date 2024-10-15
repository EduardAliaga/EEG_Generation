import sys
sys.path.insert(0, '../../')
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
    # stimulus_time = np.linspace(0, total_time, n_time_points)
    # stimuli = np.zeros_like(stimulus_time)
    # j = 0
    # sq = 0
    # for i_stimulus, stimulus_ in enumerate(stimulus_time):
    #     # stimuli[i_stimulus] = ((stimulus_  // period_square) % 2)
    #     if ((stimulus_  // period_square) % 2) == 1:
    #         if int(j/30)%10 == 0:
    #             stimuli[i_stimulus] = ((stimulus_  // period_square) % 2)
    #         else:
    #             stimuli[i_stimulus] = ((stimulus_  // period_square) % 2) *0.5
    #         j += 1
    # return stimuli
    stimulus_time = np.linspace(0, 300, n_time_points)
    period_square = 30
    stimuli = np.zeros_like(stimulus_time)
    j = 0
    sq = 0
    stimuli = np.zeros(n_time_points)
    r = rnd.randint(0,9)
    for i_stimulus in range(0, n_time_points, period_square):
        if (i_stimulus // period_square) % 2:
            if j == r:
                stimuli[i_stimulus:i_stimulus+period_square] = np.ones(period_square)
            else:
                stimuli[i_stimulus:i_stimulus+period_square] = np.ones(period_square)*0.5
            j += 1
            if j==10:
                j = 0
                r = rnd.randint(0,9)
    return stimuli

def generate_states_for_linear_and_sigmoid_case(stimuli, model, state_dim, aug_state_dim, sources, dt, theta, tau, M, W, H):

    n_time_points = len(stimuli)
    states = np.zeros((n_time_points, 9))

    for t in range(1, n_time_points):
        x = states[t-1]
        if model =='linear':
            states[t] = x + dt * (-x / tau + W @ x + M * stimuli[t-1])
        elif model == 'sigmoid':
            states[t] = x + dt * (-x / tau + W @ sigmoid(x, theta) + M * stimuli[t-1])
    return states

def generate_states_for_dcm_case(stimuli, state_dim, aug_state_dim, sources, H, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b):

    n_time_points = stimuli.shape[0]
    states = np.ones((aug_state_dim, sources, n_time_points))
    for t in range(1, n_time_points):
        x = states[:, :, t-1]
        states[:, :, t] = sf_dcm.f_o2(x, stimuli[t-1], dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b)
        # for source in range(0, sources):
        #     states[state_dim - source, :, t] = H[source]
    return states

def generate_measurements_dcm_case(states, H, state_dim, aug_state_dim, sources, noise = 1e-1, noise_seed=2002):
    n_stimuli = states.shape[2]
    measurements = np.zeros((n_stimuli, sources))
    x0 = states[1, :, 0] - states[2, :, 0]
    measurements[0] = H @ x0
    for t in range(1, n_stimuli):
        x0 = states[2, :, t] - states[3, :, t]
        measurements[t] = H @ x0

    rng = rnd.default_rng(noise_seed)
    measurements_noisy = measurements + rng.multivariate_normal(mean=np.zeros(sources), cov=np.eye(sources) * noise, size=n_stimuli)
    
    return measurements, measurements_noisy

def generate_measurements_linear_and_sigmoid_cases(states, H, state_dim, aug_state_dim, sources, noise = 1e-2, noise_seed=2002):

    n_time_points = states.shape[0]
    measurements = np.zeros((n_time_points, 3))
    measurements[0] = H @ states[0][0:state_dim]
    for t in range(1, n_time_points):
        measurements[t] = H @ states[t][0:state_dim]

    rng = rnd.default_rng(noise_seed)
    measurements_noisy = measurements + rng.multivariate_normal(mean=np.zeros(3), cov=np.eye(3) * noise, size = n_time_points)
    
    return measurements, measurements_noisy

def generate_synthetic_data(period_square, total_time, n_time_points, params_x, params_y, model):

    stimuli = generate_stimuli(period_square, total_time, n_time_points)
    if model == 'linear' or model == 'sigmoid':
        states = generate_states_for_linear_and_sigmoid_case(stimuli, model, **params_x)
        measurements, measurements_noisy = generate_measurements_linear_and_sigmoid_cases(states, **params_y)

    elif model == 'dcm':
        stimuli = np.array([stimuli, np.zeros(n_time_points), np.zeros(n_time_points)]).T
        states = generate_states_for_dcm_case(stimuli, **params_x)
        measurements, measurements_noisy = generate_measurements_dcm_case(states, **params_y)
    
    np.save(f'/Users/aliag/Desktop/EEG_Generation/data/synthetic_data/synthetic_data_{model}.npy', {
        'stimuli': stimuli,
        'states': states,
        'measurements': measurements,
        'measurements_noisy': measurements_noisy,
        'params': params_x,
    })

