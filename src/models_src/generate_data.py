import numpy as np
import numpy.random as rnd
from utils import sigmoid

def generate_stimuli(n_stimuli, period_square):
    """
    Generate a square wave stimuli.

    Parameters:
    n_stimuli (int): Number of stimuli.
    period_square (int): Period of the square wave stimuli.

    Returns:
    array: Generated stimuli.
    """
    stimuli = np.zeros(n_stimuli)
    for i_stimulus in range(0, n_stimuli, period_square):
        if (i_stimulus // period_square) % 2:
            stimuli[i_stimulus: i_stimulus + period_square] = np.ones(period_square)
    return stimuli

def generate_membrane_potentials(stimuli, tau, dt, M, W, theta, f):
    """
    Generate membrane potentials based on stimuli.

    Parameters:
    stimuli (array): Input stimuli.
    tau (float): Time constant.
    dt (float): Time step.
    M (float): Input scaling factor.
    W (array): Weight matrix.

    Returns:
    array: Generated membrane potentials.
    """
    n_stimuli = len(stimuli)
    membrane_potentials = np.zeros((n_stimuli, 2))
    membrane_potentials[0] = np.array([-70, -70])
    
    for t in range(1, n_stimuli):
        x = membrane_potentials[t-1]
        if f == 'sigmoid':
            membrane_potentials[t] = x + dt * (-x / tau + W @ sigmoid(x, theta) + M * stimuli[t-1])
        elif f == 'tanh':
            membrane_potentials[t] = x + dt * (-x / tau + W @ np.tanh(x) + M * stimuli[t-1])
        elif f == 'linear':
            membrane_potentials[t] = x + dt * (-x / tau + W @ x + M * stimuli[t-1])
    
    return membrane_potentials

def generate_measurements(membrane_potentials, H, noise_seed=2002):
    """
    Generate measurements and add noise.

    Parameters:
    membrane_potentials (array): Membrane potentials.
    H (array): Measurement matrix.
    noise_seed (int): Random seed for noise generation.

    Returns:
    tuple: Measurements and noisy measurements.
    """
    n_stimuli = len(membrane_potentials)
    measurements = np.zeros((n_stimuli, 2))
    measurements[0] = H @ membrane_potentials[0]
    
    for t in range(1, n_stimuli):
        measurements[t] = H @ membrane_potentials[t]

    rng = rnd.default_rng(noise_seed)
    measurements_noisy = measurements + rng.multivariate_normal(mean=np.zeros(2), cov=np.eye(2), size=n_stimuli)
    
    return measurements, measurements_noisy

def generate_synthetic_data(n_stimuli, period_square, W, H, tau, dt, theta, M):
    """
    Generates synthetic data and saves it as a .npy file.

    Parameters:
    W (array): Weight matrix.
    H (array): Measurement matrix.
    n_stimuli (int): Number of stimuli.
    period_square (int): Period of the square wave stimuli.
    tau (float): Time constant.
    dt (float): Time step.
    theta (float): Parameter for sigmoid function.
    M (float): Input scaling factor.
    """
    stimuli = generate_stimuli(n_stimuli, period_square)
    
    membrane_potentials = generate_membrane_potentials(stimuli, tau, dt, M, W, theta, 'linear')

    measurements, measurements_noisy = generate_measurements(membrane_potentials, H)
    
    np.save('synthetic_data.npy', {
        'stimuli': stimuli,
        'membrane_potentials': membrane_potentials,
        'measurements': measurements,
        'measurements_noisy': measurements_noisy,
        'params': {
            'W': W,
            'M': M,
            'H': H,
            'tau': tau,
            'theta': theta
        }
    })

