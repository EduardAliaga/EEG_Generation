import numpy as np
import numpy.random as rnd
from utils import sigmoid


def generate_stimuli(period_square, total_time, n_stimuli):
    """
    Generate a square wave stimuli.

    Parameters:
    n_stimuli (int): Number of stimuli.
    period_square (int): Period of the square wave stimuli.

    Returns:
    array: Generated stimuli.
    """
    stimulus_time = np.linspace(0, total_time, n_stimuli)
    stimuli = np.zeros_like(stimulus_time)
    for i_stimulus, stimulus_ in enumerate(stimulus_time):
        stimuli[i_stimulus] = (stimulus_  // period_square) % 2

    return stimuli

def generate_membrane_potentials(stimuli, tau, dt, M, W, theta, H, f):
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
    membrane_potentials = np.zeros((n_stimuli, 6))
    membrane_potentials[0] = np.array([-0.07, -0.07, 0, 0, 0, 0])
    membrane_potentials[0][2:6] = H.flatten()
    
    for t in range(1, n_stimuli):
        #membrane_potentials[t-1] = np.array([membrane_potentials[t-1][0], membrane_potentials[t-1][1], H[0,0], H[0,1], H[1,0], H[1,1]])
        x = membrane_potentials[t-1]
        if f == 'sigmoid':
            membrane_potentials[t] = x + dt * (-x / tau + W @ sigmoid(x, theta) + M * stimuli[t-1])
        elif f == 'tanh':
            membrane_potentials[t] = x + dt * (-x / tau + W @ np.tanh(x) + M * stimuli[t-1])
        elif f == 'linear':
            membrane_potentials[t] = x + dt * (-x / tau + W @ x + M * stimuli[t-1])
        membrane_potentials[t][2:6] = H.flatten()
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
    measurements[0] = H @ membrane_potentials[0][0:2]
    membrane_potentials[0][2:6] = H.flatten()
    for t in range(1, n_stimuli):
        measurements[t] = H @ membrane_potentials[t][0:2]

    rng = rnd.default_rng(noise_seed)
    measurements_noisy = measurements + rng.multivariate_normal(mean=np.zeros(2), cov=np.eye(2)*1e-4, size=n_stimuli)
    
    return measurements, measurements_noisy

def generate_synthetic_data(n_stimuli, total_time, period_square, W, H, tau, dt, theta, M, f):
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
    stimuli = generate_stimuli(period_square, total_time, n_stimuli)
    
    membrane_potentials = generate_membrane_potentials(stimuli, tau, dt, M, W, theta, H, f)

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

