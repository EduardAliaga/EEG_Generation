import numpy as np
from scipy.integrate import odeint
from src.utils import membrane_potential_equation, eeg_linear_model_computation

def generate_synthetic_membrane_potentials(tau, theta, Cmat, linear_model_weights, stimulis, x0, t):

    synthetic_membrane_potentials = []
    synthetic_time_series_eeg = []

    x = odeint(membrane_potential_equation, x0, t, args=(tau, Cmat, theta, stimulis))
    synthetic_membrane_potentials.append(x)

    VE = eeg_linear_model_computation(x, linear_model_weights)
    synthetic_time_series_eeg.append(VE)

    synthetic_membrane_potentials = np.array(synthetic_membrane_potentials)
    synthetic_time_series_eeg = np.array(synthetic_time_series_eeg)

    return synthetic_membrane_potentials, synthetic_time_series_eeg

