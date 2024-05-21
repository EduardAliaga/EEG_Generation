import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import argparse
from prepare_data import read_subject_data, compute_stimuli

def sigma(x, theta):
    return 1 / (1 + np.exp(-theta * x))

def model_x(x, t, tau, w, theta, u):
    dxdt = np.zeros_like(x)
    N = len(x)
    time_index = min(int(t), len(u) - 1)  # Ensure index is within bounds
    for i in range(N):
        dxdt[i] = -x[i] / tau + np.sum(w[i, :] * sigma(x, theta)) + u[time_index]
    return dxdt

def model_VE(x, matrix):
    return x @ matrix

def generate_synthetic_data(tau, theta, Cmat, linear_model_weights, stimulis, x0, t):
    # Time vector
    synthetic_membrane_potentials = []
    synthetic_time_series_eeg = []
    x = odeint(model_x, x0, t, args=(tau, Cmat, theta, stimulis))
    synthetic_membrane_potentials.append(x)
    VE = model_VE(x, linear_model_weights)
    synthetic_time_series_eeg.append(VE)

    synthetic_membrane_potentials = np.array(synthetic_membrane_potentials)
    synthetic_time_series_eeg = np.array(synthetic_time_series_eeg)
    return synthetic_membrane_potentials, synthetic_time_series_eeg

"""def main(data_path, sensor):
    # Load input signal
    inquiries, timings, labels = read_subject_data(data_path, sensor)
    stimulis = compute_stimuli(timings, labels)  # Ensure this is your squared signal input of size (100, 3, 792)
    tau = 10.0  # Time constant
    theta = 1.0  # Sigmoid function parameter
    neurons = 3
    Cmat = np.random.randn(neurons, neurons)  # Weight matrix for model x
    linear_model_weights = np.random.randn(neurons)  # Matrix M for model VE

    # Initial conditions
    x0 = -70e-3 * np.ones(neurons)
    synthetic_membrane_potentials, synthetic_time_series_eeg = generate_synthetic_data(tau, theta, Cmat, linear_model_weights, stimulis, x0)

    np.save('synthetic_membrane_potentials.npy', synthetic_membrane_potentials)
    np.save('synthetic_time_series_eeg.npy', synthetic_time_series_eeg)
    print("Synthetic data saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic neural data based on given parameters and input signals.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--sensor', type=str, required=True, help='List of sensors')
    args = parser.parse_args()

    main(args.data_path, args.sensor)
"""