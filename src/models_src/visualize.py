import numpy as np
import matplotlib.pyplot as plt

def plot_stimulus(stimuli):
    """
    Plot stimuli
    """
    plt.figure("Stimuli")
    plt.plot(stimuli, c='r', label="Stimuli")
    plt.title("Stimuli")
    plt.xlabel("Samples")
    plt.legend()

def plot_state(membrane_potentials, label_1 = "membrane potential 1", label_2 = "membrane potential 2", linestyle="-", figname="Membrane Potentials"):
    """
    Plot state evolution.
    """
    plt.figure(figname)
    plt.plot(membrane_potentials[:, 0], c='#1f77b4', label=label_1, linestyle=linestyle)
    plt.plot(membrane_potentials[:, 1], c='orange', label=label_2, linestyle=linestyle)
    plt.title(figname)
    plt.xlabel("Samples")
    plt.ylabel("Membrane Potential (V)")
    plt.legend()

def plot_measurement(y, figname="Predicted Measurements"):
    """
    Plot predicted measurements.
    """
    plt.figure(figname)
    plt.plot(y[:, 0], c='#1f77b4', label="Predicted Measurement 1")
    plt.plot(y[:, 1], c='orange', label="Predicted Measurement 2")
    plt.title(figname)
    plt.xlabel("Samples")
    plt.ylabel("EEG potential (V)")
    plt.legend()

def plot_norm_squared_error(norm_squared_error, figname="Norm Squared Error"):
    """
    Plot norm squared error.
    """
    plt.figure(figname)
    plt.semilogy(norm_squared_error, label = figname)
    plt.title(figname)
    plt.xlabel("Iteration")
    plt.ylabel("Norm Squared Error")
    plt.legend()

def plot_norm_squared_error_m(norm_squared_errors):
    """
    Plot norm squared error for M parameter.
    """
    plot_norm_squared_error(norm_squared_errors['M'], figname='Norm Squared Error for M')

def plot_norm_squared_error_tau(norm_squared_errors):
    """
    Plot norm squared error for tau parameter.
    """
    plot_norm_squared_error(norm_squared_errors['tau'], figname='Norm Squared Error for tau')

def plot_norm_squared_error_theta(norm_squared_errors):
    """
    Plot norm squared error for theta parameter.
    """
    if 'theta' in norm_squared_errors:
        plot_norm_squared_error(norm_squared_errors['theta'], figname='Norm Squared Error for theta')

def plot_matrix_norm_squared_errors(norm_squared_error, matrix):
    """
    Plot norm squared errors for matrix parameters.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 12), sharex=True)

    for i in range(2):
        for j in range(2):
            key =f'{matrix}_{i}_{j}'
            if matrix == 'W':
                axs[i, j].semilogy(norm_squared_error[key], label=key)
            elif matrix == 'H':
                axs[i, j].semilogy(norm_squared_error[i+j], label=key)
            axs[i, j].set_ylabel('Norm Squared Error')
            axs[i, j].set_title(f'Norm Squared Error for {key}')
            axs[i, j].legend()
    axs[-1, -1].set_xlabel('Iteration')
    plt.tight_layout()


def compare_estimates(predictions, measurements,label_1, label_2, figname):
    plot_state(predictions, "predicted " + label_1, "predicted " + label_2, figname=figname)
    plot_state(measurements, label_1, label_2, linestyle="--", figname=figname)

def plot_measurements_norm_squared_error(norm_squared_error, figname="Norm Squared Error"):
    plt.figure(figname)
    plt.semilogy(norm_squared_error[0], label = figname + " for measurement 1")
    plt.semilogy(norm_squared_error[1], label = figname + " for measurement 2")
    plt.title(figname)
    plt.xlabel("Iteration")
    plt.ylabel("Norm Squared Error")
    plt.legend()

def plot_membrane_potential_norm_squared_error(norm_squared_error, figname="Norm Squared Error"):
    plt.figure(figname)
    plt.semilogy(norm_squared_error[0], label = figname + " for membrane potential 1")
    plt.semilogy(norm_squared_error[1], label = figname + " for membrane potential 2")
    plt.title(figname)
    plt.xlabel("Iteration")
    plt.ylabel("Norm Squared Error")
    plt.legend()

def plot_all(synthetic_data_file='synthetic_data.npy', estimated_params_file='estimated_params.npy'):
    # Load synthetic data
    synthetic_data = np.load(synthetic_data_file, allow_pickle=True).item()
    stimuli = synthetic_data['stimuli']
    membrane_potentials = synthetic_data['membrane_potentials']
    measurements = synthetic_data['measurements']
    measurements_noisy = synthetic_data['measurements_noisy']

    # Load estimated parameters and states
    estimated_params = np.load(estimated_params_file, allow_pickle=True).item()
    membrane_potentials_predicted = estimated_params['membrane_potentials_predicted']
    norm_squared_errors = estimated_params['norm_squared_errors']
    print(norm_squared_errors['membrane_potentials'].T[2:6].shape)
    
    # Plot data
    plot_stimulus(stimuli)
    plot_state(membrane_potentials, figname="Synthetic Membrane Potentials", label_1 = "membrane potential 1", label_2 = "membrane potential 2")
    plot_state(membrane_potentials_predicted, figname="Predicted Membrane Potentials", label_1 = "membrane potential 1", label_2 = "membrane potential 2")

    y = []
    for t in range(len(stimuli)):
        y.append(np.dot(np.array([[1, 0.7], [0.5, 0.8]]), membrane_potentials_predicted[t][0:2]))
    y = np.array(y)
    #plot_measurement(y)

    plot_state(measurements, label_1 = "measurement 1", label_2 = "measurement 2", figname="Synthetic Measurements")
    plot_state(measurements_noisy, label_1 = "measurement 1", label_2 = "measurement 2",  figname="Synthetic Noisy Measurements")


    compare_estimates(y, measurements, label_1 = "measurement 1", label_2 = "measurement 2", figname="Predictions vs Synthetic Measurements")
    compare_estimates(membrane_potentials_predicted, membrane_potentials, label_1 = "membrane potential 1", label_2 = "membrane potential 2", figname="Predicted vs Synthetic Membrane Potentials")

    plot_matrix_norm_squared_errors(norm_squared_errors['W'], 'W')
    plot_norm_squared_error_m(norm_squared_errors)
    plot_norm_squared_error_tau(norm_squared_errors)
    plot_norm_squared_error_theta(norm_squared_errors)

    plot_membrane_potential_norm_squared_error(norm_squared_errors['membrane_potentials'].T, figname="Norm Squared Error Membrane Potential")
    #plot_norm_squared_error(norm_squared_errors['membrane_potentials'][:,2:6], figname="Norm Squared Error h_components")
    plot_matrix_norm_squared_errors(norm_squared_errors['membrane_potentials'].T[2:6], 'H')
    #plot_h_parameter_error(norm_squared_errors, figname="Norm Squared Error h components")

    plot_measurements_norm_squared_error(norm_squared_errors['measurements'].T, figname="Norm Squared Error measurements")

    # Show all plots
    plt.show()

if __name__ == "__main__":
    plot_all()

