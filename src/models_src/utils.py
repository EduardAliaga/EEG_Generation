import numpy as np

def sigmoid(x, theta):
    """
    Applies the sigmoid function with a scaling factor theta.

    Parameters:
    x (float): Input value.
    theta (float): Scaling factor.

    Returns:
    float: Sigmoid of the input.
    """
    return 1 / (1 + np.exp(-x * theta))

def sigmoid_derivative(x, theta, d):
    """
    Calculates the derivative of the sigmoid function.

    Parameters:
    x (float): Input value.
    theta (float): Scaling factor.
    d (str): Derivative type ('d_x' or 'd_theta').

    Returns:
    float: Derivative of the sigmoid.
    """
    sig = sigmoid(x, theta)
    if d == 'd_x':
        return theta * sig * (1 - sig)
    if d == 'd_theta':
        return x * sig * (1 - sig)

def get_norm_squared_error(x, x_hat, regularization_term=1e-4):
    """
    Calculates the normalized squared error.

    Parameters:
    x (array): Actual values.
    x_hat (array): Predicted values.
    regularization_term (float): Regularization term to avoid division by zero.

    Returns:
    array: Normalized squared error.
    """
    squared_error = get_squared_error(x, x_hat)

    norm_sq_err = squared_error / (x + regularization_term)**2
    return norm_sq_err

def get_squared_error(x, x_hat):
    """
    Calculates the squared error.

    Parameters:
    x (array): Actual values.
    x_hat (array): Predicted values.

    Returns:
    array: Squared error.
    """
    return (x - x_hat)**2

def load_synthetic_data(data_file, f):
    """
    Load synthetic data from a .npy file.

    Parameters:
    data_file (str): Path to the .npy file containing the synthetic data.

    Returns:
    dict: Dictionary containing stimuli, measurements, and parameters.
    """
    data = np.load(data_file, allow_pickle=True).item()
    stimuli = data['stimuli']
    membrane_potentials = data['membrane_potentials']
    measurements = data['measurements']
    measurements_noisy = data['measurements_noisy']
    real_params = {
        'W': data['params']['W'],
        'M': data['params']['M'],
        'tau': data['params']['tau'],
        'H' : data['params']['H']
        
    }
    if f == 'sigmoid':
        real_params['theta'] = data['params']['theta']
    return stimuli, membrane_potentials, measurements, measurements_noisy, real_params


def params_to_vector(params_dict):
    W = params_dict['W'].flatten()
    if 'theta' in params_dict:
        return np.hstack((W, params_dict['M'], params_dict['tau'], params_dict['theta']))
    return np.hstack((W, params_dict['M'], params_dict['tau']))

def vector_to_params(aug_state_dim, vector):
    W = vector[:aug_state_dim**2].reshape((aug_state_dim, aug_state_dim))
    M = vector[aug_state_dim**2]
    tau = vector[aug_state_dim**2 + 1]
    params = {'W': W, 'M': M, 'tau': tau}
    if len(vector) > aug_state_dim**2 + 2:
        params['theta'] = vector[aug_state_dim**2 + 2]
    return params

def get_params_norm_squared_error(norm_squared_errors, params_dict, real_params):
        
    for key in params_dict:
        norm_squared_errors[key] = []
        if key == 'W':
            norm_squared_errors['W'] = {f'W_{i}_{j}': [] for i in range(params_dict['W'].shape[0]) for j in range(params_dict['W'].shape[1])}
            for i in range(params_dict['W'].shape[0]):
                for j in range(params_dict['W'].shape[1]):
                    norm_error = get_norm_squared_error(real_params['W'][i, j], params_dict['W'][i, j])
                    norm_squared_errors['W'][f'W_{i}_{j}'].append(norm_error)
        else:
            norm_error = get_norm_squared_error(real_params[key], params_dict[key])
            norm_squared_errors[key].append(norm_error)
    return norm_squared_errors

def get_predictions_norm_squared_error(membrane_potentials_predicted, measurements_predicted, membrane_potentials, measurements, state_dim):
    nsqe_membrane_potentials = get_norm_squared_error(membrane_potentials, membrane_potentials_predicted)
    # Look how mesaurements prediced is computed, should we keep using that or change it for the predictions from the beginning?
    y, H = compute_measurements_with_known_H(state_dim, membrane_potentials_predicted)
    nsqe_measurements = get_norm_squared_error(measurements, y)
    return nsqe_membrane_potentials, nsqe_measurements

def get_norm_squared_errors(membrane_potentials_predicted, measurements_predicted, membrane_potentials, measurements, params_dict, real_params, state_dim):
    norm_squared_errors = {}
    norm_squared_errors = get_params_norm_squared_error(norm_squared_errors, params_dict, real_params)
    norm_squared_errors['membrane_potentials'], norm_squared_errors['measurements'] = get_predictions_norm_squared_error(membrane_potentials_predicted, measurements_predicted, membrane_potentials, measurements, state_dim)
    return norm_squared_errors


def compute_measurements_with_known_H(state_dim,membrane_potentials_predicted):
    y = []
    x = membrane_potentials_predicted[-1]
    H = x[state_dim-1:-1].reshape((state_dim, state_dim))
    for t in range(0,len(membrane_potentials_predicted)):
        y.append(H @ membrane_potentials_predicted[t][:2])
    y = np.array(y)
    return y, H

def save_results(params_dict, membrane_potentials_predicted, norm_squared_errors):
    save_dict = {}
    for key in params_dict:
        save_dict[key] = params_dict[key]
    save_dict['membrane_potentials_predicted'] = membrane_potentials_predicted
    save_dict['norm_squared_errors'] = norm_squared_errors
    np.save('estimated_params.npy', save_dict)

