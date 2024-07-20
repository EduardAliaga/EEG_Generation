import numpy as np
import sys
sys.path.insert(0, '../')
from src.utils import  get_norm_squared_error, f_o, jacobian_f_o_x, jacobian_f_o, jacobian_h

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
    }
    H = data['params']['H']
    if f == 'sigmoid':
        real_params['theta'] = data['params']['theta']
    return stimuli, membrane_potentials, measurements, measurements_noisy, real_params, H

def initialize_params(dim_latent, f):
    """
    Initialize parameters for the recursive update process.
    """
    W_init = np.zeros((dim_latent,dim_latent))
    W_init[0,0] = 1
    W_init[1,1] = 1 
    M_init = 0.5
    tau_init = 80.0
    theta_init = 0.5
    H_init = np.eye(2)
    params_dict = {
        'W': W_init,
        'M': M_init,
        'H' : H_init,
        'tau': tau_init,
    }
    if f == 'sigmoid':
        params_dict['theta'] = theta_init
        
    n_params = dim_latent**2 + len(params_dict)-2
    Q_x = np.eye(dim_latent) * 1e-6
    R_y = np.eye(2) * 1e-6
    P_x_ = np.eye(dim_latent) * 1e-6
    P_x = np.eye(dim_latent) * 1e-6
    P_params_ = np.eye(n_params) * 1e-6
    P_params = np.eye(n_params) * 1e-6
    Q_params = np.eye(n_params) * 1e-6
    
    return params_dict, Q_x, R_y, P_x_, P_x, P_params_, P_params, Q_params

def params_to_vector(params):
    """
    Convert parameters dictionary to a vector.

    Parameters:
    params (dict): Parameters dictionary.

    Returns:
    array: Parameters vector.
    """
    W = params['W'].flatten()
    
    if 'theta' in params.keys():
        params_vec = np.hstack((W, params['M'], params['tau'], params['theta']))
    else:
        params_vec = np.hstack((W, params['M'], params['tau']))
    return params_vec

def vector_to_params(vector, dim_latent):
    """
    Convert parameters vector to a dictionary.

    Parameters:
    vector (array): Parameters vector.
    dim_latent (int): Dimension of the latent state.

    Returns:
    dict: Parameters dictionary.
    """
    W = vector[:dim_latent**2].reshape((dim_latent, dim_latent))
    M = vector[dim_latent**2]
    tau = vector[dim_latent**2 + 1]
    params_dict = {'W': W, 'M': M, 'tau': tau}
    if len(vector) > dim_latent**2 + 2:   
        theta = vector[dim_latent**2 + 2]
        params_dict['theta'] = theta

    return params_dict

def update_params(params_dict, params_vec, x, u, t, dt, F_x, F_params, P_x_, P_x, P_params_, P_params, Q_x, Q_params, R_y, dH, y_hat, y):
    """
    Perform parameter updates during the recursive process.

    Parameters:
    params_vec (array): Current parameters vector.
    x (array): Current state.
    u (array): Input stimuli.
    t (int): Current time step.
    dt (float): Time step.
    F_x (array): Jacobian of the state.
    F_params (array): Jacobian of the parameters.
    P_x_ (array): Predicted state covariance.
    P_x (array): State covariance.
    P_params_ (array): Predicted parameter covariance.
    P_params (array): Parameter covariance.
    Q_x (array): Process noise covariance for the state.
    Q_params (array): Process noise covariance for the parameters.
    R_y (array): Measurement noise covariance.
    H (array): Measurement matrix.
    y_hat (array): Predicted measurements.
    y (array): Actual measurements.

    Returns:
    tuple: Updated state, parameters vector, and covariances.
    """
    
    S = dH @ P_x_ @ dH.T + R_y 
    S_inv = np.linalg.inv(S)
    x_hat = f_o(x, u[t-1], params_dict, dt, 'sigmoid') - P_x_ @ dH.T @ S_inv @ (y_hat - y[t])

    I = np.eye(len(x))
    P_x_ = F_x @ P_x @ F_x.T + Q_x

    P_x = P_x_ @ (I + dH.T @ (R_y - dH @ P_x_ @ dH.T) @ dH @ P_x_)
   
    params_vec = params_vec - P_params_ @ F_params.T @ (f_o(x, u[t-1], params_dict, dt, 'sigmoid') - x_hat)
    
    P_params_ = P_params - P_params @ F_params.T @ (Q_x + F_params @ P_params @ F_params.T) @ F_params @ P_params
    P_params = P_params_ + Q_params

    return x_hat, P_x_, P_x, params_vec, P_params_, P_params

def recursive_update(x, params, H, u, y, dt, P_x_, P_x, Q_x, P_params_, P_params, Q_params, R_y, function, num_iterations, real_params):
    """
    Perform a recursive update for estimating parameters and states.

    Parameters:
    x (array): Initial state.
    params (dict): Initial parameters.
    H (array): Measurement matrix.
    u (array): Input stimuli.
    y (array): Noisy measurements.
    dt (float): Time step.
    P_x_ (array): Predicted state covariance.
    P_x (array): State covariance.
    Q_x (array): Process noise covariance for the state.
    P_params_ (array): Predicted parameter covariance.
    P_params (array): Parameter covariance.
    Q_params (array): Process noise covariance for the parameters.
    R_y (array): Measurement noise covariance.
    function (str): Nonlinearity function type.
    num_iterations (int): Number of iterations.
    real_params (dict): Real parameters for error calculation.

    Returns:
    tuple: Final state, predicted membrane potentials, updated parameters, and norm squared errors.
    """
    membrane_potentials_predicted = []
    dim_latent = 6
    state_dim = 2
    params_vec = params_to_vector(params)
    norm_squared_errors = {key: [] for key in params.keys()}
    norm_squared_errors['W'] = {f'W_{i}_{j}': [] for i in range(params['W'].shape[0]) for j in range(params['W'].shape[1])}

    membrane_potentials_predicted.append(x)
    for t in range(1, num_iterations):
        params_dict = vector_to_params(params_vec, dim_latent)

        F_x = jacobian_f_o_x(x, params_dict, dt, function)
        F_params = jacobian_f_o(x, u[t-1], params_dict, dt, function)
        # TODO: Check this!
        # x_hat = f_o(x, u[t-1], params_dict, dt, function)
        H = x[2:6].reshape((state_dim,state_dim))

        dH = jacobian_h(x, state_dim)
        y_hat = H @ x[:state_dim]
        x, P_x_, P_x, params_vec, P_params_, P_params = update_params(params_dict, params_vec, x, u, t, dt, F_x, F_params, P_x_, P_x, P_params_, P_params, Q_x, Q_params, R_y, dH, y_hat, y)
        Q_x = dt * Q_x
        Q_params = dt * Q_params
    
        membrane_potentials_predicted.append(x)
        
        # Calculate norm squared errors for each parameter
        # TODO: Under construction.
        for key in params_dict:
            if key == 'W':
                for i in range(params_dict['W'].shape[0]):
                    for j in range(params_dict['W'].shape[1]):
                        norm_error = get_norm_squared_error(real_params['W'][i, j], params_dict['W'][i, j])
                        norm_squared_errors['W'][f'W_{i}_{j}'].append(norm_error)
            else:
                norm_error = get_norm_squared_error(real_params[key], params_dict[key])
                norm_squared_errors[key].append(norm_error)
    params = vector_to_params(params_vec, dim_latent)
    return x, np.array(membrane_potentials_predicted), params, P_x_, P_x, P_params_, P_params, norm_squared_errors

def estimate_parameters_and_states(f, data_file='synthetic_data.npy'):
    """
    Estimate parameters and states from synthetic data.

    Parameters:
    data_file (str): Path to the .npy file containing the synthetic data.
    """
    
    stimuli, membrane_potentials, measurements, measurements_noisy, real_params, H = load_synthetic_data(data_file, f)
    n_stimuli = len(stimuli)
    dt = 1e-4
    x = np.zeros(6)
    x[0:2] = membrane_potentials[0][0:2]
    x[2] = 1
    x[5] = 1
    params, Q_x, R_y, P_x_, P_x, P_params_, P_params, Q_params = initialize_params(len(x), f)

    x, membrane_potentials_predicted, params, P_x_, P_x, P_params_, P_params, norm_squared_errors = recursive_update(
        x, params, params['H'], stimuli, measurements_noisy, dt, P_x_, P_x, Q_x, P_params_, P_params, Q_params, R_y, f, n_stimuli, real_params)
    norm_squared_errors['membrane_potentials'] = get_norm_squared_error(membrane_potentials, membrane_potentials_predicted)
    y = []
    state_dim = 2
    H = x[state_dim-1:-1].reshape((state_dim,state_dim))
    for t in range(0, n_stimuli):
        y.append(H @ membrane_potentials_predicted[t][:2])
    y = np.array(y)
    norm_squared_errors['measurements'] = get_norm_squared_error(measurements, y)
    save_dict = {
        'W': params['W'],
        'M': params['M'],
        'H': H,
        'tau': params['tau'],
        'membrane_potentials_predicted': membrane_potentials_predicted,
        'norm_squared_errors': norm_squared_errors
    }

    if 'theta' in params:
        save_dict['theta'] = params['theta']

    np.save('estimated_params.npy', save_dict)


