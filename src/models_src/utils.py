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
    x_clipped = np.clip(x * theta, -100, 100)
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

def f_o(x, u, params_dict, dt, function):
    """
    Computes the function value based on the specified nonlinearity.

    Parameters:
    x (array): Current state.
    u (array): Input.
    theta (float): Parameter for sigmoid function.
    W (array): Weight matrix.
    M (float): Input scaling factor.
    tau (float): Time constant.
    dt (float): Time step.
    function (str): Nonlinearity type ('sigmoid', 'tanh', 'linear').

    Returns:
    array: Updated state.
    """
    if function == 'sigmoid':
        x[0:2] = x[0:2] + dt * (-x[0:2] / params_dict['tau'] + params_dict['W'][0:2] @ sigmoid(x, params_dict['theta']) + params_dict['M'] * u)
    elif function == 'tanh':
        return x + dt * (-x / params_dict['tau'] + params_dict['W'] @ np.tanh(x) + params_dict['M'] * u)
    elif function == 'linear':
        # TODO: Issue.
        x[0:2] = x[0:2] + dt * (-x[0:2] / params_dict['tau'] + params_dict['W'][0:2] @ x + params_dict['M'] * u)
    return x

def g_o(param):
    """
    Placeholder function for parameter update (identity function).

    Parameters:
    param (array): Parameters to be updated.

    Returns:
    array: Updated parameters (same as input).
    """
    return param

def jacobian_f_o_x(x, params_dict, dt, function):
    """
    Computes the Jacobian of f_o with respect to x.

    Parameters:
    x (array): Current state.
    theta (float): Parameter for sigmoid function.
    W (array): Weight matrix.
    tau (float): Time constant.
    dt (float): Time step.
    function (str): Nonlinearity type ('sigmoid', 'tanh', 'linear').

    Returns:
    array: Jacobian matrix.
    """
    dim_latent = len(x)
    if function == 'sigmoid':
        fx = sigmoid(x, params_dict['theta'])
        diag_matrix = np.diag(sigmoid_derivative(x, params_dict['theta'], 'd_x'))
    elif function == 'tanh':
        fx = np.tanh(x)
        diag_matrix = np.diag(1 - fx ** 2)
    elif function == 'linear':
        return (1 - dt / params_dict['tau']) * np.eye(dim_latent) + dt * params_dict['W']
    return np.eye(dim_latent) + dt * (-1 / params_dict['tau'] * np.eye(len(x)) + params_dict['W'] @ diag_matrix)

def jacobian_f_o_W(x, params_dict, dt, function):
    """
    Computes the Jacobian of f_o with respect to W.

    Parameters:
    x (array): Current state.
    theta (float): Parameter for sigmoid function.
    tau (float): Time constant.
    dt (float): Time step.
    function (str): Nonlinearity type ('sigmoid', 'linear').

    Returns:
    array: Jacobian matrix.
    """
    F_W = np.zeros((6, 36))
    if function == 'sigmoid':
        F_W[0,0:2] = [dt * sigmoid(x[0], params_dict['theta']), dt * sigmoid(x[1], params_dict['theta'])]
        F_W[1,6:8] = [dt * sigmoid(x[0], params_dict['theta']), dt * sigmoid(x[1], params_dict['theta'])]
    elif function == 'linear':
        F_W[0, 0:2] = dt * np.array([x[0], x[1]])
        F_W[1, 6:8] = dt * np.array([x[0], x[1]])

    return F_W

def jacobian_f_o_M(u, dt):
    """
    Computes the Jacobian of f_o with respect to M.

    Parameters:
    x (array): Current state.
    theta (float): Parameter for sigmoid function.
    u (array): Input.
    dt (float): Time step.

    Returns:
    array: Jacobian matrix.
    """
    return dt * u

def jacobian_f_o_tau(x, params_dict, dt):
    """
    Computes the Jacobian of f_o with respect to tau.

    Parameters:
    x (array): Current state.
    theta (float): Parameter for sigmoid function.
    W (array): Weight matrix.
    tau (float): Time constant.
    dt (float): Time step.

    Returns:
    array: Jacobian matrix.
    """
    return dt * x[0:2] / params_dict['tau']**2

def jacobian_f_o_theta(x, params_dict, dt):
    """
    Computes the Jacobian of f_o with respect to theta.

    Parameters:
    x (array): Current state.
    theta (float): Parameter for sigmoid function.
    W (array): Weight matrix.
    tau (float): Time constant.
    dt (float): Time step.
    function (str): Nonlinearity type ('sigmoid').

    Returns:
    array: Jacobian matrix.
    """
    derivative = np.array([sigmoid_derivative(x, params_dict['theta'], 'd_theta')])
    return dt * (params_dict['W'] @ derivative.T)

def jacobian_h(x, state_dim):
    dH = np.zeros((2,6))
    dH[0,0] = x[state_dim]
    dH[0,1] = x[state_dim+1]
    dH[0,2] = x[0]
    dH[0,3] = x[1]
    dH[1,0] = x[state_dim+2]
    dH[1,1] = x[state_dim+3]
    dH[1,4] = x[0]
    dH[1,5] = x[1]
    return dH

def jacobian_f_o(x, u, params_dict, dt, function):
    """
    Computes the combined Jacobian of f_o with respect to all parameters.

    Parameters:
    x (array): Current state.
    u (array): Input.
    theta (float): Parameter for sigmoid function.
    W (array): Weight matrix.
    M (float): Input scaling factor.
    tau (float): Time constant.
    dt (float): Time step.
    function (str): Nonlinearity type ('sigmoid', 'tanh', 'linear').

    Returns:
    array: Combined Jacobian matrix.
    """
    F_W = jacobian_f_o_W(x, params_dict, dt, function)
    F_M = jacobian_f_o_M(u, dt)
        
    F_M_array = np.zeros((6,1))
    F_M_array[0:2] = F_M
    F_tau = jacobian_f_o_tau(x, params_dict, dt).reshape(2,1)
    F_tau_array = np.zeros((6,1))
    F_tau_array[0:2] = F_tau
    if function == 'sigmoid':
        F_theta = jacobian_f_o_theta(x, params_dict, dt)
        J_combined = np.hstack((F_W, F_M_array, F_tau_array, F_theta))
    else:
        J_combined = np.hstack((F_W, F_M_array, F_tau_array))
    return J_combined


# if __name__ == "__main__":
#     omega = 2 * np.pi * 10
#     t = np.linspace(0, 1, 10)
#     x = np.sin(omega * t)
#     noise = np.random.default_rng(seed=1714).normal(loc=0, scale=1)
#     x_hat = x + noise
#     norm_squared_error = get_norm_squared_error(x, x_hat)
#     test_ = ((x - x_hat) / (x + 1e-6)) ** 2
#     diff = test_ - norm_squared_error
#     print(f"diff {diff}")
