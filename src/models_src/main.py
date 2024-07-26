import numpy as np
from generate_data import generate_synthetic_data
from visualize import plot_all
from linear_model import LinearModel
from sigmoid_model import SigmoidModel
from dcm_model import DCM
from utils import *
import jax.numpy as jnp
from jax import grad
from jax import jacobian
import matplotlib.pyplot as plt

def main():
    total_time = 30
    dt = 1e-3
    n_time_points = int(total_time / dt)
    period_square = 0.3
    sources = 2
    model = 'dcm'
    #params for dcm model: state_dim, aug_state_dim, sources, H, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b
    params_x = {
        'state_dim': 9,
        'aug_state_dim': 11,
        'sources' : 2,
        'H': np.array([[1, 0.7], [0.5, 0.8]]),
        'dt': dt,
        'theta': 1.0,
        'H_e': 0.1,
        'tau_e': 10.0,
        'H_i': 16.0,
        'tau_i': 32.0,
        'gamma_1': 1.0,
        'gamma_2': 4/5,
        'gamma_3': 1/4,
        'gamma_4': 1/4,  # gamma_3 value
        'theta': 0.56,
        'sources': 2,
        'C_f': np.random.rand(sources, sources),
        'C_l': np.random.rand(sources, sources), 
        'C_u': np.random.rand(sources),
        'C_b': np.random.rand(sources, sources), 
    }
    params_y = {
        'H': params_x['H'],
        'state_dim': params_x['state_dim'],
        'aug_state_dim': params_x['aug_state_dim'],
        'sources': params_x['sources']
    }
    generate_synthetic_data(period_square, total_time, n_time_points, params_x, params_y, model)
    data_file='synthetic_data.npy'
    #stimuli, states, measurements, measurements_noisy, real_params = load_synthetic_data(data_file, f)
    data = np.load(data_file, allow_pickle=True).item()
    stimuli = data['stimuli']
    states = data['states']
    states = np.array(states)
    print(states.shape)
    measurements = data['measurements']
    measurements_noisy = data['measurements_noisy']
    C_f = np.random.rand(sources, sources)
    C_l = np.random.rand(sources, sources)
    C_u = np.random.rand(sources)
    C_b = np.random.rand(sources, sources)
    model = DCM(C_f, C_l, C_u, C_b,)
    states_predicted, measurements_predicted = model.fit(stimuli, measurements_noisy)

    H = states_predicted[-1, 9:11, :]
    y = []
    for t in range(0,len(states_predicted)):
        x0 = states_predicted[t, 2, :] - states_predicted[t, 3, :]
        y.append(H @ x0)
    nsqe_measurements = get_norm_squared_error(measurements, y)


    #norm_squared_errors =  get_norm_squared_errors(states_predicted, measurements_predicted, states, measurements, model.params_dict, real_params, model.state_dim)

    #save_results(model.params_dict, states_predicted, norm_squared_errors)

    #estimate_parameters_and_states(f)
    for i in range(0,2):
        plt.figure()
        plt.plot(measurements[:,i])
        plt.plot(measurements_predicted[:,i])
    plt.show()

    # for i in range(0,9):
    #     plt.figure()
    #     plt.plot(states[i, 0,:])
    #     plt.plot(states_predicted[:,i, 0])
    # plt.show()
    #plot_all()

if __name__ == "__main__":
    main()
